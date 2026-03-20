import argparse
import os
import shutil
import urllib.request
import zipfile
import json
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO
import ultralytics
 
# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "datasets" / "vehicles"
RUNS_DIR    = PROJECT_DIR / "runs"
BEST_PT     = RUNS_DIR / "detect" / "vehicle_model" / "weights" / "best.pt"
LAST_PT     = RUNS_DIR / "detect" / "vehicle_model" / "weights" / "last.pt"
 
# ---------------------------------------------------------------------------
# Vehicle classes
# ---------------------------------------------------------------------------
VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle", "van", "suv", "ambulance"]
 
# COCO128 ships with ultralytics. Its class IDs (0-indexed) for vehicles:
# 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
# van/suv/ambulance are not in COCO so we remap truck->van/suv when possible
COCO_VEHICLE_IDS = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
 
# Our final class ID mapping
CLASS_ID = {name: i for i, name in enumerate(VEHICLE_CLASSES)}
 
# ---------------------------------------------------------------------------
# Speed presets
# ---------------------------------------------------------------------------
PRESETS = {
    # name : (model,       epochs, imgsz, batch, patience)
    "fast":  ("yolov8n.pt", 30,    416,   16,    10),
    "normal":("yolov8n.pt", 80,    640,   8,     20),
    "accurate":("yolov8m.pt",150,  640,   -1,    40),
}
 
# ---------------------------------------------------------------------------
# Fixed high-quality augmentation settings (same across all presets)
# ---------------------------------------------------------------------------
AUG = {
    "optimizer":       "AdamW",
    "lr0":             0.001,
    "lrf":             0.01,
    "momentum":        0.937,
    "weight_decay":    0.0005,
    "warmup_epochs":   3,
    "warmup_momentum": 0.8,
    "warmup_bias_lr":  0.1,
    "cos_lr":          True,
    "box":             7.5,
    "cls":             0.5,
    "dfl":             1.5,
    "mosaic":          1.0,
    "mixup":           0.1,
    "copy_paste":      0.1,
    "close_mosaic":    5,
    "flipud":          0.1,
    "fliplr":          0.5,
    "hsv_h":           0.015,
    "hsv_s":           0.7,
    "hsv_v":           0.4,
    "translate":       0.1,
    "scale":           0.5,
    "shear":           1.0,
    "perspective":     0.0,
    "degrees":         5.0,
    "label_smoothing": 0.1,
    "amp":             True,
    "overlap_mask":    True,
    "multi_scale":     False,   # off by default for speed; --accurate enables it
    "plots":           True,
    "verbose":         True,
}
 
 
# ===========================================================================
# Device
# ===========================================================================
def get_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"[GPU] {name}  ({vram:.1f} GB VRAM)\n")
        return "0"
    print("[CPU] No GPU detected -- using CPU (slow).\n")
    return "cpu"
 
 
# ===========================================================================
# Dataset  --  filter COCO128 vehicle images only
# ===========================================================================
 
def get_dataset():
    """
    Build a vehicle-only dataset from COCO128 (tiny, ~7 MB download).
    Returns path to data.yaml.
    """
    yaml_out  = DATASET_DIR / "data.yaml"
    train_dir = DATASET_DIR / "images" / "train"
    val_dir   = DATASET_DIR / "images" / "val"
 
    # Already built?
    if yaml_out.exists() and train_dir.exists() and _count_files(train_dir) > 0:
        n = _count_files(train_dir)
        v = _count_files(val_dir)
        print(f"[OK] Vehicle dataset ready: {n} train / {v} val images\n")
        return _fix_yaml(yaml_out)
 
    print("[>>] Building vehicle-only dataset from COCO128 (fast, ~7 MB)...")
 
    # Locate COCO128 inside ultralytics package (already on disk after pip install)
    ul_dir    = Path(ultralytics.__file__).resolve().parent
    coco128_candidates = list(ul_dir.rglob("coco128")) + \
                         list((Path.home() / "ultralytics").rglob("coco128")) + \
                         list((Path.home() / ".cache" / "ultralytics").rglob("coco128"))
 
    coco128_root = None
    for c in coco128_candidates:
        if c.is_dir():
            coco128_root = c
            break
 
    # If not found locally, let ultralytics download it (~7 MB)
    if coco128_root is None or not (coco128_root / "images" / "train2017").exists():
        print("[>>] COCO128 not cached yet -- downloading via ultralytics (~7 MB)...")
        _download_coco128()
        # Search again
        for c in [Path.home() / "ultralytics" / "assets" / "coco128",
                  Path.home() / "ultralytics" / "coco128",
                  Path.home() / ".cache" / "ultralytics" / "coco128",
                  PROJECT_DIR / "coco128"]:
            if c.exists() and (c / "images").exists():
                coco128_root = c
                break
 
    if coco128_root is None:
        print("[!!] Could not locate COCO128. Falling back to scratch scaffold.")
        _scaffold_empty()
        return _fix_yaml(yaml_out)
 
    print(f"[OK] COCO128 found at {coco128_root}")
 
    # Parse COCO128 labels and copy vehicle images only
    labels_dir = coco128_root / "labels" / "train2017"
    images_dir = coco128_root / "images" / "train2017"
 
    if not labels_dir.exists() or not images_dir.exists():
        print("[!!] COCO128 structure unexpected. Falling back to scaffold.")
        _scaffold_empty()
        return _fix_yaml(yaml_out)
 
    vehicle_items = []   # list of (img_path, [(our_cls_id, cx, cy, w, h), ...])
 
    for lbl_file in sorted(labels_dir.glob("*.txt")):
        img_path = images_dir / (lbl_file.stem + ".jpg")
        if not img_path.exists():
            img_path = images_dir / (lbl_file.stem + ".png")
        if not img_path.exists():
            continue
 
        vehicle_boxes = []
        with open(lbl_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                coco_cls = int(parts[0])
                if coco_cls not in COCO_VEHICLE_IDS:
                    continue
                our_name = COCO_VEHICLE_IDS[coco_cls]
                our_id   = CLASS_ID[our_name]
                cx, cy, w, h = map(float, parts[1:5])
                if w < 0.005 or h < 0.005:
                    continue
                vehicle_boxes.append((our_id, cx, cy, w, h))
 
        if vehicle_boxes:
            vehicle_items.append((img_path, vehicle_boxes))
 
    print(f"[OK] {len(vehicle_items)} vehicle images found in COCO128")
 
    if len(vehicle_items) == 0:
        print("[!!] No vehicle images found. Creating empty scaffold.")
        _scaffold_empty()
        return _fix_yaml(yaml_out)
 
    # Split 85% train / 15% val
    split = max(1, int(len(vehicle_items) * 0.85))
    splits = {
        "train": vehicle_items[:split],
        "val":   vehicle_items[split:],
    }
 
    for sp, items in splits.items():
        img_out = DATASET_DIR / "images" / sp
        lbl_out = DATASET_DIR / "labels" / sp
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
 
        for img_path, boxes in items:
            # Copy image
            shutil.copy2(str(img_path), str(img_out / img_path.name))
            # Write label
            lbl_path = lbl_out / (img_path.stem + ".txt")
            with open(lbl_path, "w") as f:
                for cls_id, cx, cy, w, h in boxes:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
 
        print(f"[OK] {sp}: {len(items)} images written")
 
    _write_data_yaml()
    print(f"\n[OK] Vehicle dataset ready at {DATASET_DIR}\n")
    return _fix_yaml(yaml_out)
 
 
def _download_coco128():
    """Trigger ultralytics to download COCO128 by running a tiny validation."""
    try:
        tmp = YOLO("yolov8n.pt")
        tmp.val(data="coco128.yaml", imgsz=32, batch=1, verbose=False, plots=False)
    except Exception:
        pass   # we only need the download side effect
 
 
def _count_files(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for _ in directory.iterdir())
 
 
def _scaffold_empty():
    """Create empty dataset structure with a placeholder so YOLO doesn't crash."""
    for sp in ("train", "val"):
        (DATASET_DIR / "images" / sp).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / sp).mkdir(parents=True, exist_ok=True)
    _write_data_yaml()
    print("[!!] Empty scaffold created.")
    print(f"     Add images to: {DATASET_DIR / 'images' / 'train'}")
    print(f"     Add labels to: {DATASET_DIR / 'labels' / 'train'}")
    print("     Label format: <class_id> <cx> <cy> <w> <h>  (normalised)\n")
 
 
def _write_data_yaml():
    cfg = {
        "path":  DATASET_DIR.resolve().as_posix(),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(VEHICLE_CLASSES),
        "names": {i: n for i, n in enumerate(VEHICLE_CLASSES)},
    }
    with open(DATASET_DIR / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
 
 
def _fix_yaml(yaml_path: Path) -> Path:
    """Guarantee absolute Windows-safe path in data.yaml."""
    yaml_path = yaml_path.resolve()
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
 
    cfg["path"] = DATASET_DIR.resolve().as_posix()
    if not cfg.get("names"):
        cfg["nc"]    = len(VEHICLE_CLASSES)
        cfg["names"] = {i: n for i, n in enumerate(VEHICLE_CLASSES)}
 
    for sp in ("train", "val"):
        if sp in cfg:
            (DATASET_DIR / cfg[sp]).mkdir(parents=True, exist_ok=True)
 
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
 
    print(f"[OK] data.yaml  path={cfg['path']}")
    print(f"     train={cfg.get('train')}  val={cfg.get('val')}  nc={cfg['nc']}")
    names_list = list(cfg["names"].values()) if isinstance(cfg["names"], dict) else cfg["names"]
    print(f"     names={names_list}\n")
    return yaml_path
 
 
# ===========================================================================
# Training
# ===========================================================================
 
def train(args):
    device = get_device()
    data_yaml = get_dataset()
 
    # Pick preset
    if args.fast:
        preset = "fast"
    elif args.accurate:
        preset = "accurate"
    else:
        preset = "normal"
 
    model_name, epochs, imgsz, batch, patience = PRESETS[preset]
 
    # CLI overrides
    if args.model:   model_name = args.model
    if args.epochs:  epochs     = args.epochs
    if args.imgsz:   imgsz      = args.imgsz
 
    cfg = dict(AUG)
    cfg["patience"] = patience
 
    # CPU adjustments
    if device == "cpu":
        batch          = min(batch, 4) if batch > 0 else 4
        cfg["amp"]     = False
        cfg["workers"] = 0
        if preset == "accurate":
            cfg["multi_scale"] = False
        print("[!!] CPU mode: batch=4, AMP off.\n")
 
    if preset == "accurate":
        cfg["multi_scale"] = True
        cfg["mixup"]       = 0.15
        cfg["copy_paste"]  = 0.3
 
    # Load model
    if args.resume and LAST_PT.exists():
        model = YOLO(str(LAST_PT))
        print(f"[>>] Resuming from {LAST_PT}\n")
    else:
        model = YOLO(model_name)
        print(f"[>>] Transfer learning from {model_name}\n")
 
    # Print summary
    sep = "=" * 62
    gpu_label = torch.cuda.get_device_name(0) if device != "cpu" else "CPU"
    est = {"fast": "~5 min", "normal": "~20 min", "accurate": "~2 hrs"}
    print(sep)
    print(f"  VEHICLE-ONLY TRAINING  [{preset.upper()} mode]  est. {est[preset]} on GPU")
    print(sep)
    print(f"  Classes    : {VEHICLE_CLASSES}")
    print(f"  Model      : {model_name}")
    print(f"  Device     : {gpu_label}")
    print(f"  Epochs     : {epochs}  (early-stop patience={patience})")
    print(f"  Image size : {imgsz} px")
    print(f"  Batch      : {'auto' if batch == -1 else batch}")
    print(f"  Dataset    : {data_yaml}")
    print(sep + "\n")
 
    model.train(
        data            = str(data_yaml),
        device          = device,
        project         = str(RUNS_DIR / "detect"),
        name            = "vehicle_model",
        exist_ok        = True,
        pretrained      = True,
        epochs          = epochs,
        imgsz           = imgsz,
        batch           = batch,
        workers         = 0 if device == "cpu" else 4,
        save            = True,
        save_period     = max(5, epochs // 10),
        patience        = patience,
        optimizer       = cfg["optimizer"],
        lr0             = cfg["lr0"],
        lrf             = cfg["lrf"],
        momentum        = cfg["momentum"],
        weight_decay    = cfg["weight_decay"],
        warmup_epochs   = cfg["warmup_epochs"],
        warmup_momentum = cfg["warmup_momentum"],
        warmup_bias_lr  = cfg["warmup_bias_lr"],
        cos_lr          = cfg["cos_lr"],
        box             = cfg["box"],
        cls             = cfg["cls"],
        dfl             = cfg["dfl"],
        mosaic          = cfg["mosaic"],
        mixup           = cfg["mixup"],
        copy_paste      = cfg["copy_paste"],
        close_mosaic    = cfg["close_mosaic"],
        flipud          = cfg["flipud"],
        fliplr          = cfg["fliplr"],
        hsv_h           = cfg["hsv_h"],
        hsv_s           = cfg["hsv_s"],
        hsv_v           = cfg["hsv_v"],
        translate       = cfg["translate"],
        scale           = cfg["scale"],
        shear           = cfg["shear"],
        perspective     = cfg["perspective"],
        degrees         = cfg["degrees"],
        label_smoothing = cfg["label_smoothing"],
        multi_scale     = cfg["multi_scale"],
        amp             = cfg["amp"],
        overlap_mask    = cfg["overlap_mask"],
        plots           = cfg["plots"],
        verbose         = cfg["verbose"],
    )
 
    _validate(data_yaml, device)
    _promote_best()
 
 
# ===========================================================================
# Validate + promote
# ===========================================================================
 
def _validate(data_yaml, device):
    if not BEST_PT.exists():
        print("[!!] best.pt not found, skipping validation.")
        return
    print("\n[>>] Validating best model with TTA...")
    try:
        vm      = YOLO(str(BEST_PT))
        metrics = vm.val(data=str(data_yaml), device=device,
                         augment=True, verbose=False, plots=True)
        mp  = metrics.box.mp
        mr  = metrics.box.mr
        f1  = 2 * mp * mr / (mp + mr + 1e-9)
        sep = "=" * 54
        print(f"\n{sep}")
        print("  VALIDATION RESULTS  (with TTA)")
        print(sep)
        print(f"  mAP@0.50      : {metrics.box.map50*100:.1f}%")
        print(f"  mAP@0.50:0.95 : {metrics.box.map*100:.1f}%")
        print(f"  Precision     : {mp*100:.1f}%")
        print(f"  Recall        : {mr*100:.1f}%")
        print(f"  F1            : {f1*100:.1f}%")
        if hasattr(metrics.box, "ap_class_index") and metrics.box.ap_class_index is not None:
            print(f"\n  Per-class mAP@0.50:")
            for idx, ap in zip(metrics.box.ap_class_index, metrics.box.ap50):
                name = VEHICLE_CLASSES[idx] if idx < len(VEHICLE_CLASSES) else str(idx)
                bar  = "#" * int(ap * 32)
                print(f"    {name:<12} {ap*100:5.1f}%  [{bar:<32}]")
        print(sep)
    except Exception as e:
        print(f"[!!] Validation failed: {e}")
 
 
def _promote_best():
    dst = PROJECT_DIR / "best.pt"
    if BEST_PT.exists():
        shutil.copy2(BEST_PT, dst)
        size = BEST_PT.stat().st_size / 1024 ** 2
        sep  = "=" * 62
        print(f"\n{sep}")
        print("  DONE!")
        print(sep)
        print(f"  Best model : {BEST_PT}  ({size:.1f} MB)")
        print(f"  Copied to  : {dst}")
        print(f"\n  Start app  : python vehicle.py")
        print(f"{sep}\n")
    else:
        print("\n[!!] best.pt not found -- check logs above.\n")
 
 
# ===========================================================================
# CLI
# ===========================================================================
 
def parse_args():
    p = argparse.ArgumentParser(
        description="Fast vehicle-only YOLOv8 training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Speed modes (pick one):
  python train.py --fast          ~5 min  on GPU  | quick test
  python train.py                 ~20 min on GPU  | good accuracy  (DEFAULT)
  python train.py --accurate      ~2 hrs  on GPU  | best accuracy
 
Override anything:
  python train.py --model yolov8s.pt --epochs 50 --imgsz 640
  python train.py --resume                         # continue from checkpoint
        """,
    )
    p.add_argument("--fast",     action="store_true", help="Fast 30-epoch smoke test")
    p.add_argument("--accurate", action="store_true", help="High-accuracy 150-epoch run")
    p.add_argument("--model",    type=str,   default=None,
                   help="Override model (e.g. yolov8s.pt, yolov8m.pt)")
    p.add_argument("--epochs",   type=int,   default=None,
                   help="Override number of epochs")
    p.add_argument("--imgsz",    type=int,   default=None,
                   help="Override image size (e.g. 416, 640, 1280)")
    p.add_argument("--resume",   action="store_true",
                   help="Resume from last checkpoint")
    return p.parse_args()
 
 
if __name__ == "__main__":
    args = parse_args()
    print("\n" + "=" * 62)
    print("   VehicleAI  --  Fast Vehicle-Only YOLOv8 Training")
    print("=" * 62 + "\n")
    train(args)