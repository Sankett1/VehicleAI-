import argparse
import json
import os
import shutil
import sys
import urllib.request
import zipfile
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO
 
# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "datasets" / "vehicles"
RUNS_DIR    = PROJECT_DIR / "runs"
BEST_PT     = RUNS_DIR / "detect" / "vehicle_model" / "weights" / "best.pt"
LAST_PT     = RUNS_DIR / "detect" / "vehicle_model" / "weights" / "last.pt"
 
# ---------------------------------------------------------------------------
# Our 8 vehicle classes  (maps name -> our class ID 0-7)
# ---------------------------------------------------------------------------
VEHICLE_CLASSES = [
    "car",          # 0
    "truck",        # 1
    "bus",          # 2
    "motorcycle",   # 3
    "bicycle",      # 4
    "van",          # 5
    "suv",          # 6
    "ambulance",    # 7
]
 
# COCO class IDs that correspond to vehicles  (from COCO 80-class list)
# coco_id -> our_class_name
COCO_TO_VEHICLE = {
    2:  "car",          # COCO: car
    5:  "bus",          # COCO: bus
    7:  "truck",        # COCO: truck
    3:  "motorcycle",   # COCO: motorcycle
    1:  "bicycle",      # COCO: bicycle
    # van/suv/ambulance are not in COCO -- they come from custom data
    # or are mapped from 'truck'/'car' if no custom data available
}
 
# COCO 2017 val split -- small (5000 images, ~1 GB) good for training too
COCO_VAL_IMGS_URL   = "http://images.cocodataset.org/zips/val2017.zip"
COCO_VAL_ANNS_URL   = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_TRAIN_IMGS_URL = "http://images.cocodataset.org/zips/train2017.zip"   # 18 GB -- optional
 
# ---------------------------------------------------------------------------
# High-accuracy training config
# ---------------------------------------------------------------------------
CFG = {
    "epochs":          100,
    "imgsz":           640,
    "batch":           -1,       # auto; set 8 manually if auto fails on Windows
    "workers":         4,
    "optimizer":       "AdamW",
    "lr0":             0.001,
    "lrf":             0.01,
    "momentum":        0.937,
    "weight_decay":    0.0005,
    "warmup_epochs":   5,
    "warmup_momentum": 0.8,
    "warmup_bias_lr":  0.1,
    "cos_lr":          True,
    "box":             7.5,
    "cls":             0.5,
    "dfl":             1.5,
    "mosaic":          1.0,
    "mixup":           0.15,
    "copy_paste":      0.3,
    "close_mosaic":    10,
    "flipud":          0.1,
    "fliplr":          0.5,
    "hsv_h":           0.015,
    "hsv_s":           0.7,
    "hsv_v":           0.4,
    "translate":       0.1,
    "scale":           0.5,
    "shear":           2.0,
    "perspective":     0.0001,
    "degrees":         10.0,
    "label_smoothing": 0.1,
    "multi_scale":     True,
    "amp":             True,
    "overlap_mask":    True,
    "patience":        50,
    "save_period":     10,
    "plots":           True,
    "verbose":         True,
}
 
 
# ===========================================================================
# Helpers
# ===========================================================================
 
def get_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"[GPU] {name}  ({vram:.1f} GB VRAM)")
        return "0"
    print("[CPU] No GPU found. Training will be slow.")
    return "cpu"
 
 
def progress_bar(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        mb  = downloaded / 1024 ** 2
        tot = total_size  / 1024 ** 2
        print(f"\r    {pct:5.1f}%  {mb:.0f} / {tot:.0f} MB", end="", flush=True)
 
 
def download_file(url, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[OK] Already downloaded: {dest.name}")
        return
    print(f"[>>] Downloading {dest.name} ...")
    urllib.request.urlretrieve(url, str(dest), reporthook=progress_bar)
    print()   # newline after progress bar
 
 
def extract_zip(zip_path: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[>>] Extracting {zip_path.name} ...")
    with zipfile.ZipFile(str(zip_path), "r") as z:
        z.extractall(str(dest))
    print(f"[OK] Extracted to {dest}")
 
 
# ===========================================================================
# COCO  ->  Vehicle-only dataset builder
# ===========================================================================
 
def build_vehicle_dataset_from_coco():
    """
    Downloads COCO val2017 (5000 images, ~1 GB) and its annotations,
    then extracts ONLY images that contain at least one vehicle,
    converts bounding boxes to YOLO format, and saves them under
    datasets/vehicles/{images,labels}/{train,val}/
    """
    coco_dir  = PROJECT_DIR / "datasets" / "_coco_raw"
    zip_imgs  = coco_dir / "val2017.zip"
    zip_anns  = coco_dir / "annotations.zip"
    imgs_dir  = coco_dir / "val2017"
    anns_file = coco_dir / "annotations" / "instances_val2017.json"
 
    # Step 1: Download
    print("\n[>>] Downloading COCO val2017 (vehicles only will be kept)...")
    print("     Images : ~800 MB")
    print("     Annotations : ~240 MB\n")
    download_file(COCO_VAL_IMGS_URL, zip_imgs)
    download_file(COCO_VAL_ANNS_URL, zip_anns)
 
    # Step 2: Extract
    if not imgs_dir.exists():
        extract_zip(zip_imgs, coco_dir)
    if not anns_file.exists():
        extract_zip(zip_anns, coco_dir)
 
    # Step 3: Parse annotations
    print("[>>] Parsing COCO annotations for vehicle classes...")
    with open(anns_file, encoding="utf-8") as f:
        coco = json.load(f)
 
    # Build image-id -> file_name map
    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    id_to_wh   = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}
 
    # Collect vehicle annotations grouped by image
    # coco category_id is 1-indexed; subtract 1 for 0-indexed
    # We use COCO_TO_VEHICLE which maps coco category_id -> our class name
    vehicle_ann_by_image = {}   # img_id -> list of (our_class_id, cx, cy, w, h)
 
    for ann in coco["annotations"]:
        cat_id = ann["category_id"]
        if cat_id not in COCO_TO_VEHICLE:
            continue
 
        our_name     = COCO_TO_VEHICLE[cat_id]
        our_class_id = VEHICLE_CLASSES.index(our_name)
        img_id       = ann["image_id"]
        iw, ih       = id_to_wh[img_id]
 
        # COCO bbox: [x_top_left, y_top_left, width, height]
        x, y, bw, bh = ann["bbox"]
        # Convert to YOLO normalised cx, cy, w, h
        cx = (x + bw / 2) / iw
        cy = (y + bh / 2) / ih
        nw = bw / iw
        nh = bh / ih
 
        # Skip tiny boxes (noise)
        if nw < 0.005 or nh < 0.005:
            continue
 
        vehicle_ann_by_image.setdefault(img_id, []).append(
            (our_class_id, cx, cy, nw, nh)
        )
 
    print(f"[OK] Found {len(vehicle_ann_by_image)} vehicle images in COCO val2017")
 
    # Step 4: Split 90% train / 10% val
    img_ids   = sorted(vehicle_ann_by_image.keys())
    split_idx = int(len(img_ids) * 0.9)
    splits = {
        "train": img_ids[:split_idx],
        "val":   img_ids[split_idx:],
    }
 
    # Step 5: Write images + labels
    for split, ids in splits.items():
        img_out = DATASET_DIR / "images" / split
        lbl_out = DATASET_DIR / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
 
        print(f"[>>] Writing {split}: {len(ids)} images...")
        for img_id in ids:
            fname  = id_to_file[img_id]
            src    = imgs_dir / fname
            if not src.exists():
                continue
 
            # Copy image
            shutil.copy2(str(src), str(img_out / fname))
 
            # Write YOLO label
            stem  = Path(fname).stem
            label = lbl_out / f"{stem}.txt"
            with open(label, "w") as lf:
                for cls_id, cx, cy, w, h in vehicle_ann_by_image[img_id]:
                    lf.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
 
        print(f"[OK] {split}: {len(ids)} images written")
 
    # Step 6: Write data.yaml
    write_data_yaml()
    print(f"\n[OK] Vehicle-only dataset ready at {DATASET_DIR}")
    print(f"     Classes: {VEHICLE_CLASSES}\n")
 
 
def write_data_yaml():
    """Write a clean data.yaml for our vehicle dataset."""
    cfg = {
        "path":  DATASET_DIR.as_posix(),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(VEHICLE_CLASSES),
        "names": {i: n for i, n in enumerate(VEHICLE_CLASSES)},
    }
    yaml_path = DATASET_DIR / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return yaml_path
 
 
# ===========================================================================
# Dataset entry point
# ===========================================================================
 
def get_dataset():
    """
    Returns absolute path to data.yaml.
    Uses existing dataset if valid, otherwise builds from COCO.
    """
    yaml_path = DATASET_DIR / "data.yaml"
 
    # Check if we already have a usable vehicle dataset
    if yaml_path.exists():
        train_dir = DATASET_DIR / "images" / "train"
        val_dir   = DATASET_DIR / "images" / "val"
        train_ok  = train_dir.exists() and any(train_dir.iterdir())
        val_ok    = val_dir.exists()   and any(val_dir.iterdir())
 
        if train_ok and val_ok:
            n_train = len(list(train_dir.glob("*.*")))
            n_val   = len(list(val_dir.glob("*.*")))
            print(f"[OK] Existing vehicle dataset: {n_train} train / {n_val} val images\n")
            return fix_yaml(yaml_path)
        else:
            print("[!!] Dataset folder exists but images are missing. Rebuilding...\n")
 
    # Build from COCO
    build_vehicle_dataset_from_coco()
    return fix_yaml(DATASET_DIR / "data.yaml")
 
 
def fix_yaml(yaml_path: Path) -> Path:
    """Ensure data.yaml has correct absolute paths for Windows."""
    yaml_path = yaml_path.resolve()
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
 
    # Force absolute posix path
    cfg["path"] = DATASET_DIR.resolve().as_posix()
 
    # Make sure names + nc match our classes if not custom
    if not cfg.get("names") or len(cfg["names"]) == 0:
        cfg["nc"]    = len(VEHICLE_CLASSES)
        cfg["names"] = {i: n for i, n in enumerate(VEHICLE_CLASSES)}
 
    # Ensure split dirs exist
    for split in ("train", "val"):
        if split in cfg:
            split_dir = DATASET_DIR / cfg[split]
            split_dir.mkdir(parents=True, exist_ok=True)
 
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
 
    print(f"[OK] data.yaml -> {yaml_path}")
    print(f"     path  : {cfg['path']}")
    print(f"     train : {cfg.get('train')}")
    print(f"     val   : {cfg.get('val')}")
    print(f"     nc    : {cfg['nc']}")
    print(f"     names : {list(cfg['names'].values()) if isinstance(cfg['names'], dict) else cfg['names']}\n")
    return yaml_path
 
 
# ===========================================================================
# Training
# ===========================================================================
 
def train(args):
    device = get_device()
    data_yaml = get_dataset()
 
    cfg = dict(CFG)
    cfg["epochs"] = args.epochs
    cfg["imgsz"]  = args.imgsz
 
    # CPU mode adjustments
    if device == "cpu":
        cfg.update({
            "batch":       8,
            "amp":         False,
            "multi_scale": False,
            "workers":     0,
        })
        print("[!!] CPU mode: batch=8, AMP off, multi-scale off.\n")
 
    # Fast smoke-test
    if args.fast:
        cfg.update({
            "epochs":      20,
            "batch":       4,
            "patience":    10,
            "multi_scale": False,
            "mixup":       0.0,
            "copy_paste":  0.0,
        })
        args.model = "yolov8n.pt"
        print("[!!] Fast mode: 20 epochs, yolov8n, minimal augmentation.\n")
 
    # Load model
    if args.resume and LAST_PT.exists():
        model = YOLO(str(LAST_PT))
        print(f"[>>] Resuming from checkpoint: {LAST_PT}\n")
    else:
        model = YOLO(args.model)
        print(f"[>>] Transfer learning from: {args.model}\n")
 
    # Summary
    sep = "=" * 62
    gpu_label = torch.cuda.get_device_name(0) if device != "cpu" else "CPU"
    print(f"{sep}")
    print("  VEHICLE-ONLY TRAINING CONFIGURATION")
    print(sep)
    print(f"  Classes     : {VEHICLE_CLASSES}")
    print(f"  Model       : {args.model}")
    print(f"  Device      : {gpu_label}")
    print(f"  Epochs      : {cfg['epochs']}  (patience={cfg['patience']})")
    print(f"  Image size  : {cfg['imgsz']} px")
    print(f"  Batch       : {'auto' if cfg['batch'] == -1 else cfg['batch']}")
    print(f"  LR          : cosine {cfg['lr0']} -> {cfg['lr0'] * cfg['lrf']:.5f}")
    print(f"  AMP (FP16)  : {cfg['amp']}")
    print(f"  Multi-scale : {cfg['multi_scale']}")
    print(f"  Mixup       : {cfg['mixup']}")
    print(f"  Copy-paste  : {cfg['copy_paste']}")
    print(f"  Dataset     : {data_yaml}")
    print(f"{sep}\n")
 
    model.train(
        data            = str(data_yaml),
        device          = device,
        project         = str(RUNS_DIR / "detect"),
        name            = "vehicle_model",
        exist_ok        = True,
        pretrained      = True,
        epochs          = cfg["epochs"],
        imgsz           = cfg["imgsz"],
        batch           = cfg["batch"],
        workers         = cfg["workers"],
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
        patience        = cfg["patience"],
        save            = True,
        save_period     = cfg["save_period"],
        plots           = cfg["plots"],
        verbose         = cfg["verbose"],
    )
 
    _validate(data_yaml, device)
    _promote_best()
 
 
# ===========================================================================
# Validation with TTA
# ===========================================================================
 
def _validate(data_yaml, device):
    if not BEST_PT.exists():
        print("[!!] best.pt not found, skipping validation.")
        return
    print("[>>] Validating best model with Test-Time Augmentation...")
    vm      = YOLO(str(BEST_PT))
    metrics = vm.val(
        data    = str(data_yaml),
        device  = device,
        augment = True,
        verbose = False,
        plots   = True,
    )
    mp  = metrics.box.mp
    mr  = metrics.box.mr
    f1  = 2 * mp * mr / (mp + mr + 1e-9)
    sep = "=" * 54
    print(f"\n{sep}")
    print("  VALIDATION RESULTS  (with TTA)")
    print(sep)
    print(f"  mAP@0.50        : {metrics.box.map50*100:.1f}%")
    print(f"  mAP@0.50:0.95   : {metrics.box.map*100:.1f}%")
    print(f"  Precision       : {mp*100:.1f}%")
    print(f"  Recall          : {mr*100:.1f}%")
    print(f"  F1              : {f1*100:.1f}%")
 
    if hasattr(metrics.box, "ap_class_index") and metrics.box.ap_class_index is not None:
        print(f"\n  Per-class mAP@0.50:")
        for idx, ap in zip(metrics.box.ap_class_index, metrics.box.ap50):
            name = VEHICLE_CLASSES[idx] if idx < len(VEHICLE_CLASSES) else str(idx)
            bar  = "#" * int(ap * 32)
            print(f"    {name:<12} {ap*100:5.1f}%  [{bar:<32}]")
    print(sep)
 
 
def _promote_best():
    dst = PROJECT_DIR / "best.pt"
    if BEST_PT.exists():
        shutil.copy2(BEST_PT, dst)
        size = BEST_PT.stat().st_size / 1024 ** 2
        sep  = "=" * 62
        print(f"\n{sep}")
        print("  TRAINING COMPLETE!")
        print(sep)
        print(f"  Best model : {BEST_PT}  ({size:.1f} MB)")
        print(f"  Copied to  : {dst}")
        print(f"\n  Run app    : python vehicle.py")
        print(f"  (vehicle.py auto-loads best.pt on startup)")
        print(f"{sep}\n")
    else:
        print("\n[!!] best.pt not found -- check training logs above.\n")
 
 
# ===========================================================================
# CLI
# ===========================================================================
 
def parse_args():
    p = argparse.ArgumentParser(
        description="Train YOLOv8 for vehicle-only detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model size guide:
  yolov8n.pt  -- nano,   fastest,  lowest accuracy
  yolov8s.pt  -- small,  fast,     decent accuracy
  yolov8m.pt  -- medium, balanced  (DEFAULT)
  yolov8l.pt  -- large,  slower,   high accuracy
  yolov8x.pt  -- xlarge, slowest,  highest accuracy
 
Examples:
  python train.py                          # auto dataset + yolov8m, 100 epochs
  python train.py --epochs 200             # longer = higher accuracy
  python train.py --model yolov8x.pt       # maximum accuracy model
  python train.py --imgsz 1280             # better for small/distant vehicles
  python train.py --resume                 # continue from last checkpoint
  python train.py --fast                   # quick 20-epoch smoke test
        """,
    )
    p.add_argument("--epochs", type=int, default=CFG["epochs"],
                   help=f"Epochs (default: {CFG['epochs']})")
    p.add_argument("--model",  type=str, default="yolov8m.pt",
                   help="Base weights (default: yolov8m.pt)")
    p.add_argument("--imgsz",  type=int, default=CFG["imgsz"],
                   help="Image size (default: 640)")
    p.add_argument("--resume", action="store_true",
                   help="Resume from last checkpoint")
    p.add_argument("--fast",   action="store_true",
                   help="Quick 20-epoch smoke test with yolov8n")
    return p.parse_args()
 
 
if __name__ == "__main__":
    args = parse_args()
    print("\n" + "=" * 62)
    print("   VehicleAI  --  Vehicle-Only YOLOv8 Training")
    print("=" * 62 + "\n")
    train(args)