# 🚙 VehicleAI — Intelligent Vehicle Detection System

A premium, high-performance Flask web application for real-time traffic and vehicle detection using YOLOv8. Designed with an elegant aesthetic ("Playfair Display" & "Cormorant Garamond") to provide advanced insights into vehicle statistics with granular tracking.

## ✨ Features

- **Advanced AI Detection:** Powered by state-of-the-art YOLOv8 object detection algorithms.
- **Eight Distinct Vehicle Classes:** Dynamically detects **Bicycle**, **Car**, **Motorcycle**, **Bus**, and **Truck**, while accurately filtering out **SUVs**, **Vans**, and **Ambulances** using advanced geometry and aspect-ratio heuristics.
- **Premium User Interface:** A meticulously designed, fully responsive dashboard featuring:
  - Drag-and-drop intelligent image scanning
  - Side-by-side original vs. annotated image comparisons
  - Dynamic vehicle distribution charts
  - High-confidence bounding boxes with opacity gradients
- **Pre-trained & Custom Model Support:** Seamlessly auto-detects custom fine-tuned models (`best.pt`) or falls back to the pretrained MS COCO `yolov8n.pt`. 

## 🚀 Quick Setup

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the application
python vehicle.py
```

Then navigate to **http://localhost:5000** in your browser.

---

## 🎯 Fine-Tuning the Model (Custom dataset)

If you want to fine-tune the base YOLOv8 model for advanced accuracy on `[bicycle, bus, car, motorcycle, truck]` or your own dataset, you can use our built-in training script.

1. **Configure Dataset:** By default, `train.py` fetches a robust vehicle detection dataset containing basic vehicle classes from Roboflow Universe via your API key.
2. **Run Training:**
   ```bash
   python train.py --epochs 10 --model yolov8n.pt
   ```
3. **Auto-Load:** Once training finishes, the best model is saved to `runs/detect/vehicle_model/weights/best.pt`. Start `vehicle.py`, and the app will automatically detect and load your bespoke model!

---

## 📁 Project Structure

```
vehicle-detection/
├── vehicle.py          # Flask backend, YOLOv8 inference, and heuristics
├── train.py            # Automated YOLOv8 fine-tuning/training script
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Premium frontend dashboard UI
└── runs/               # (Auto-generated) Model checkpoints
```
