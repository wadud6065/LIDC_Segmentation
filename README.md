# U-Net Segmentation Training

This repository contains code for training a **U-Net model** for image segmentation.  
It supports **training/validation logging, checkpointing, and early stopping**.

---

## 🚀 Features
- U-Net model for semantic segmentation
- Training and validation loops with logging
- **Metrics**: BCE Loss, Dice Score, IoU
- **Checkpoints**:
  - `last_model.pth`: saved every epoch (latest state)
  - `best.pth`: saved only when validation Dice improves
- CSV logging of training progress (`log.csv`)
- Early stopping support

---
## 📂 Model Download
You can access the latest UNet model training file from this [Google Drive link](https://drive.google.com/file/d/1x3rS_Xjx6Q_jwUu4L8xrOW0DwJX7ZE2_/view?usp=sharing).

## 📂 Folder Structure
model_outputs/ <br>
└── {experiment_name}/  <br>
    ├── best.pth          # Best model (based on val Dice)  <br>
    ├── last_model.pth    # Latest checkpoint  <br>
    └── log.csv           # Training log (loss, dice, iou, etc.)  <br>

## 📥 Dataset

To download the full dataset, run:

```bash
pip install gdown
gdown 1azf2iuIiI9a8F0s8IPBhldD6aUHd5eeB



