# Garbage Detection Model

Detect garbage-like objects from a webcam, images, folders, or video files using YOLO.

The default model is `yolov8n.pt`, which is trained on COCO. It can detect common waste-like objects such as bottles, cups, bowls, and food items. For real garbage classes such as `plastic_bag`, `cardboard`, or `food_waste`, train a custom model with your own labeled dataset.

## Features

- Webcam, image, folder, and video input
- Class filtering for garbage-like objects
- Recursive folder search with `--recursive`
- Annotated image outputs
- Custom YOLO training script and dataset config

## Install

```powershell
pip install -r requirements.txt
```

## Run On Webcam

```powershell
python garbage_detector.py --source 0
```

Press `Q` to quit.

## Run On An Image

```powershell
python garbage_detector.py --source path\to\image.jpg
```

Annotated images are saved in `outputs`.

## Run On A Folder

```powershell
python garbage_detector.py --source path\to\images
```

Search inside subfolders too:

```powershell
python garbage_detector.py --source path\to\images --recursive
```

## Run On A Video

```powershell
python garbage_detector.py --source path\to\video.mp4
```

## Compare MobileNet And YOLO

Run both models on each webcam frame in two separate windows, one for MobileNet and one for YOLO. Each window shows inference time, estimated inference FPS, and detection count. Both use the same confidence threshold.

```powershell
python compare_models.py --source 0
```

Use a video file or custom YOLO weights:

```powershell
python compare_models.py --source path\to\video.mp4 --yolo-model runs\garbage_detector\weights\best.pt
```

The MobileNet model is SSD MobileNet V2 from TensorFlow Hub. By default, YOLO uses `yolov8n.pt`; pass custom weights to compare against a model trained on your garbage dataset. For a meaningful accuracy comparison, evaluate both models against the same labeled images and classes; the live view compares speed and predictions, not accuracy.

## Useful Options

Detect every class in the model:

```powershell
python garbage_detector.py --source 0 --classes all
```

Use only selected classes:

```powershell
python garbage_detector.py --source 0 --classes bottle,cup,bowl
```

Raise or lower confidence:

```powershell
python garbage_detector.py --source 0 --confidence 0.5
```

## Train A Custom Garbage Model

Create a YOLO dataset like this:

```text
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
```

Each label file must use YOLO format:

```text
class_id x_center y_center width height
```

Then train:

```powershell
python train_garbage_model.py --data garbage_dataset.yaml --epochs 50
```

After training, run detection with the best weights:

```powershell
python garbage_detector.py --model runs\garbage_detector\weights\best.pt --source 0 --classes all
```

## Dataset Classes

The included `garbage_dataset.yaml` uses these classes:

```text
0 plastic_bottle
1 glass_bottle
2 can
3 paper
4 cardboard
5 plastic_bag
6 food_waste
7 other_trash
```

## Project Files

```text
garbage_detector.py      # Detect from webcam, image, folder, or video
train_garbage_model.py   # Train a custom garbage detector
garbage_dataset.yaml     # YOLO dataset configuration
requirements.txt         # Python dependencies
```

## Notes

Model weights, training runs, datasets, and output images are ignored by Git so the repository stays lightweight.
