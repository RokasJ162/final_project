# YOLOv8 Object Detection App

This project is a simple Python application for object detection. As dataset I choose Russian military equipment.

It supports three modes:

- Detect: run object detection on webcam, image, video, or stream.
- Train: train a YOLO model on your dataset.
- Export: export a trained model.

Requirements

- Python 3.9 or later
- PyTorch
- Ultralytics YOLOv8
- OpenCV

USE:
Once you download files from this repo:
1. Put your weights and data
   
- put your trained model (best.pt) inside models/ folder
- put your data.yaml inside data/ folder
- dataset should look like this:

    data/
      data.yaml
      images/train
      images/val
      labels/train
      labels/val

2. Run detection
   
webcam (0 is default camera):
python object_detection_0.7.py --mode detect --weights models/best.pt --source 0 (python3 .... for MacOS)

detect on image:
python object_detection_0.7.py --mode detect --weights models/best.pt --source myimage.jpg

detect on video and save it:
python object_detection_0.7.py --mode detect --weights models/best.pt --source video.mp4 --save out.mp4

3. Train model
python object_detection_0.7.py --mode train --weights yolov8n.pt --data data/data.yaml --epochs 50 
After training, check runs/ folder for best.pt

4. Export model
python app.py --mode export --weights models/best.pt --format onnx --half

notes
-----
- press Q to close the detect window
- runs/ folder will get a lot of graphs and results when you train (thats default in YOLO models)
  
Source for dataset:
https://universe.roboflow.com/capstoneproject/russian-military-vehicles



