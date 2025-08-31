import cv2
import time
import os
import argparse
from ultralytics import YOLO   # Ultralytics YOLOv8 framework (for training, detection, exporting)
import platform

# Default paths and parameters
DEFAULT_WEIGHTS = "/file/path/to/best.pt" # Path to trained YOLO model weights (best.pt after training)
DEFAULT_DATA_YAML = "/file/path/to/data.yaml" # Path to dataset YAML (contains train/val paths and class names)
DEFAULT_EPOCHS = 50 # Default number of epochs for training
DEFAULT_IMGSZ = 640 # Default image size for training/detection
DEFAULT_FPS = 30 # Default frames per second for live detection
WINDOW_NAME = "Object Detection (press Q to quit)" # Window title for OpenCV


# Function to open video source
def open_source(source_str: str) -> cv2.VideoCapture:
    if source_str.isdigit():
        cam_index = int(source_str)

        # On macOS, OpenCV sometimes needs CAP_AVFOUNDATION for webcams
        if platform.system() == "Darwin":
            cap = cv2.VideoCapture(cam_index, cv2.CAP_AVFOUNDATION)
            if cap.isOpened():
                return cap

        # Fallback (works on Windows/Linux normally)
        return cv2.VideoCapture(cam_index)

    # Otherwise, assume it's a file path or URL
    return cv2.VideoCapture(source_str)


# Detection mode
def detect_mode(weights, source, imgsz, fps, headless, save_path, conf):
    """
    Run object detection on a video source (camera, video, image, or stream).
    - weights: YOLO model weights file (best.pt or yolov8n.pt, etc.)
    - source: camera index, path to image/video, or stream URL (0 for internal camera and 1 for external webcam)
    - imgsz: image size for inference
    - fps: target frames per second
    - headless: if True, don’t open a window (useful for servers/drones)
    - save_path: optional path to save annotated video
    - conf: confidence threshold (0–1) for showing detections
    """
    print("DETECT MODE")
    print("Loading model from:", weights)

    # Check if weights file exists
    if not os.path.exists(weights):
        print("ERROR: weights file not found:", weights)
        return

    # Load the YOLO model
    try:
        model = YOLO(weights)
        print("Model loaded! Classes:", getattr(model, "names", None))
    except Exception as exception:
        print("Could not load model:", exception)
        return

    # Try to open the video/camera source
    print("Opening source:", source)
    cap = open_source(source)
    if not cap.isOpened():
        print("ERROR: video source not opened.")
        return

    # Optional: save annotated video to a file
    writer = None
    if save_path:
        print("Saving annotated video to:", save_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")   # codec for mp4
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        stream_fps = cap.get(cv2.CAP_PROP_FPS)
        if not stream_fps or stream_fps < 1:
            stream_fps = fps
        try:
            writer = cv2.VideoWriter(save_path, fourcc, stream_fps, (width, height))
        except Exception as e:
            print("WARNING: could not create VideoWriter:", e)
            writer = None

    # If not running headless, open a display window
    if not headless:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    print("Starting loop... Press Q to quit the window")
    frame_interval = 1.0 / max(1, fps)   # how long each frame should take

    try:
        while True:
            t0 = time.time()

            # Grab a frame from the source
            ok, frame = cap.read()
            if not ok:
                print("Can't read frame (end of file or camera problem). Stopping.")
                break

            # Run YOLO inference on the frame
            try:
                results = model(frame, imgsz=imgsz, conf=0.6) # confidence threshold applied here
                annotated = results[0].plot() # draw bounding boxes
            except Exception as e:
                print("Inference error:", e)
                annotated = frame # fallback to raw frame if inference fails

            # If saving, write annotated frame to file
            if writer is not None:
                try:
                    writer.write(annotated)
                except Exception as e:
                    print("Writer error:", e)

            # Show the frame in a window (unless headless mode)
            if not headless:
                cv2.imshow(WINDOW_NAME, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q')):
                    print("Q pressed. Bye.")
                    break

            # Keep loop close to target FPS
            elapsed = time.time() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        # Clean up resources when done
        if writer is not None:
            writer.release()
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
        print("Detect mode finished.")


# Training mode
def train_mode(weights, data_yaml, epochs, imgsz):
    """
    Train a YOLO model.
    - weights: base weights 
    - data_yaml: path to dataset YAML
    - epochs: number of training epochs
    - imgsz: image size
    """
    print("TRAIN MODE")
    print("Base weights:", weights)
    print("Data yaml:", data_yaml)
    print("Epochs:", epochs, "imgsz:", imgsz)

    if not os.path.exists(data_yaml):
        print("ERROR: dataset yaml not found:", data_yaml)
        return

    try:
        model = YOLO(weights) # Load base model (downloads if not local)
        model.train(data=data_yaml, epochs=epochs, imgsz=imgsz)
        print("Training done! Check the runs/ folder for results (like best.pt).")
    except Exception as e:
        print("Training failed:", e)


# Export mode
def export_mode(weights, fmt, imgsz, int8, half):
    """
    Export a trained YOLO model to other formats (for deployment).
    Options: onnx, tflite, engine, coreml
    """
    print("=== EXPORT MODE ===")
    print("Weights:", weights)
    print("Format:", fmt)
    print("imgsz:", imgsz, "INT8:", int8, "Half:", half)

    if not os.path.exists(weights):
        print("ERROR: trained weights not found:", weights)
        return

    try:
        model = YOLO(weights)
        kwargs = dict(format=fmt, imgsz=imgsz)
        if fmt == "tflite" and int8:
            kwargs["int8"] = True
        if fmt in ("onnx", "engine") and half:
            kwargs["half"] = True
        out_path = model.export(**kwargs)
        print("Exported to:", out_path)
    except Exception as e:
        print("Export failed:", e)


# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Object detection app (detect/train/export)")
    parser.add_argument("--mode", type=str, default="detect", help="detect or train or export")

    # Common
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS, help="model weights path")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="image size")

    # Detect-specific
    parser.add_argument("--source", type=str, default="0", help="camera index, video path, or stream URL")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="target fps")
    parser.add_argument("--headless", action="store_true", help="run without window")
    parser.add_argument("--save", type=str, default="", help="save annotated video to mp4 path")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold (0.0–1.0)")

    # Train-specific
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_YAML, help="dataset yaml path")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="epochs for training")

    # Export-specific
    parser.add_argument("--format", type=str, default="onnx", help="export format: onnx/tflite/engine/coreml")
    parser.add_argument("--int8", action="store_true", help="tflite int8 quantization")
    parser.add_argument("--half", action="store_true", help="fp16 for onnx/engine")

    return parser.parse_args()


# Main entry point
def main():
    args = parse_args()
    print("Args:", args)

    if args.mode == "detect":
        detect_mode(args.weights, args.source, args.imgsz, args.fps, args.headless, args.save, args.conf)
    elif args.mode == "train":
        train_mode(args.weights, args.data, args.epochs, args.imgsz)
    elif args.mode == "export":
        export_mode(args.weights, args.format, args.imgsz, args.int8, args.half)
    else:
        print("Unknown mode:", args.mode)
        print("Try: --mode detect  OR  --mode train  OR  --mode export")


if __name__ == "__main__":
    main()
