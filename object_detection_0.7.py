import cv2
import time
import os
import argparse
from ultralytics import YOLO

# defaults (change these to your real paths)
DEFAULT_WEIGHTS = "/Users/rokasjonusas/Desktop/FinalProject/runs/detect/train8/weights/best.pt"
DEFAULT_DATA_YAML = "/Users/rokasjonusas/Desktop/object_detection/RussianEquipment/data.yaml"
DEFAULT_EPOCHS = 50
DEFAULT_IMGSZ = 640
DEFAULT_FPS = 30
WINDOW_NAME = "Object Detection (press Q to quit)"

def open_source(source_str: str) -> cv2.VideoCapture:
    # if the source is a number like "0", use that camera index
    if source_str.isdigit():
        cam_index = int(source_str)
        return cv2.VideoCapture(cam_index)
    # else assume it's a file path or URL (e.g., rtsp)
    return cv2.VideoCapture(source_str)

def detect_mode(weights, source, imgsz, fps, headless, save_path):
    print("DETECT MODE")
    print("Loading model from:", weights)
    if not os.path.exists(weights):
        print("ERROR: weights file not found:", weights)
        return

    try:
        model = YOLO(weights)
        print("Model loaded! Classes:", getattr(model, "names", None))
    except Exception as exception:
        print("Could not load model:", exception)
        return

    print("Opening source:", source)
    cap = open_source(source)
    if not cap.isOpened():
        print("ERROR: video source not opened.")
        return

    # optional: save annotated output if --save given
    writer = None
    if save_path:
        print("Saving annotated video to:", save_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
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

    if not headless:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    print("Starting loop... Press Q to quit the window")
    frame_interval = 1.0 / max(1, fps)

    try:
        while True:
            t0 = time.time()

            ok, frame = cap.read()
            if not ok:
                print("Can't read frame (end of file or camera problem). Stopping.")
                break

            # Run inference
            try:
                results = model(frame, imgsz=imgsz, conf=0.6)   # you can add conf=0.25 later
                annotated = results[0].plot()         # draw boxes on the frame
            except Exception as e:
                print("Inference error:", e)
                annotated = frame

            # write to file if requested
            if writer is not None:
                try:
                    writer.write(annotated)
                except Exception as e:
                    print("Writer error:", e)

            # show the window (unless headless)
            if not headless:
                cv2.imshow(WINDOW_NAME, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q')):
                    print("Q pressed. Bye.")
                    break

            # keep near target FPS (sleep INSIDE the loop)
            elapsed = time.time() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        if writer is not None:
            writer.release()
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
        print("Detect mode finished.")

def train_mode(weights, data_yaml, epochs, imgsz):
    print("=== TRAIN MODE ===")
    print("Base weights:", weights)
    print("Data yaml:", data_yaml)
    print("Epochs:", epochs, "imgsz:", imgsz)

    if not os.path.exists(data_yaml):
        print("ERROR: dataset yaml not found:", data_yaml)
        print("Please check your path. Example minimal content:")
        print("""
train: images/train
val: images/val
nc: 2
names: ['class_a','class_b']
""")
        return

    try:
        model = YOLO(weights)  # if not local, ultralytics may download base weights
        model.train(data=data_yaml, epochs=epochs, imgsz=imgsz)
        print("Training done! Check the runs/ folder for results (like best.pt).")
    except Exception as e:
        print("Training failed:", e)

def export_mode(weights, fmt, imgsz, int8, half):
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

def parse_args():
    parser = argparse.ArgumentParser(description="Object detection app (detect/train/export)")
    parser.add_argument("--mode", type=str, default="detect", help="detect or train or export")

    # common
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS, help="model weights path")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="image size")

    # detect
    parser.add_argument("--source", type=str, default="0", help="camera index like 0, or video path, or rtsp url")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="target fps")
    parser.add_argument("--headless", action="store_true", help="no window (good for servers)")
    parser.add_argument("--save", type=str, default="", help="save annotated video to mp4 path")

    # train
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_YAML, help="dataset yaml path")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="epochs for training")

    # export
    parser.add_argument("--format", type=str, default="onnx", help="export format: onnx/tflite/engine/coreml")
    parser.add_argument("--int8", action="store_true", help="tflite int8 quantization")
    parser.add_argument("--half", action="store_true", help="fp16 for onnx/engine")

    return parser.parse_args()

def main():
    args = parse_args()
    print("Args:", args)

    if args.mode == "detect":
        detect_mode(args.weights, args.source, args.imgsz, args.fps, args.headless, args.save)
    elif args.mode == "train":
        train_mode(args.weights, args.data, args.epochs, args.imgsz)
    elif args.mode == "export":
        export_mode(args.weights, args.format, args.imgsz, args.int8, args.half)
    else:
        print("Unknown mode:", args.mode)
        print("Try: --mode detect  OR  --mode train  OR  --mode export")

if __name__ == "__main__":
    main()

