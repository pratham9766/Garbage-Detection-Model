"""Run MobileNet and YOLO on the same camera frames for a live comparison."""

import argparse
import time

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from ultralytics import YOLO


MOBILENET_MODEL_URL = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
COCO_LABELS = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow",
    22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe", 27: "backpack",
    28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase", 34: "frisbee",
    35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite", 39: "baseball bat",
    40: "baseball glove", 41: "skateboard", 42: "surfboard", 43: "tennis racket",
    44: "bottle", 46: "wine glass", 47: "cup", 48: "fork", 49: "knife",
    50: "spoon", 51: "bowl", 52: "banana", 53: "apple", 54: "sandwich",
    55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza",
    60: "donut", 61: "cake", 62: "chair", 63: "couch", 64: "potted plant",
    65: "bed", 67: "dining table", 70: "toilet", 72: "television", 73: "laptop",
    74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone", 78: "microwave",
    79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator", 84: "book",
    85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear", 89: "hair dryer",
    90: "toothbrush",
}


def draw_box(frame, x1, y1, x2, y2, label, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text_y = max(20, y1 - 7)
    cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def run_mobilenet(model, frame, threshold):
    height, width = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = tf.convert_to_tensor(np.expand_dims(rgb, axis=0), dtype=tf.uint8)
    start = time.perf_counter()
    detections = model(tensor)
    elapsed_ms = (time.perf_counter() - start) * 1000

    output = frame.copy()
    boxes = detections["detection_boxes"][0].numpy()
    classes = detections["detection_classes"][0].numpy()
    scores = detections["detection_scores"][0].numpy()
    count = 0
    for box, class_id, score in zip(boxes, classes, scores):
        if float(score) < threshold:
            continue
        ymin, xmin, ymax, xmax = box
        x1, y1 = max(0, int(xmin * width)), max(0, int(ymin * height))
        x2, y2 = min(width - 1, int(xmax * width)), min(height - 1, int(ymax * height))
        name = COCO_LABELS.get(int(class_id), f"class {int(class_id)}")
        draw_box(output, x1, y1, x2, y2, f"{name} {float(score):.2f}", (60, 220, 60))
        count += 1
    return output, elapsed_ms, count


def run_yolo(model, frame, threshold):
    start = time.perf_counter()
    result = model(frame, conf=threshold, verbose=False)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000
    output = frame.copy()
    count = 0
    for box in result.boxes:
        class_id = int(box.cls[0])
        score = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        draw_box(output, x1, y1, x2, y2, f"{result.names[class_id]} {score:.2f}", (40, 180, 255))
        count += 1
    return output, elapsed_ms, count


def open_capture(source):
    capture = cv2.VideoCapture(int(source), cv2.CAP_DSHOW) if source.isdigit() else cv2.VideoCapture(source)
    if source.isdigit() and not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(int(source))
    return capture


def main():
    parser = argparse.ArgumentParser(description="Compare MobileNet and YOLO in separate windows on the same video frames.")
    parser.add_argument("--source", default="0", help="Webcam index or video file path (default: 0).")
    parser.add_argument("--yolo-model", default="yolov8n.pt", help="YOLO weights path.")
    parser.add_argument("--confidence", type=float, default=0.45, help="Shared minimum confidence threshold.")
    args = parser.parse_args()

    print("Loading MobileNet from TensorFlow Hub...")
    mobilenet = hub.load(MOBILENET_MODEL_URL)
    print(f"Loading YOLO model: {args.yolo_model}")
    yolo = YOLO(args.yolo_model)

    capture = open_capture(args.source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    print("Both models are ready. Press Q to quit.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            mobile_view, mobile_ms, mobile_count = run_mobilenet(mobilenet, frame, args.confidence)
            yolo_view, yolo_ms, yolo_count = run_yolo(yolo, frame, args.confidence)
            for view, title, elapsed, count, color in (
                (mobile_view, "MobileNet", mobile_ms, mobile_count, (60, 220, 60)),
                (yolo_view, "YOLO", yolo_ms, yolo_count, (40, 180, 255)),
            ):
                cv2.rectangle(view, (0, 0), (view.shape[1], 70), (25, 25, 25), -1)
                cv2.putText(view, title, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(view, f"{elapsed:.1f} ms | {1000 / elapsed:.1f} FPS | {count} detections",
                            (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            cv2.imshow("MobileNet", mobile_view)
            cv2.imshow("YOLO", yolo_view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
