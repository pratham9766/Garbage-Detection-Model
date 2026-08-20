import argparse
from pathlib import Path
from typing import Iterable

import cv2
from ultralytics import YOLO


DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_WASTE_CLASSES = {
    "bottle",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
}


def parse_class_filter(value: str) -> set[str]:
    if value.strip().lower() == "all":
        return set()
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def selected_class_ids(model: YOLO, target_classes: set[str]) -> list[int] | None:
    if not target_classes:
        return None

    names = model.names
    matches = [class_id for class_id, name in names.items() if name.lower() in target_classes]
    missing = sorted(target_classes - {names[class_id].lower() for class_id in matches})

    if missing:
        print(f"Warning: model does not contain these classes: {', '.join(missing)}")
        print("Use --classes all for custom garbage models with different class names.")

    return matches


def draw_filtered_detections(frame, result, target_classes: set[str], min_confidence: float):
    detected = []

    for box in result.boxes:
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = result.names[class_id]

        if confidence < min_confidence:
            continue
        if target_classes and class_name.lower() not in target_classes:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"{class_name} {confidence:.2f}"
        detected.append((class_name, confidence))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 220, 60), 2)
        cv2.rectangle(frame, (x1, max(0, y1 - 24)), (x1 + len(label) * 10, y1), (30, 220, 60), -1)
        cv2.putText(
            frame,
            label,
            (x1 + 4, y1 - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return detected


def open_capture(source: str):
    if source.isdigit():
        return cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
    return cv2.VideoCapture(source)


def run_stream(
    model: YOLO,
    source: str,
    target_classes: set[str],
    class_ids: list[int] | None,
    min_confidence: float,
) -> None:
    capture = open_capture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            result = model(frame, conf=min_confidence, classes=class_ids, verbose=False)[0]
            detected = draw_filtered_detections(frame, result, target_classes, min_confidence)

            status = f"Garbage objects: {len(detected)}"
            cv2.putText(frame, status, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 220, 60), 2)
            cv2.putText(frame, "Press Q to quit", (12, frame.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 220, 60), 2)
            cv2.imshow("Garbage Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def output_path_for(image_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{image_path.stem}_garbage{image_path.suffix}"


def run_images(
    model: YOLO,
    image_paths: Iterable[Path],
    target_classes: set[str],
    class_ids: list[int] | None,
    min_confidence: float,
    output_dir: Path,
) -> None:
    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        result = model(frame, conf=min_confidence, classes=class_ids, verbose=False)[0]
        detected = draw_filtered_detections(frame, result, target_classes, min_confidence)
        output_path = output_path_for(image_path, output_dir)
        cv2.imwrite(str(output_path), frame)
        print(f"{image_path}: {len(detected)} garbage-like object(s) -> {output_path}")


def expand_image_sources(source: str, recursive: bool) -> list[Path]:
    path = Path(source)
    if path.is_dir():
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        iterator = path.rglob("*") if recursive else path.iterdir()
        return sorted(item for item in iterator if item.is_file() and item.suffix.lower() in extensions)
    return [path]


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect garbage-like objects in images, videos, or webcam input.")
    parser.add_argument("--source", default="0", help="Image path, image folder, video path, or webcam index. Default: 0")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLO model path. Use your trained garbage model for best results.")
    parser.add_argument("--confidence", type=float, default=0.35, help="Minimum detection confidence.")
    parser.add_argument(
        "--classes",
        default=",".join(sorted(DEFAULT_WASTE_CLASSES)),
        help="Comma-separated class names to keep, or 'all'.",
    )
    parser.add_argument("--output-dir", default="outputs", help="Folder for annotated image outputs.")
    parser.add_argument("--recursive", action="store_true", help="Search image folders recursively.")
    args = parser.parse_args()

    target_classes = parse_class_filter(args.classes)
    model = YOLO(args.model)
    class_ids = selected_class_ids(model, target_classes)

    source_path = Path(args.source)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    is_image_input = source_path.is_dir() or source_path.suffix.lower() in image_extensions

    if is_image_input:
        image_paths = expand_image_sources(args.source, args.recursive)
        if not image_paths:
            raise RuntimeError(f"No supported images found in: {args.source}")
        run_images(model, image_paths, target_classes, class_ids, args.confidence, Path(args.output_dir))
    else:
        run_stream(model, args.source, target_classes, class_ids, args.confidence)


if __name__ == "__main__":
    main()
