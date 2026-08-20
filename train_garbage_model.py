import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a custom YOLO garbage detection model.")
    parser.add_argument("--data", default="garbage_dataset.yaml", help="Path to YOLO dataset YAML.")
    parser.add_argument("--base-model", default="yolov8n.pt", help="Starting model weights.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    model = YOLO(args.base_model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="runs",
        name="garbage_detector",
    )


if __name__ == "__main__":
    main()
