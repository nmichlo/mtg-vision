def main():
    from ultralytics import YOLO

    # Load your model
    model = YOLO(
        "/Users/nathanmichlo/Desktop/active/mtg/mtg-vision/data/yolo_mtg_dataset_v3/models/1ipho2mn_best.pt"
    )

    # Export to TensorFlow.js format
    path = model.export(format="onnx", nms=True)
    print(path)

    # load onnx model


if __name__ == "__main__":
    main()
