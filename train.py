from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from YAML
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolov8n.yaml").load("yolov8n.pt")  # build from YAML and transfer weights


if __name__ == "__main__":
    # Train the model
    results = model.train(data="D:\\Documents\\bs\\test_train\\Tello\\data.yaml", epochs=100, imgsz=640)
    results = model.val()  # evaluate model performance on the validation set