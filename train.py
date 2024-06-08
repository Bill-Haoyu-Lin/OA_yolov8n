from ultralytics import YOLO

def main():
    # Create a new YOLO model from scratch

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO('yolov8n.pt')

    # Train the model using the 'coco128.yaml' dataset for 3 epochs
    results = model.train(data='./datasets/furniture.yaml', epochs=200,batch=32,workers=2)

    # Evaluate the model's performance on the validation set
    results = model.val()

    # Export the model to ONNX format
    success = model.export(format='onnx')

if __name__ == '__main__':
    main()