from ultralytics import YOLO

dataset_dir = "datasets"
model = YOLO('yolov5mu.pt')

# Train the model
model.train(data=dataset_dir+'\\data.yaml', epochs=112, imgsz=640,batch=10, workers=0)