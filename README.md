# I. Training
## 1. Mobile, OMMC

```python
from ultralytics import YOLO
model = YOLO('yolov8n.yaml').load('yolov8n.pt')
results = model.train(data='ultralytics/cfg/datasets/afo.yaml', patience=100,epochs=1000, imgsz=480, half=True, mixup=0.1)  # train the model
```

## 2. Cloud
```python
from ultralytics import YOLO
model = YOLO('yolov8m.yaml').load('yolov8m.pt')
results = model.train(data='ultralytics/cfg/datasets/afo.yaml', patience=100,epochs=1000, half=True, mixup=0.1)  # train the model
```