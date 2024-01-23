from ultralytics import YOLO
from ultralytics import NAS
from ultralytics import RTDETR

# # Load a model
# model = YOLO('runs/detect/train/weights/last.pt')
# model.train(resume=True)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')
# results = model.train(data='ultralytics/cfg/datasets/afo.yaml', patience=1000,epochs=1000, imgsz=480, device=[1], batch=128, half=True,mixup=0.1)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set

# model=YOLO('saved_ckpt_clean2/a0/weights/last.pt')
# model.fuse()
# model.export(format='onnx', simplify=True, imgsz=480, half=True, opset=11)

# metrics = model.val()
# for _ in range(1000):
#     results = model.predict("/home/golftec/workspace/ngocnt/develop/mmdetection/frames/000284.jpg")  # predict on an image
# 'torchscript', 'onnx', 'openvino', 'engine', 'coreml', 'saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs', 'paddle', 'ncnn'
# model.export(format='tflite', nms=True, half=True, optimize=True, simplify=True, imgsz=[640, 576])
# model.export(format='ncnn', nms=True, half=True, simplify=True, imgsz=[640, 576])

# model.export(format='tensorrt', nms=True, imgsz=640, half=True, simplify=True)

# from coremltools.models.neural_network import quantization_utils
# import coremltools as ct
# model_fp32 = ct.models.MLModel('s_afo_train_combine/weights/last.mlmodel')
# model_fp16 = quantization_utils.quantize_weights(model_fp32, nbits=16)
# model_name = "afo_small.mlmodel"  
# model_fp16.save(model_name)

# Nas
# model = RTDETR('rtdetr-s.pt')
# model.export(format='mlmodel', nms=True, half=True, optimize=True, simplify=True, imgsz=[480, 480], dynamic=True)
# model.train(data="ultralytics/cfg/datasets/afo.yaml", imgsz=480, epochs=300, device=[0, 1], batch=64) 




#pose
# from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n-pose.yaml').load('yolov8n-pose.pt')  # build from YAML and transfer weights
# Train the model
# results = model.train(data='ultralytics/cfg/datasets/coco-pose.yaml', patience=100 , epochs=100, imgsz=480, device=[0], batch=64*2, half=True)

# #convert
# model=YOLO('pose_combine_480/weights/last.pt')
# model.fuse()
# # model.export(format='onnx', simplify=True, imgsz=480, half=True)
# model.export(format='ncnn', half=True, imgsz=480)
# model = YOLO('runs/pose/train/weights/last.pt')
# model.train(resume=True)