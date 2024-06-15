import torch

# Model
model = torch.hub.load('yolov5',
                                         'custom',
                                         path='./yolov5/weights/yolov5s.pt',
                                         source='local')# or yolov5n - yolov5x6, custom

# Images
img = "./zidane.jpg"  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.