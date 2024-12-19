import torch
import numpy as np

model = torch.jit.load(r'/workspace/data/yolov9-c-converted.pt')
model.to("cpu").eval()
x = torch.rand((1,1,640,640))
torch.onnx.export(model,x,'/workspace/data/yolov9.onnx',opset_version=14)
