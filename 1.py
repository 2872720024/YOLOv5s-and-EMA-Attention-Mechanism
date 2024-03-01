import torch
import torch.nn
import onnx
 
model = torch.load('best.pt')
 
input_names = ['input']
output_names = ['output']
 
x = torch.randn(1,3,32,32,requires_grad=True)
 
torch.onnx.export(model, x, 'best.onnx', input_names=input_names, output_names=output_names, verbose='True')