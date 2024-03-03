
```bash
pip install ultralytics
```

<div align="center">
  <a href="https://ultralytics.com/yolov8" target="_blank">
  <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/yolo-comparison-plots.png"></a>
</div>

## <div align="center">Documentation</div>

See the [YOLOv5 Docs](https://docs.ultralytics.com/yolov5) for full documentation on training, testing and deployment. See below for quickstart examples.

<details open>
<summary>Install</summary>

Clone repo and install [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a
[**Python>=3.7.0**](https://www.python.org/) environment, including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

```python
import torch

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom

# Images
img = "https://ultralytics.com/images/zidane.jpg"  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
```


### Train

YOLOv5 segmentation training supports auto-download COCO128-seg segmentation dataset with `--data coco128-seg.yaml` argument and manual download of COCO-segments dataset with `bash data/scripts/get_coco.sh --train --val --segments` and then `python train.py --data coco.yaml`.

```bash
# Single-GPU
python segment/train.py --data coco128-seg.yaml --weights yolov5s-seg.pt --img 640

# Multi-GPU DDP
python -m torch.distributed.run --nproc_per_node 4 --master_port 1 segment/train.py --data coco128-seg.yaml --weights yolov5s-seg.pt --img 640 --device 0,1,2,3
```

### Val

Validate YOLOv5s-seg mask mAP on COCO dataset:

```bash
bash data/scripts/get_coco.sh --val --segments  # download COCO val segments split (780MB, 5000 images)
python segment/val.py --weights yolov5s-seg.pt --data coco.yaml --img 640  # validate
```

### Predict

Use pretrained YOLOv5m-seg.pt to predict bus.jpg:

```bash
python segment/predict.py --weights yolov5m-seg.pt --source data/images/bus.jpg
```

```python
model = torch.hub.load(
    "ultralytics/yolov5", "custom", "yolov5m-seg.pt"
)  # load from PyTorch Hub (WARNING: inference not yet supported)
```

| ![zidane](https://user-images.githubusercontent.com/26833433/203113421-decef4c4-183d-4a0a-a6c2-6435b33bc5d3.jpg) | ![bus](https://user-images.githubusercontent.com/26833433/203113416-11fe0025-69f7-4874-a0a6-65d0bfe2999a.jpg) |
| ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |

### Export

Export YOLOv5s-seg model to ONNX and TensorRT:

```bash
python export.py --weights yolov5s-seg.pt --include onnx engine --img 640 --device 0
```

</details>

## <div align="center">Classification</div>

YOLOv5 [release v6.2](https://github.com/ultralytics/yolov5/releases) brings support for classification model training, validation and deployment! See full details in our [Release Notes](https://github.com/ultralytics/yolov5/releases/v6.2) and visit our [YOLOv5 Classification Colab Notebook](https://github.com/ultralytics/yolov5/blob/master/classify/tutorial.ipynb) for quickstart tutorials.


### Train

YOLOv5 classification training supports auto-download of MNIST, Fashion-MNIST, CIFAR10, CIFAR100, Imagenette, Imagewoof, and ImageNet datasets with the `--data` argument. To start training on MNIST for example use `--data mnist`.

```bash
# Single-GPU
python classify/train.py --model yolov5s-cls.pt --data cifar100 --epochs 5 --img 224 --batch 128

# Multi-GPU DDP
python -m torch.distributed.run --nproc_per_node 4 --master_port 1 classify/train.py --model yolov5s-cls.pt --data imagenet --epochs 5 --img 224 --device 0,1,2,3
```

### Val

Validate YOLOv5m-cls accuracy on ImageNet-1k dataset:

```bash
bash data/scripts/get_imagenet.sh --val  # download ImageNet val split (6.3G, 50000 images)
python classify/val.py --weights yolov5m-cls.pt --data ../datasets/imagenet --img 224  # validate
```

### Predict

Use pretrained YOLOv5s-cls.pt to predict bus.jpg:

```bash
python classify/predict.py --weights yolov5s-cls.pt --source data/images/bus.jpg
```

```python
model = torch.hub.load(
    "ultralytics/yolov5", "custom", "yolov5s-cls.pt"
)  # load from PyTorch Hub
```

### Export

Export a group of trained YOLOv5s-cls, ResNet and EfficientNet models to ONNX and TensorRT:

```bash
python export.py --weights yolov5s-cls.pt resnet50.pt efficientnet_b0.pt --include onnx engine --img 224
```

