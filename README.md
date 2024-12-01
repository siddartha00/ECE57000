# Efficient PS

Paper: [Efficient PS](http://panoptic.cs.uni-freiburg.de/#main)

Code from the authors: [here](https://github.com/DeepSceneSeg/EfficientPS)

## Dependencies

To create this code I used multiple frameworks:

- [EfficientNet-Pytorch](https://github.com/lukemelas/EfficientNet-PyTorch) for the backbone
- [detectron2](https://github.com/facebookresearch/detectron2) for the instance head (Mask-RCNN)
- [In-Place Activated BatchNorm](https://github.com/mapillary/inplace_abn)
- [COCO 2018 Panoptic Segmentation Task API (Beta version)](https://github.com/cocodataset/panopticapi) to compute panoptic quality metric


## How to use

- Download Cityscape Dataset:
```
git clone https://github.com/mcordts/cityscapesScripts.git
# City scapes script
pip install git+https://github.com/mcordts/cityscapesScripts.git
# Panoptic
CITYSCAPES_DATASET=/path/to/abovementioned/cityscapes python cityscapesscripts/preparation/createPanopticImgs.py
```
- Install [pytorch](https://pytorch.org/)
```
pip install torch==2.5.0
```
- Install [Albumentation](https://albumentations.ai/)
```
pip install -U albumentations
```
- Install [Inplace batchnorm](https://github.com/mapillary/inplace_abn)
```
pip install inplace-abn
```
- Install [EfficientNet Pytorch](https://github.com/lukemelas/EfficientNet-PyTorch)
```
pip install efficientnet_pytorch
```
- Install [detecron 2 dependencies](https://github.com/facebookresearch/detectron2)
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
- Install [Panoptic api](https://github.com/cocodataset/panopticapi)
```
pip install git+https://github.com/cocodataset/panopticapi.git
```
- Install [Pytorch lighting](https://www.pytorchlightning.ai/). Due to compatability issues use `pip==24.0.0`
```
pip install pytorch-lightning==1.5.8
```
- Modify `config.yaml`
- Run `train_net.py`


## Results

Best metrics I obtained, with the config given as default:

```python
Epoch 27  |    PQ     SQ     RQ     N
-------------------------------------- 
All       |  46.0   75.6   58.4    19
Things    |  34.1   73.1   45.8     8
Stuff     |  54.7   77.4   67.5    11 
```