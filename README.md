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

## Choice of implementation

**1 - Original Configuration of the authors**

```python
Training config
	Solver: SGD
		lr: 0.007
		momentum: 0.9
	Batch_size: 16
	Image_size: 1024 x 2048
	Norm: SyncInplaceBN
	Augmentation:
		- RandomCrop
		- RandomFlip
			- Normalize
	Warmup: 200 iterations 1/3 lr to lr
	Scheduler: StepLR
		- step [120, 144]
		- total epoch: 160
```

**2 - Adapted configuration to my resources**

The authors trained their models using 16 NVIDIA Titan X GPUs. Due to the fact that I only had one GPU to train the model, I could not use the same configuration. Here is a summary of the necessary implementation decisions:

- I reduced the size of the images by 2 leading to `512 x 1024` images.
- In order to increase the batch size and the speed of the training I decided to use **mixed precision training**. Mixed precision training is simply combining single precision (32 bit) tensor with half precision (16bit) tensor. Using 16bit tensor frees up a lot of memory and also speeds up the overall training, but it can also reduce the performance of the overall training. (More information in the [paper](https://arxiv.org/pdf/1802.00930.pdf))
- For the optimizer, I decided to use `Adam` which is more stable and so requires less optimisation to reach good performances, I reduced the learning rate base on the ratio of batch size between their implementation and mine, giving me a learning rate of `1.3e-3` . Base on my experiments changing the learning rate did not seem to make a big impact.
- Since I was not able to train for the number of epochs used during the training (160 epochs), I decided to use `ReduceLROnPlateau` as a scheduler in order to optimize my performance on a small number of epochs.
- For the augmentations:
    - `RandomFlip` and `Normalisation` (with the statistics of the dataset) are applied
- On the testing pipeline: I did not do multiscaling for the testing procedure.

To sum up, we have:

```python
Training config
	Solver: Adam
		lr: 1.3e-3
	Batch_size: 3
	Image_size: 512 x 1024
	Norm: InplaceBN
	Augmentation:
		- RandomFlip
			- Normalize
	Warmup: 500 iterations lr/500 to lr
	Scheduler: ReduceLROnPlateau
		- patience 3
		- min lr: 1e-4
```

## Results

Best metrics I obtained, with the config given as default:

```python
Epoch 27  |    PQ     SQ     RQ     N
-------------------------------------- 
All       |  46.0   75.6   58.4    19
Things    |  34.1   73.1   45.8     8
Stuff     |  54.7   77.4   67.5    11 
```