import os
import json
import numpy as np
import albumentations as A
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.image as mpimg
from datasets.panoptic_dataset import PanopticDataset
from detectron2.config import get_cfg, CfgNode
from detectron2.utils.visualizer import Visualizer
from efficientps import EffificientPS
from detectron2.data import MetadataCatalog

def add_box(ax, box, color='b', thickness=2):
    """ Draws annotations in an image.
    # Arguments
        ax          : The matplotlib ax to draw on.
        box         : A [1, 5] matrix (x1, y1, x2, y2, label).
        color       : The color of the boxes.
        thickness   : (optional) thickness of the bbox.
    """
    rect = Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                        color=color,
                        fill=False,
                        linewidth=thickness
                    )
    ax.add_patch(rect)

def add_boxes(ax, boxes, color, thickness=2):
    """ Draws boxes on an image with a given color.
    # Arguments
        image     : The matplotlib ax to draw on.
        boxes     : A [N, 4] matrix (x1, y1, x2, y2).
        color     : The color of the boxes.
        thickness : The thickness of the lines to draw boxes with.
    """
    for b in boxes:
        add_box(ax, b, color, thickness=thickness)

def vizualise_input_targets(dataset, seed=65):
    # Get a sample
    sample = dataset[seed]

    # Figure
    fig = plt.figure(figsize=(15,10))
    for i, (name, tensor) in enumerate(sample.items()):
        if name in ['instance', 'image_id']:
            continue

        ax = fig.add_subplot(2, 3, i+1)
        if name == 'image':
            add_boxes(ax, sample['instance'].gt_boxes.tensor.numpy(), 'g')

        ax.set_title(name)
        plt.imshow(tensor)

    plt.show()


def add_box_on_image(im, box, color, thickness, label):
    start_point = (int(box[0]),int(box[1]))
    end_point = (int(box[2]),int((box[3])))
    col = (color[0],color[1],color[2])
    print(f"pt1: {start_point} pt2: {end_point} color: {col}")
    cv2.rectangle(im,start_point,end_point,col,thickness)
    # Add the label
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)  # White color
    font_thickness = 1

    # Calculate text size and position
    text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
    text_w, text_h = text_size
    text_x = start_point[0]
    text_y = end_point[1]  # Adjust y-coordinate for label position

    # Draw the text
    cv2.rectangle(im,(text_x,text_y-text_h),(text_x+text_w,text_y),col,-1)
    cv2.putText(im, label, (text_x, text_y), font, font_scale, font_color, font_thickness,)

def add_custom_param(cfg):
    """
    In order to add custom config parameter in the .yaml those parameter must
    be initialised
    """
    # Model
    cfg.MODEL_CUSTOM = CfgNode()
    cfg.MODEL_CUSTOM.BACKBONE = CfgNode()
    cfg.MODEL_CUSTOM.BACKBONE.EFFICIENTNET_ID = 5
    cfg.MODEL_CUSTOM.BACKBONE.LOAD_PRETRAIN = False
    # DATASET
    cfg.NUM_CLASS = 19
    cfg.DATASET_PATH = "/home/ubuntu/Elix/cityscapes"
    cfg.TRAIN_JSON = "gtFine/cityscapes_panoptic_train.json"
    cfg.VALID_JSON = "gtFine/cityscapes_panoptic_val.json"
    cfg.PRED_DIR = "preds"
    cfg.PRED_JSON = "cityscapes_panoptic_preds.json"
    # Transfom
    cfg.TRANSFORM = CfgNode()
    cfg.TRANSFORM.NORMALIZE = CfgNode()
    cfg.TRANSFORM.NORMALIZE.MEAN = (106.433, 116.617, 119.559)
    cfg.TRANSFORM.NORMALIZE.STD = (65.496, 67.6, 74.123)
    cfg.TRANSFORM.RESIZE = CfgNode()
    cfg.TRANSFORM.RESIZE.HEIGHT = 512
    cfg.TRANSFORM.RESIZE.WIDTH = 1024
    cfg.TRANSFORM.RANDOMCROP = CfgNode()
    cfg.TRANSFORM.RANDOMCROP.HEIGHT = 512
    cfg.TRANSFORM.RANDOMCROP.WIDTH = 1024
    cfg.TRANSFORM.HFLIP = CfgNode()
    cfg.TRANSFORM.HFLIP.PROB = 0.5
    # Solver
    cfg.SOLVER.NAME = "SGD"
    cfg.SOLVER.ACCUMULATE_GRAD = 1
    # Runner
    cfg.BATCH_SIZE = 1
    cfg.CHECKPOINT_PATH = ""
    cfg.PRECISION = 32
    # Callbacks
    cfg.CALLBACKS = CfgNode()
    cfg.CALLBACKS.CHECKPOINT_DIR = None
    # Inference
    cfg.INFERENCE = CfgNode()
    cfg.INFERENCE.AREA_TRESH = 0

def model_demo(image_path, json_path):

    config_path = os.path.join(os.getcwd(),'config.yaml')
    cfg = get_cfg()
    add_custom_param(cfg)
    cfg.merge_from_file(config_path)


    img = np.asarray(Image.open(image_path))

    transform_inference = A.Compose([
        A.Resize(height=cfg.TRANSFORM.RESIZE.HEIGHT, width=cfg.TRANSFORM.RESIZE.WIDTH),
        A.Normalize(mean=cfg.TRANSFORM.NORMALIZE.MEAN, std=cfg.TRANSFORM.NORMALIZE.STD),
        A.RandomCrop(height=512, width=1024),
        A.HorizontalFlip(p=0.5),
    ])

    img = transform_inference(image = img)
    img = img['image']
    image = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()  # NCHW

    # Create model or load a checkpoint
    if os.path.exists(cfg.CHECKPOINT_PATH):
        print('""""""""""""""""""""""""""""""""""""""""""""""')
        print("Loading model from {}".format(cfg.CHECKPOINT_PATH))
        print('""""""""""""""""""""""""""""""""""""""""""""""')
        efficientps = EffificientPS.load_from_checkpoint(cfg=cfg,
            checkpoint_path=cfg.CHECKPOINT_PATH)
    else:
        print('""""""""""""""""""""""""""""""""""""""""""""""')
        print("Creating a new model")
        print('""""""""""""""""""""""""""""""""""""""""""""""')
        efficientps = EffificientPS(cfg)
        cfg.CHECKPOINT_PATH = None

    device = torch.device('cpu')
    efficientps.eval()

    image_tensor = image.to(device)
    inputs = {"image": image_tensor}

    # Run inference
    with torch.no_grad():
        output = efficientps(inputs)

    instances = output['instance'][0]
    boxes = instances.pred_boxes.tensor.numpy()
    categoreis = instances.pred_classes
    masks = instances.pred_masks
    print(boxes)

    im = cv2.imread(image_path)
    
    # Resize the image to 512x1024
    im = cv2.resize(im, (1024, 512))
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

    json_data = json.load(open(json_path))
    thing_classes = []
    thing_colors = []

    for i in json_data['categories']:
        if i['isthing'] == 1:
            thing_classes.append(i['name'])
            thing_colors.append(i['color'])

    for i in range(len(boxes)):
        add_box_on_image(im,box=boxes[i],color=thing_colors[categoreis[i]],thickness=2,label=thing_classes[categoreis[i]])

    # Display the image with axes
    plt.imshow(im)
    plt.axis('on')  # Turn on the axes
    plt.show()



    

def main():
    base_path = "/media/siddartha/DevDrive/test_run/gtFine"
    train_json = "gtFine/cityscapes_panoptic_train.json"

    transform = A.Compose([
        A.Resize(height=512, width=1024),
        A.RandomCrop(height=512, width=1024),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(106.433, 116.617, 119.559), std=(65.496, 67.6, 74.123)),
        # A.RandomScale(scale_limit=[0.5, 2]),
        # A.RandomSizedCrop()
    ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

    train_dataset = PanopticDataset(train_json, base_path, 'train', transform=transform)

    json_path = os.path.join(base_path, train_json)

    vizualise_input_targets(train_dataset)
    model_demo('sample.jpg', json_path)
    model_demo('sample_1.png',json_path)
    model_demo('sample_2.png',json_path)
    model_demo('sample_3.png',json_path)

if __name__ == '__main__':
    main()