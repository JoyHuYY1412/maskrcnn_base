# Usage

## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.

## LVIS dataset

### generate subset
check [train_top_b.ipynb](train_top_b.ipynb)

### path
```bash
# symlink the coco dataset
cd ~/github/maskrcnn-benchmark
mkdir -p datasets/lvis
# for LVIS dataset:
ln -s /path_to_coco_dataset/train2017 datasets/lvis/images/train2017
ln -s /path_to_coco_dataset/test2017 datasets/lvis/images/test2017
ln -s /path_to_coco_dataset/val2017 datasets/lvis/images/val2017

ln -s /path_to_generated_lvis_annotations datasets/lvis/annotations
```

## config to change ><
**1. edit [e2e_mask_rcnn_R_50_FPN_1x_periodically_testing_topb_maskxrcnn_from_scratch.yaml](https://github.com/JoyHuYY1412/maskrcnn_base/blob/master/configs/lvis/e2e_mask_rcnn_R_50_FPN_1x_periodically_testing_topb_maskxrcnn_from_scratch.yaml)**

change
```python
NUM_CLASSES:b+1   //line 35

DATASETS:         //line 50
  TRAIN: ("lvis_v0.5_train_topb",)
  TEST: ("lvis_v0.5_val_topb",)
  
OUTPUT_DIR: "./ckps/ckp-topb_freeze2_from_scratch"  //line 67
TENSORBOARD_EXPERIMENT: "./logs-topb_freeze2_from_scratch/logs-top271_freeze2"
```

**2. edit [paths_catalog.py](https://github.com/JoyHuYY1412/maskrcnn_base/blob/master/maskrcnn_benchmark/config/paths_catalog.py)**

add
```python
        "lvis_v0.5_train_topb": {
            "img_dir": "lvis/images/train2017",
            "ann_file": "lvis/annotaions/lvis_v0.5_train_topb.json"
        },
        "lvis_v0.5_val_topb": {
            "img_dir": "lvis/images/val2017",
            "ann_file": "lvis/annotaions/lvis_v0.5_val_topb.json"
        },
```

**3. edit [lvis.py](https://github.com/JoyHuYY1412/maskrcnn_base/blob/master/maskrcnn_benchmark/data/datasets/lvis.py)**

edit
```python
sorted_id_file = path_to_sorted_id_topb (absolute path) //line 39
```

**4. edit [__init__.py](https://github.com/JoyHuYY1412/maskrcnn_base/blob/master/maskrcnn_benchmark/data/datasets/evaluation/lvis/__init__.py)**

edit
```python
   gt_path="datasets/lvis/annotaions/lvis_v0.5_val_topb.json",   //line 16
```

## code for training



```python
python -m torch.distributed.launch --nproc_per_node=8 --master_addr 127.0.0.3 --master_port 29503 ./tools/train_net.py --use-tensorboard --config-file "configs/lvis/e2e_mask_rcnn_R_50_FPN_1x_periodically_testing_topb_maskxrcnn_from_scratch.yaml" MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 1000
```

