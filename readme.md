# DPF: Learning Dense Prediction Fields with Weak Supervision


## Installation

### Requirements
    
    torch==1.8.1
    torchvision==0.9.1
    opencv-python==4.5.2
    timm==0.5.4

### Data preparation

#### Scene Parsing
Download PascalContext from [HERE](https://drive.google.com/file/d/13zPUAlmMrwWcDUNqO2CWyFQi28zxgf3J/view?usp=sharing). 

Download ADE20k from
[HERE](https://drive.google.com/file/d/1dV7C3Jc1lKIfWrj0_0KDw3rO04mV6aA1/view?usp=sharing).

Unzip them and the folder should have the following hierarchy:

    dataset_path
    └───PASCALContext
    |     └───    train(images for training)  
                   |   trainnpy(labels for training with full supervision)
                   |   val(images for validation)
                   |   valnpy(labels for validation)
                   |   Point_Annotation(labels for training with point-supervision)
                   |   info.json
                   |   ...
    └───ade20k
    |     └───    train(images for training)  
                   |   trainnpy(labels for training with full supervision)
                   |   val(images for validation)
                   |   valnpy(labels for validation)
                   |   Point_Annotation(labels for training with point-supervision)
                   |   info.json
                   |   ...


#### Intrinsic Decomposition

Download iiw dataset from [HERE](labelmaterial.s3.amazonaws.com/release/iiw-dataset-release-0.zip). Unzip it directly and merge it with the current ++iiw-dataset++ folder.

## Training and evaluating

#### Scene Parsing 
To train a DPF on PascalContext or ADE20k with four GPUs:

    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_port=27916 main.py train \
    --data-dir {your_data_path} \
    --crop-size 512 \
    --batch-size 2 \
    --random-scale 2 \
    --random-rotate 10 \
    --epochs 120 \
    --lr 0.028 \
    --momentum 0.9 \
    --lr-mode poly \
    --train_data [PASCAL/ade20k] \
    --workers 12 

To test the trained model with its checkpoint:

    
    CUDA_VISIBLE_DEVICES=0 python main.py test \
    -d {your_data_path} \
    -s 512 \
    --resume [model path] \
    --phase test \
    --batch-size 1 \
    --ms \
    --workers 10 \
    --train_data [PASCAL/ade20k]


#### Intrinsic Decomposition
To train a DPF on IIW with a single GPU:

    CUDA_VISIBLE_DEVICES=0 python main_iiw.py  -d  [iiw dataset path]  --ms -s 512 --batch-size 2 --random-scale 2 --random-rotate 10 --epochs 30 --lr 0.007 --momentum 0.9 --lr-mode poly --workers 12  -c 1
    
To test the trained model with its checkpoint:

    CUDA_VISIBLE_DEVICES=0 python main_iiw.py  -d  [iiw dataset path]  --ms -s 512 --batch-size 2 --random-scale 2 --random-rotate 10 --epochs 30 --lr 0.007 --momentum 0.9 --lr-mode poly --workers 12  -c 1 -e --resume [model path]

