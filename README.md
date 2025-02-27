# DentalX: Context-aware Dental Disease Detection

## Description
Diagnosing dental diseases from radiographs is time-consuming and challenging due to the subtle nature of diagnostic evidence. Existing methods, which rely on object detection models designed for natural images with more distinct target patterns, struggle to detect dental diseases that present with far less visual support. To address this challenge, we propose DentalX, a novel context-aware dental disease detection approach that leverages oral structure information to mitigate the visual ambiguity inherent in radiographs. Specifically, we introduce a structural context extraction module that learns an auxiliary task: dental anatomy semantic segmentation. The module extracts meaningful structural context and integrates it into the primary disease detection task to enhance the detection of subtle dental diseases. 

<p align="center">Figure 1: Overview of the proposed DentalX. The structural context extraction module introduces the auxiliary dental anatomy segmentation task and extracts useful structural information to improve dental disease detection. DentalX jointly learns both tasks from partially annotated data by computing loss value only when the corresponding ground truth label is present.
</p><p align="center"><img src="assets/dentalx.png" width="70%"></p>


## Setup Environment
1. Install required libraries by running 
    ```bash
    conda create -n dentalx python=3.8
    conda activate dentalx
    pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
    pip install -r requirements.txt
    pip install -v -e .
    ```

2. Download pretrained model weights (optional)
    ```bash
    mkdir pretrained_weights
    cd pretrained_weights
    wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth
    cd ..
    ```

## Bring your datasets
1. Ensure your detection dataset is in COCO format. The dataset folder should have the following structure:
   ```
    coco
    ├── annotations
    │   ├── instances_train2017.json
    │   ├── instances_val2017.json
    │   ├── instances_test2017.json
    ├── train2017
    ├── val2017
    ├── test2017
   ```
2. Ensure your segmentation dataset is in COCO Stuff10k format. The dataset folder should have the following structure:
   ```
    coco_stuff10k
    ├── images
    │   ├── train2014
    │   ├── test2014
    ├── annotations
    │   ├── train2014
    │   ├── test2014
    ├── imagesLists
    │   ├── train.txt
    │   ├── test.txt
    │   ├── all.txt
   ```
3. Update the following configuration in `exps/default/dentalx.py`
   - `self.num_classes`: number of classes in detection task
   - `self.data_dir`: path to the `coco` dataset folder
   - `self.semantic_num_classes`: number of classes in segmentation task
   - `self.semantic_train_image_dir`: path to the `coco_stuff10k/images/train2014` folder
   - `self.semantic_train_ann_dir`: path to the `coco_stuff10k/annotations/train2014` folder
   - `self.semantic_test_image_dir`: path to the `coco_stuff10k/images/test2014` folder
   - `self.semantic_test_ann_dir`: path to the `coco_stuff10k/annotations/test2014` folder

## How to run
Run the scripts with `--help` argument to see arguments descriptions.
The following example trains our proposed model with the default configuration:
```bash
python tools/train.py -f exps/default/dentalx.py -d 1 -b 8 --fp16 -o -c pretrained_weights/yolox_l.pth \
    --logger wandb wandb-project <project_name>
```

All experiments metrics are logged to Tensorboard and WandB. They can be disabled by excluding the `--logger` argument.
