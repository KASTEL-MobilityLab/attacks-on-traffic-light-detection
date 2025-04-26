# Fool the Stoplight: Realistic Adversarial Patch Attacks on Traffic Light Detectors

This repository contains a PyTorch implementation of patch-based attacks on CNNs for traffic light detection from our IV2024 paper "Fool the Stoplight: Realistic Adversarial Patch Attacks on Traffic Light
Detectors".

## Installation Guide
1. Clone this repository.
2. Install required Python packages:
    ```
    pip install -r requirements.txt
    ```
3. Install YOLOv7
    - Clone the YOLOv7 repository into your `adversarial_attack` folder:
        ```
        git clone https://github.com/WongKinYiu/yolov7.git
        ``` 
    - NOTE: Sometimes this repository has a bug where it can't find pre-trained weights to download. If this occurs, replace `yolov7/utils/google_utils.py` with `yolov7_fix/utils/google_utils.py`.
4. Install ultralytics to use YOLOv8.


## Usage
### Training a Patch
- Select a config file from `configs/` and adjust the paths to your dataset and model weights.
- Run the training script from the project directory:
    ```
    python3 train_patch_yolov7.py configs/<CONFIG_NAME>.yaml
    ```

### Evaluating a Patch
- Run the patch evaluation script from the project directory. 
- You must pass the following parameters:
    - `weights`: Path to the weights of the target model.
    - `test_imgs_path`: Path to the directory of evaluation images.
    - `classes_path`: Path to the classes of the target model.
    - `trained_patch_path`: Path to the `.pt` file of the patch.
- Specifying the following parameters is recommended:
    - `iou`: IOU threshold for inference.
    - `bbox_width_multiplier`: Desired width of the patch. This should match what you used in training.
- See the script for the full list of parameters. Here is an example usage of the script:
```
python3 evaluate_patch.py --task test --weights <PATH_TO_MODEL_WEIGHTS> --test_imgs_path <PATH_TO_TEST_IMGS> --classes_path <PATH_TO_CLASSES> --trained_patch_path <PATH_TO_TRAINED_PATCH> --bbox_width_multiplier 2 --iou 0.65
```

## General Hints

### WandB
This project is set up to log your training to Weights and Biases. Before starting the scripts, login into your Weights & Biases account using the command line: `wandb login`, for more information see [the W&B tutorials](https://docs.wandb.ai/quickstart).

### Config files
The attacks run using YAML config files that can be found in `configs/`. 

If you wish to create a new config file, please see `configs/config_patch_template.yaml` for an explanation of what each config variable means.

## Project Folder Structure Overview
- `classnames/`: Class names for each dataset.
- `configs/`: Config files for patch training.
- `notebooks/`: Notebooks used for physical patch analysis (including GradCAM).
- `patch_utils/`: Utilities used in patch training and evaluation.
- `trained_patches/`: Folder where trained patches are stored. If you change the name then you must change it in config files. 
- `yolov7_fix/`: Bug fix from WongKinYiu YOLOv7 repo.
- `evaluate_patch.py`: Main script to digitally evaluate an adversarial patch.
- `train_patch.py_yolov7`: Main script to generate an adversarial patch for YOLOv7.
- `train_patch.py_yolov8`: Main script to generate an adversarial patch for YOLOv8.

# Citation

If you find this code useful for your research, please cite our paper:

```latex
@InProceedings{pavlitska2025fooling,
  author    = {Pavlitska, Svetlana and Robb, Jamie and Polley, Nikolai and Yazgan, Melih and Zöllner, J. Marius},
  title={Fool the Stoplight: Realistic Adversarial Patch Attacks on Traffic Light
Detectors},
  booktitle = {IEEE Intelligent Vehicles Symposium (IV)},
  year      = {2025}
}
```