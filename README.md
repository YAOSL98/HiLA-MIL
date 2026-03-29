# Weakly Supervised Artificial Intelligence for Pan-cancer Detection of Lymph Node Metastasis on Whole Slide Images: Advantages for Isolated Tumor Cells


![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)   ![License](https://img.shields.io/badge/license-MIT-green)   ![Languages](https://img.shields.io/badge/language-Python%20100%25-orange)

HiLA-MIL is a weakly supervised learning framework for **pan-cancer lymph node metastasis detection** on whole slide images, with special advantages in identifying Isolated Tumor Cells (ITCs).

## System Requirements
```
Operating System: Linux (x86_64)
Python: 3.10
CUDA: 11.8
GPU: NVIDIA GeForce RTX 3090 (24GB)
PyTorch: 2.1.1
All dependencies: listed in requirements.txt
```
## Installation Guide
```
# Create environment with Python 3.10
conda create -n your_env_name python=3.10

# Activate the environment
conda activate your_env_name

conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install -r requirements.txt
```

## Instructions for Use
### Preprocessing
For WSI preprocessing, we integrated more pre-trained networks (e.g., UNI/Gigapath/VIRCHOW) based on the codebase from https://github.com/mahmoodlab/CLAM. Please perform data preprocessing following the sections below.
```
cd ./CLAM_ours/

python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --seg --patch --stitch --patch_level 1

CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py --data_h5_dir DIR_TO_COORDS --data_slide_dir DATA_DIRECTORY --csv_path CSV_FILE_NAME --feat_dir FEATURES_DIRECTORY --batch_size 512 --slide_ext .ndpi --model_name uni_v1
```
When running extract_features_fp.py, also set --model_name to uni_v1, gigapath, VIRCHOW, or resnet50_trunc to use the respective encoder
### Traning
```
python3 main.py --project=$PROJECT_NAME --dataset_root=$DATASET_PATH --model_path=$OUTPUT_PATH --cv_fold=3 --title=mambamil --model=pure --baseline=mymamba --seed=2021
```

## Demo
```
cd ./Demo/
python demo.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
