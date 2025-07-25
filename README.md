# HiLA-MIL
Weakly Supervised Artificial Intelligence for Pan-cancer Detection of Lymph Node Metastasis on Whole Slide Images: Advantages for Isolated Tumor Cells 

## Prepare Patch Features
To preprocess WSIs, we used https://github.com/mahmoodlab/CLAM/tree/master#wsi-segmentation-and-patching.

## Traning
### Prepare Teacher Initiation Weight
```
python3 main.py --project=$PROJECT_NAME --dataset_root=$DATASET_PATH --model_path=$OUTPUT_PATH --cv_fold=3 --title=mambamil --model=pure --baseline=mambamil --seed=2021

```
### Teacher-Student Training
```
python3 main.py --project=$PROJECT_NAME --dataset_root=$DATASET_PATH --model_path=$OUTPUT_PATH --cv_fold=3 \
--teacher_init=./modules/init_ckp/c16_3fold_init_mambamil_seed2021 --title=mambamil_101_mr50h1-0r50_is --baseline=mambamil \
--num_workers=0 --cl_alpha=0.1 --mask_ratio_h=0.01 --mask_ratio_hr=0.5 --mrh_sche --init_stu_type=fc --mask_ratio=0.5 --mask_ratio_l=0. --seed=2021
```
