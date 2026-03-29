
# Model Card for https://github.com/YAOSL98/HiLA-MIL

## Model Details

### Model Description
HiLA-MIL is a weakly supervised learning framework designed for pan-cancer lymph node metastasis detection on whole slide images (WSIs). It demonstrates significant advantages in identifying isolated tumor cells (ITCs), a critical and challenging task in clinical pathology. The framework integrates advanced encoders (UNI, Gigapath, VIRCHOW, ResNet50) and uses a Mamba-based MIL backbone for efficient feature modeling on gigapixel pathological images.

- **Developed by:** Lili Sun, Shuilian Yao and our research team.
- **Model type:** Weakly Supervised Deep Learning / Multiple Instance Learning (MIL) / Computational Pathology Model
- **Language(s) (NLP):** Not applicable (non-NLP, medical image model)
- **License:** MIT License

## Uses
### Direct Use
This model can be directly used for pan-cancer lymph node metastasis detection in whole slide images (WSIs), especially for the recognition and localization of isolated tumor cells (ITCs) under weakly supervised conditions (without pixel-level annotations). It supports WSI feature extraction, patch processing, and end-to-end metastasis prediction.

### Out-of-Scope Use
- Not intended for clinical diagnosis or direct patient treatment decisions without professional pathologist validation.
- Not suitable for non-cancer, non-lymph-node, or non-WSI image types.
- Not designed for use with insufficient GPU memory or incompatible hardware environments.
- Not validated for extremely low-quality, blurred, or damaged pathological slides

## Bias, Risks, and Limitations
- Data Limitation: Performance may vary across different cancer types, patient populations, and scanners due to variations in WSI staining and imaging protocols.
- Clinical Risk: The model provides auxiliary analysis only and must not replace professional pathological diagnosis.
- ITC Sensitivity: While optimized for ITCs, very tiny or sparse tumor cells may still lead to false negatives in some cases.
- Domain Bias: Model performance is dependent on the distribution of training datasets; external validation across multiple medical centers is required for real-world deployment

### Recommendations
- Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model.
- Always perform external validation on multi-center datasets before clinical application.
- Use the model only as an auxiliary tool for pathologists.
- Ensure consistent WSI preprocessing and staining standards to maintain stable performance.

## How to Get Started with the Model
```
# Environment setup
conda create -n hila-mil python=3.10
conda activate hila-mil
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
pip install ./mamba_requirements/causal_conv1d*.whl
pip install ./mamba_requirements/mamba_ssm*.whl

# Preprocessing
cd CLAM_ours/
python create_patches_fp.py --source DATA_DIR --save_dir RESULTS_DIR --patch_size 256 --seg --patch --stitch --patch_level 1
CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py --data_h5_dir COORDS_DIR --data_slide_dir DATA_DIR --csv_path FILE.csv --feat_dir FEAT_DIR --batch_size 512 --model_name uni_v1

# Training
python3 main.py --project=PROJECT --dataset_root=DATASET --model_path=OUTPUT --cv_fold=3 --title=mambamil --model=pure --baseline=mymamba --seed=2021

# Demo
cd Demo/
python demo.py
```

## Training Details

### Training Data
The model is trained on lymph node whole slide images (WSIs) for pan-cancer metastasis detection. Datasets include multiple clinical centers with various cancer types, and support multi-center validation as described in our paper.

### Training Procedure

#### Preprocessing
WSI segmentation, patching (256×256), stitching.

Feature extraction using UNI, Gigapath, VIRCHOW, or ResNet50.

Implementation based on the CLAM framework.

#### Training Hyperparameters
```
Training regime: FP32 (full precision training)
Cross-validation: 10-fold
Backbone: Mamba-MIL (custom mymamba)
Seed: 2021
```
#### Speeds, Sizes, Times [optional]

| Feature Dimension | Corresponding Encoder | Input Shape | 4096 | Model Size (MB) | Trainable Parameters (M) | Average Inference Time (ms) |
| :---------------- | :-------------------- | :---------- | :--------------- | :-------------- | :---------------------- | :-------------------------- |
| 1024              | UNI/ResNet            | [4096, 1024] | Number of patches | 25.94 | 6.80 | 3.68 |
| 1536              | Gigapath              | [4096, 1536] | Number of patches | 57.66 | 15.11 | 7.41 |
| 2560              | Virchow               | [4096, 2560] | Number of patches | 158.59 | 41.57 | 19.42 |

*All tests were conducted on an NVIDIA GeForce RTX 3090 (24GB) GPU.*

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data
Five multi-center independent datasets from different clinical centers (as reported in the revised paper).

#### Factors
Cancer types, Clinical centers, ITC proportion, WSI magnification and staining conditions.

#### Metrics
Standard pathological image evaluation metrics including accuracy, AUC, precision, recall, F1-score, and specificity.

### Results
Based on Gigapath backbone.
| Group    | Methods                     | AUC           | Accuracy      | Precision     | Recall        | Specificity   | F1-Score      |
|----------|-----------------------------|---------------|---------------|---------------|---------------|---------------|---------------|
| Overall  | MambaMIL+HiLA-MIL (Ours)    | 0.9759±0.0114 | 0.9291±0.0307 | 0.8522±0.0599 | 0.8678±0.0560 | 0.9774±0.0095 | 0.8468±0.0610 |
| Negative | MambaMIL+HiLA-MIL (Ours)    | 1.0000±0.0000 | 0.9985±0.0048 | 1.0000±0.0000 | 0.9957±0.0137 | 1.0000±0.0000 | 0.9978±0.0070 |
| ITC      | MambaMIL+HiLA-MIL (Ours)    | 0.9707±0.0149 | 0.9578±0.0220 | 0.6289±0.1413 | 0.8750±0.2318 | 0.9631±0.0213 | 0.7226±0.1177 |
| Micro    | MambaMIL+HiLA-MIL (Ours)    | 0.9427±0.0302 | 0.9351±0.0254 | 0.8151±0.1301 | 0.6361±0.1566 | 0.9776±0.0164 | 0.7024±0.1225 |
| Macro    | MambaMIL+HiLA-MIL (Ours)    | 0.9900±0.0090 | 0.9668±0.0185 | 0.9649±0.0230 | 0.9645±0.0355 | 0.9689±0.0209 | 0.9642±0.0204 |

#### Summary
HiLA-MIL provides reliable weakly supervised WSI-based metastasis detection and offers clear advantages for challenging ITC recognition tasks. Multi-center validation confirms its robustness and generalization ability.}

## Environmental Impact
- Hardware Type: NVIDIA GeForce RTX 3090 (24GB)
- Cloud Provider: None (local compute)
- Compute Region: Not applicable
- Carbon Emitted: Not estimated

## Technical Specifications [optional]

### Model Architecture and Objective
- Architecture: Mamba-based Multiple Instance Learning (MIL)
- Objective: Weakly supervised pan-cancer lymph node metastasis & ITC detection in WSIs

### Compute Infrastructure
#### Hardware
```
OS: Linux (x86_64)
GPU: NVIDIA RTX 3090 (24GB)
CUDA: 11.8
```

#### Software
```
Python: 3.10
PyTorch: 2.1.1
Libraries: torchvision, torchaudio, mamba_ssm, causal_conv1d, CLAM-based preprocessing
```

**BibTeX:**
```
@article{,
  title={Weakly Supervised Artificial Intelligence for Pan-cancer Detection of Lymph Node Metastasis on Whole Slide Images},
  author={},
  journal={},
  year={}
}
```

**APA:**
(To be updated with published paper information)


## Glossary [optional]
- WSI: Whole Slide Image
- MIL: Multiple Instance Learning
- ITC: Isolated Tumor Cells
- Mamba: Selective State Space Model for efficient sequence modeling

## Model Card Contact
GitHub: https://github.com/YAOSL98/HiLA-MIL
