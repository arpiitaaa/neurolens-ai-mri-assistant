# NeuroLens AI: Agentic Brain MRI Research Assistant
 
A research project that aims to build an end-to-end system for brain MRI analysis: segmenting suspicious regions, estimating how confident the model is, and generating a radiology-style evidence report, with an agent that flags uncertain cases for human review.
 
> **Disclaimer:** This is a research prototype built for learning and experimentation. It is **not** a medical device and must not be used for diagnosis or any clinical decision.
 
## How it will work
 
```
MRI scan (4 modalities)
        |
        v
 3D segmentation  --->  tumour mask
        |
        v
 uncertainty map  --->  confidence score
        |
        v
 agent: quality check
   |-- confidence OK  ---> evidence report
   |-- confidence low ---> second analysis ---> still low? flag for human review
```
 
## What is implemented
 
- **Data pipeline** (`src/segmentation/data.py`): loads the Medical Segmentation Decathlon brain tumour dataset (FLAIR, T1w, T1gd, T2w), reorients scans, z-score normalizes each modality over brain voxels, and samples random 96x96x96 training patches with flip and intensity augmentation.
- **Model** (`src/segmentation/model.py`): 3D SegResNet with dropout, kept in the design so it can be reused for Monte Carlo dropout uncertainty.
- **Training and validation** (`src/segmentation/train.py`): Dice + cross-entropy loss, AdamW with cosine learning-rate decay, mixed precision on GPU, sliding-window inference for validation, and per-class Dice reporting (edema, non-enhancing tumour, enhancing tumour). The best checkpoint is saved automatically.
## Dataset
 
[Medical Segmentation Decathlon](http://medicaldecathlon.com/), Task 01: Brain Tumour (484 labeled multimodal MRI cases). The data is not stored in this repository.
 
Label classes: `0` background, `1` edema, `2` non-enhancing tumour, `3` enhancing tumour.
 
## Results
 
Baseline run on Google Colab (T4 GPU): 100 training cases, 20 validation cases, 20 epochs, 96x96x96 patches, seed 0.
 
| Class | Dice |
|-------|------|
| Edema | 0.720 |
| Non-enhancing tumour | 0.433 |
| Enhancing tumour | 0.693 |
| **Mean** | **0.615** |
 
Validation Dice over training (mean of the three classes):
 
| Epoch | 4 | 8 | 12 | 16 | 20 |
|-------|---|---|----|----|----|
| Mean Dice | 0.179 | 0.296 | 0.536 | 0.575 | 0.615 |
 
Notes: this is a single baseline run on a subset of the data with a small validation set, so the numbers are indicative rather than a benchmark. Dice was still improving at epoch 20, so longer training on the full dataset should score higher. Non-enhancing tumour is the hardest class.
 
## Quick start
 
```bash
git clone https://github.com/arpiitaaa/neurolens-ai-mri-assistant.git
cd neurolens-ai-mri-assistant
python -m venv .venv
# Windows: .venv\Scripts\activate    Mac/Linux: source .venv/bin/activate
pip install torch torchvision          # pick the build for your hardware at pytorch.org
pip install "monai[nibabel,tqdm]" requests
```
 
Quick pipeline check (downloads about 7 GB, runs on CPU):
 
```bash
python -m src.segmentation.train --data-dir data --download --epochs 2 --max-train 4 --max-val 2
```
 
### Training on a GPU (Google Colab)
 
Training needs a GPU. The run reported above used a free Colab T4 with the dataset archive stored on Google Drive, so it only has to be downloaded once:
 
```bash
# one time: save the dataset archive to Drive (resumable)
wget -c -P /content/drive/MyDrive/msd https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar
 
# every new session: unpack the data
mkdir -p /content/data
tar -xf /content/drive/MyDrive/msd/Task01_BrainTumour.tar -C /content/data
 
# train
python -m src.segmentation.train --data-dir /content/data --epochs 20 \
  --max-train 100 --max-val 20 --val-every 4 \
  --out-dir /content/drive/MyDrive/neurolens-checkpoints
```
 
Remove `--max-train` and `--max-val` to train on the full dataset (much slower per epoch).
 
## Repository structure
 
```
neurolens-ai-mri-assistant/
├── src/
│   ├── segmentation/   # data loading, model, training
│   ├── agent/          # quality control and human-review logic
│   ├── vlm/            # report generation
│   ├── ssl/            # self-supervised pretraining
│   └── diffusion/      # denoising experiments
├── notebooks/
├── reports/
├── tests/
└── requirements.txt
```
 
## Roadmap
 
1. Train on the full dataset for more epochs and report validation Dice.
2. Add Monte Carlo dropout uncertainty maps and a confidence score per case.
3. Build the agent that checks segmentation quality, requests a second analysis when confidence is low, and flags cases for human review.
4. Generate structured, radiology-style reports from the segmentation and uncertainty outputs.
5. Explore self-supervised pretraining and diffusion-based denoising.
## Acknowledgements
 
- [MONAI](https://monai.io/) for the medical imaging framework.
- Antonelli et al., "The Medical Segmentation Decathlon", *Nature Communications*, 2022, for the dataset.