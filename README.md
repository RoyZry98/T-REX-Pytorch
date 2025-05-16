# Trex

## Environment Setup

```bash
# Create a new conda environment
conda create -n trex python=3.10
conda activate trex

# Navigate to the project directory
cd ./trex
```

## Installation

1. Install PyTorch 2.5.1 according to your CUDA version from the [PyTorch official website](https://pytorch.org/get-started/locally/)

2. Install dependencies and project packages:
```bash
# Install requirements
pip install -r requirements.txt

# Install local packages in development mode
pip install -e ./transformers
pip install -e ./peft
```

## Dataset Preparation

Download the Trex dataset from Hugging Face:
```bash
# Download dataset
huggingface-cli download --repo-type dataset --resume-download leoboy20/trex_dataset --local-dir datasets

# Extract validation datasets
cd ./datasets
tar -xzvf npys_val_datasets.tar.gz
```

## Training

1. Configure your training settings:
   - Open the JSON configuration file in the `train_args` directory
   - Modify the `model_name_or_path` parameter to your desired pre-trained model

2. Start training:
```bash
bash trex_train.sh
```

## Evaluation

1. Configure your evaluation settings:
   - Modify `adapter_name_or_path` in `trex_eval.sh` to point to your trained adapter in the output directory
   - Update `model_name_or_path` to match your base model

2. Run evaluation:
```bash
bash trex_eval.sh
```


