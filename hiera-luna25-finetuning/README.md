# Finetuning Hiera on LUNA25 3D Patches

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)
[![Code License](https://img.shields.io/badge/code_license-Apache_2.0-olive)](https://opensource.org/licenses/Apache-2.0)

This work, participating in the [LUNA25 Challenge](https://luna25.grand-challenge.org/), finetuned Hiera for lung nodule malignancy risk estimation.

## Usage
### Set Up
```
git clone https://github.com/Zhengro/hiera-luna25-finetuning.git

conda create -y -n venv2025 python==3.9
conda activate venv2025
python -m pip install -r hiera-luna25-finetuning/requirements.txt
```

### Get Data
```
wget https://zenodo.org/records/14223624/files/luna25_nodule_blocks.zip.001
wget https://zenodo.org/records/14223624/files/luna25_nodule_blocks.zip.002
wget https://zenodo.org/records/14673658/files/LUNA25_Public_Training_Development_Data.csv

cat luna25_nodule_blocks.zip.001 luna25_nodule_blocks.zip.002 > luna25_nodule_blocks.zip
apt update && apt -y install p7zip-full
7z x luna25_nodule_blocks.zip
rm luna25_nodule_blocks.zip.001 luna25_nodule_blocks.zip.002
```

### Perform Training
Before starting the training, ensure that the correct `experiment_config.py` is in the project root directory.

By default, `experiment_config.py` is a copy of `configs/experiment_config_3d_hiera.py`, which represents the submitted version for the LUNA25 Challenge. To train 2D models, copy one of the available config files:
```
cp hiera-luna25-finetuning/configs/experiment_config_XXX.py hiera-luna25-finetuning/experiment_config.py
```
When training using k-fold cross validation, set `kind="k-fold"` in `experiment_config.py` then run
```
python hiera-luna25-finetuning/split_data.py
python hiera-luna25-finetuning/finetune.py --k_fold
```
When training using all data, set `kind="all-data"` in `experiment_config.py` then run
```
python hiera-luna25-finetuning/finetune.py --all_data
```

## License
This work's code is licensed under the [Apache License, Version 2.0](https://opensource.org/licenses/Apache-2.0).

## References

This work has been adapted from and is inspired by the following repositories:

- [luna25-baseline-public](https://github.com/DIAGNijmegen/luna25-baseline-public)
- [hiera](https://github.com/facebookresearch/hiera)
- [SlowFast](https://github.com/facebookresearch/SlowFast)
- [torchvision](https://github.com/pytorch/vision/tree/main/torchvision)