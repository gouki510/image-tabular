# Integrate image and tabular
## Prepare packages
```
conda create -n image-tabular python=3.8
conda activate image-tabular
pip install -r requirements.txt
```
## Downloads dataset
Use dataset [OSIC Pulmonary Fibrosis Progression](https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression/data) in kaggle comp.
```
kaggle competitions download -c osic-pulmonary-fibrosis-progression
```
## Adjust config parameter
You need change the content in config/config.yaml.

## Train
```
python train.py
```
