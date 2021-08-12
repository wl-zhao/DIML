# DIML
Created by [Wenliang Zhao](https://wl-zhao.github.io/), [Yongming Rao](https://raoyongming.github.io/), [Ziyi Wang](https://github.com/LavenderLA), [Jiwen Lu](https://scholar.google.com/citations?user=TN8uDQoAAAAJ&hl=en&authuser=1), [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1)

This repository contains PyTorch implementation for paper **Towards Interpretable Deep Metric Learning with Structural Matching**.

We present a deep interpretable metric learning (DIML) that adopts a structural matching strategy to explicitly aligns the spatial embeddings by computing an optimal matching flow between feature maps of the two images. Our method enables deep models to learn metrics in a more human-friendly way, where the similarity of two images can be decomposed to several part-wise similarities and their contributions to the overall similarity. Our method is model-agnostic, which can be applied to off-the-shelf backbone networks and metric learning methods.

![intro](figs/intro.gif)

## Usage
### Requirement
- python3
- PyTorch 1.7

### Dataset Preparation
Please follow the instruction in [RevisitDML](https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch) to download the datasets and put all the datasets in `data` folder. The structure should be:
```
data
├── cars196
│   └── images
├── cub200
│   └── images
└── online_products
    ├── images
    └── Info_Files
```

### Training & Evaluation
To train the baseline models, run the scripts in `scripts/baselines`. For example:
```bash
CUDA_VISIBLE_DEVICES=0 ./script/baselines/cub_runs.sh
```
The checkpoints are saved in Training_Results folder.

To test the baseline models with our proposed DIML, first edit the checkpoint paths in `test_diml.py`, then run
```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/diml/test_diml.sh cub200
```
The results will be written to `test_results/test_diml_<dataset>.csv` in CSV format.

You can also incorporate DIML into the training objectives. We provide two examples which apply DIML to  Margin and Multi-Similarity loss. To train DIML models, run
```bash
# ./scripts/diml/train_diml.sh <dataset> <batch_size> <loss> <num_epochs>
# where loss could be margin_diml or multisimilarity_diml
# e.g.
CUDA_VISIBLE_DEVICES=0 ./scripts/diml/train_diml.sh cub200 112 margin_diml 150
```

## Acknowledgement
The code is based on [RevisitDML](https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch).
