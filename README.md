# MTA-CLIP: Language-Guided Semantic Segmentation with Mask-Text Alignment
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)


This is the official repository accompanying the ECCV paper:

[Anurag Das](https://anurag-198.github.io/), [Xinting Hu](https://joyhuyy1412.github.io/), [Li Jiang](https://llijiang.github.io/), and [Bernt Schiele](https://scholar.google.com/citations?user=z76PBfYAAAAJ&hl=en). **MTA-CLIP: Language-Guided Semantic Segmentation with Mask-Text Alignment**. ECCV 2024.

[Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07040.pdf) | [Video](https://www.youtube.com/watch?v=TYAOs8EYHNA&t=1s) | [Supplemental](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07040-supp.pdf)


## Usage

#### For Conda:
Create a conda environment using the provided environment.yml file:

```bash
conda env create -f environment.yml
conda activate m200
```

### Experiment setup:

#### Preparing the data:
1. The dataset supported are Cityscapes and ADE20k. The dataset directory should have the following structure:

```
data
 ade/
 ├── ADEChallengeData2016/
 cityscapes/
 ├── leftImg8bit/
 ├── gtFine/
```

#### Training:
1. Training
```bash
python3 -m torch.distributed.launch --nproc_per_node=2 train.py local_configs/r50_1.py --resume --launcher pytorch ${@:3}
```

## Citation

If you find our work useful, please consider citing our paper:

```
@inproceedings{10.1007/978-3-031-72949-2_3,
author = {Das, Anurag and Hu, Xinting and Jiang, Li and Schiele, Bernt},
title = {MTA-CLIP: Language-Guided Semantic Segmentation with Mask-Text Alignment},
year = {2024},
isbn = {978-3-031-72948-5},
publisher = {Springer-Verlag},
address = {Berlin, Heidelberg},
url = {https://doi.org/10.1007/978-3-031-72949-2_3},
doi = {10.1007/978-3-031-72949-2_3},
booktitle = {Computer Vision – ECCV 2024: 18th European Conference, Milan, Italy, September 29–October 4, 2024, Proceedings, Part LIV},
pages = {39–56},
numpages = {18},
keywords = {Scene Understanding, Vision Language Models},
location = {Milan, Italy}
}
```
## Acknowledgement

Code is largely based on Mask2Former (https://github.com/facebookresearch/Mask2Former) and DenseCLIP (https://github.com/raoyongming/DenseCLIP).
