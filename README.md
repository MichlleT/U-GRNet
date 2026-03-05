## **Multi-scale Graph Reasoning Network for Discontinuity Segmentation in Borehole Images of Underground Coal Mines**

## Introduction

A U-shaped multi-scale Graph Reasoning Network for the segmentation of discontinuous regions in borehole images, named U-GRNet.

This new method addresses a long-standing challenge in subsurface imaging: automated segmentation of geological discontinuities in borehole images characterized by strong anisotropy, long-range continuity, and category imbalance.

![](figs/results_all.png)
Segmentation visualization results in the full borehole image.

## Requirements

1. Environment
   ```
   torch>=1.12.1
   torchvision>=0.13.1
   numpy<=1.19.5
   ```
2. Dataset
   ```
   Your_datasets/
      train/
          images/
            001.png
            002.png
            ...
          mask/
            001.png
            002.png
            ...
      val/
          images/
            001.png
            002.png
            ...
          mask/
            001.png
            002.png
            ...
   ```
## Getting Started

1. 🚀 Quick Start
For a quick test, you can download our weights trained on drill images here at https://pan.baidu.com/s/1ZzkVljoXAeQam3nEx7F5OQ?pwd=iykz passwords: iykz
```bash
cd tools/
python testings.py -i ../datasets/inputs/your_test_dir -r your_save_img/
```
2. Training & Evaluation in Command Line
```bash
cd tools/
python trainings.py -i ../datasets/inputs/your_test_dir -e 100 -o ../datasets/outputs/checkpoint/
```

## Citation

If you use the U-GRNet codes or the U-GRNet dataset, please cite our paper:

```bibtex
@article{T20260303,
title = {Multi-scale Graph Reasoning Network for Discontinuity Segmentation in Borehole Images of Underground Coal Mines},
author = {},
journal = {},
volume = {},
pages = {},
year = {},
issn = {},
}
```
## License
U-GRNet is released under the Apache 2.0 license.
