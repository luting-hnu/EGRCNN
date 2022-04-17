# EGRCNN
This repository provides the code for the methods and experiments presented in our paper '**Edge-guided Recurrent Convolutional Neural Network for Multi-temporal Remote Sensing Image Building Change Detection**'.
You can find the PDF of this paper on: https://ieeexplore.ieee.org/document/9524849

![net](https://github.com/luting-hnu/EGRCNN/blob/main/figure/net.png)

**If you have any questions, you can send me an email. My mail address is baibeifang@gmail.com.**

## Datasets
Download the building change detection dataset. 
- LEVIR-CD: https://justchenhao.github.io/LEVIR/
- WHU: https://study.rsgis.whu.edu.cn/pages/download/

In the following experiments, each image in the dataset is pre-cropped into multiple image patches of size 256 × 256.
## Directory structure
```
path to dataset:
                ├─train
                  ├─A
                  ├─B
                  ├─label
                  ├─label_edge
                ├─val
                  ├─A
                  ├─B
                  ├─label
                  ├─label_edge
                ├─test
                  ├─A
                  ├─B
                  ├─label
                  ├─label_edge
```
## Edge extraction
```bash
generate edges.py
```

## Train
```bash
train.py
```
## Test
You can use your own trained model or download our [pre-trained model](https://drive.google.com/file/d/1WhhCof1bfq1JCyHtsgmirGzW6tY0996x/view)

```bash
test.py
```

## Citation
If you find this paper useful, please cite:
```bash
Beifang Bai, Wei Fu, Ting Lu, and Shutao Li, "Edge-Guided Recurrent Convolutional Neural Network for Multitemporal Remote Sensing Image Building Change Detection," in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-13, 2022, Art no. 5610613, doi: 10.1109/TGRS.2021.3106697.
```
