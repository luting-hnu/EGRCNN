# EGRCNN
Edge-guided Recurrent Convolutional Neural Network for Multi-temporal Remote Sensing Image Building Change Detection
![net](https://github.com/luting-hnu/EGRCNN/blob/main/figure/net.png)
## Edge extraction
```bash
change_dataset_np.py
```
## Datasets
Download the building change detection dataset
- LEVIR-CD: https://justchenhao.github.io/LEVIR/
- WHU: https://study.rsgis.whu.edu.cn/pages/download/
## Directory structure
```
path to dataset:
                ├─train
                │  ├─A
                │  ├─B
                │  ├─label
                │  ├─label_edge
                ├─val
                │  ├─A
                │  ├─B
                │  ├─label
                │  ├─label_edge
                ├─test
                │  ├─A
                │  ├─B
                │  ├─label
                │  ├─label_edge
```
## Train
```bash
train.py
```
## Test
```bash
test.py
```
