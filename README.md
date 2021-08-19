# EGRCNN
Edge-guided Recurrent Convolutional Neural Network for Multi-temporal Remote Sensing Image Building Change Detection
![net](https://github.com/luting-hnu/EGRCNN/blob/main/figure/net.png)
## Edge extraction
```bash
change_dataset_np.py
```
## Datasets
```
path to LEVIR-CD/WHU dataset:
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
