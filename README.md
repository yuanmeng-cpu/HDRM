# HDRM
This is the implementation of HDRM

- Model framework

![pdf](https://github.com/yuanmeng-cpu/HDRM/blob/main/www25.pdf)

# Requirements
PyTorch  1.11.0

Python  3.8(ubuntu20.04)

Cuda  11.3

scikit-learn 0.23.2

numpy 1.19.1

scipy 1.5.4

tqdm 4.48.2

## Code Structures

```
.
├── hgcn_utils
│   ├── math_utils.py
│   ├── train_utils.py
├── manifolds
│   ├── base.py
│   ├── hyperboloid.py
├── rgd
│   ├── rsgd.py
├── dataloader.py
├── main.py
├── model.py
├── parse.py
├── Procedure.py
├── register.py
├── train.py
├── utils.py
└── world.py

```

# Run Code
python train.py


## Dataset

|  Dataset   |  Users  |  Items(C)  |  Interactions(C)  |   Item(N)   |   Interactions |
|:----------:|:--------:|:--------:|:---------------:|:-----------:|:-----------:|
|    Amazon-book    |  1088,822   |  94,949   |     3,146,256     | 178,181  |3,145,223  |
|  Yelp2020   |  54,574   |  34,395  |    1,402,736    | 77,405  |1,471,675  |
|  ML-1M  |  5,949   |  2,810  |     571,531     | 3,494  |618,297  |
