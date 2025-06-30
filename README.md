# HDRM  
This is the implementation of HDRM  [Hyperbolic Diffusion Recommender Model](https://arxiv.org/pdf/2504.01541)
Meng Yuan, Yutian Xiao, Wei Chen, Chu Zhao, Deqing Wang, and Fuzhen Zhuang.  
In Proceedings of The 2025 Web Conference (WWW 2025)


- Model framework

![png](https://github.com/yuanmeng-cpu/HDRM/blob/main/hdrm.png)

# Requirements
PyTorch  1.11.0

Python  3.8(ubuntu20.04)

Cuda  11.3

scikit-learn 0.23.2

numpy 1.19.1

scipy 1.5.4

tqdm 4.48.2

# Datasets
The Amazon-book dataset can be downloaded through the following link:
[Amazon-book Dataset Download Link](https://drive.google.com/file/d/1QYmTpnChuii9CvPBWFecFmhUHczJtjan/view)
You can download the pre-trained weights through the following link.
[pre-trained weights Download Link](https://drive.google.com/drive/folders/1wFmQbXPj-bxUct4frUZ8U5i4zaD3n7OQ?usp=sharing
)

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

# Run ML-1M
python train_ml.py

# Run Amazon-book
python train_amazon.py

# Run Yelp2020
python train_yelp.py


## Dataset

|  Dataset   |  Users  |  Items(C)  |  Interactions(C)  |   Item(N)   |   Interactions(N) |
|:----------:|:--------:|:--------:|:---------------:|:-----------:|:-----------:|
|    Amazon-book    |  1088,822   |  94,949   |     3,146,256     | 178,181  |3,145,223  |
|  Yelp2020   |  54,574   |  34,395  |    1,402,736    | 77,405  |1,471,675  |
|  ML-1M  |  5,949   |  2,810  |     571,531     | 3,494  |618,297  |
