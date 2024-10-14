# HDRM
This is the implementation of HDRM

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

# Run Code
python train.py


## Dataset

|  Dataset   |  Users  |  Items  |  Interactions  |   Density   |
|:----------:|:--------:|:--------:|:---------------:|:-----------:|
|    Food    |  7,809   |  6,309   |     216,407     | 4.4 × 10⁻³  |
|  KuaiRec   |  7,175   |  10,611  |    1,153,797    | 1.5 × 10⁻³  |
|  Yelp2018  |  8,090   |  13,878  |     398,216     | 3.5 × 10⁻³  |
|   Douban   |  8,735   |  13,143  |     354,933     | 3.1 × 10⁻³  |
