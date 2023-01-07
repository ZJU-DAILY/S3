# A Lightweight Framework for Fast Trajectory Simplification

![image-20230107134547723](C:/Users/ASUS/AppData/Roaming/Typora/typora-user-images/image-20230107134547723.png)

## Introduction

This repository holds source code for the paper "A Lightweight Framework for Fast Trajectory Simplification".


## Environment Preparation
  - Python 3.8.8 (Recommend Anaconda)
  - CentOS 7.0
  - Pytorch == 1.8.1+cu111
  - A Nvidia GPU with cuda 11.7
  - Please refer to the `requirement.txt` to install all required packages of Python.


## Datasets Description & Preprocessing
Prepare your own trajectory data with the following format and each trajectory is split by "\n". Put the data file into `./datasets` and run `python preprocess/createDataset.py`. The preprocessed datasets will be split into training sets, validation sets, and testing sets which is stored in `./datasets` folder.

> [[116.51172, 39.92123], [116.51135, 39.93883],  [116.69171, 39.85182]]
> [[116.69171, 39.85184], [116.6917, 39.85184], [116.6916, 39.85177]]

```shell
python preprocess/createDataset.py
```

Besides, prepare the roadmap data named as "edge.edgelist" which is corresponding to the trajectory. Run the code like `python preprocess/node2vec_main.py --input datasets/edge.edgelist --output datasets/beijing.emd` to generate your own road node embedding vector with semantic information.

```shell
python preprocess/node2vec_main.py --input datasets/edge.edgelist --output datasets/beijing.emd
```

## Running Procedures

### Hyperparameters

You can create a config file like `model_configs/s3.yaml`and specify your own hyperparameters for better performance, including learning rate, loss function, min/max compression ratio and some network layer dimension.

### Training

Run `s3.py`, the generated models will be stored in the folder `./checkpoints` automatically, and you can pick one model with best performance as your model to do some experiments.

```shell
python models/s3.py --config ./model_configs/camera/s3.yaml
```

Note: you can choose any config yaml file with the format like `--config ./model_configs/camera/s3.yaml`.

### Evaluation

Run `debug_outputs.py` to generate the compressed trajectory and constructed trajectory and the output will be saved in the folder `./evaluation`.

```shell
python generate/debug_outputs.py
```

## Citation

If you use our code for research work, please cite our paper as below:

```tex
@article{,
  title={A Lightweight Framework for Fast Trajectory Simplification},
  author={Ziquan Fang, Changhao He, Lu Chen, Danlei Hu, Qichen Sun, Linsen Li, Yunjun Gao},
  year={2023},
}
```

