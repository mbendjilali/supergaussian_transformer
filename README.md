# supergaussian_transformer

## Overview

This repository is mostly based on the SPT (Super Point Transformer) codebase (https://github.com/drprojects/superpoint_transformer). Our primary contribution is the implementation of the hierarchical Gaussian Mixture Model (GMM) algorithm in C++ for superpoint grouping, its integration to the preprocessing scripts, and the configuration files needed to run experiments on both DALES and S3DIS.

## Our Contribution

- **src/utils/cpp/gaussian_mixture.cpp**: Core GMM algorithm implementation

**Note:** Other files in `src/utils/cpp` (e.g., headers, bindings, build scripts) support the Python extension and are for further research.

**Note:** Other files in the repository have been modified to acclimate SPT's scripts to hGMM, like src/utils/partition.py, src/utils/distance/neighbors.py.

## GMM Configurations

GMM datamodule configurations are located in the `configs/datamodule/semantic` directory:

- `configs/datamodule/semantic/s3dis_gmm.yaml` (S3DIS dataset)
- `configs/datamodule/semantic/dales_gmm.yaml` (DALES dataset)

## Setup

Follow Superpoint Transformer's README instructions to set up your environment or just run :
```bash
bash install.sh
```

Build the hGMM algorithm :
```bash
cd /home/moussabendjilali/supergaussian_transformer/src/utils/cpp
pip install -e .
```

## Running Experiments

To train the model, you need to download and place the datasets's zip files following Superpoint Transformer's format : data/dales/dales.zip. Then, to run the 11Gb-GPU experiments:

S3DIS 11G:
```bash
python train.py experiment=semantic/s3dis_11g datamodule.lite_preprocessing=True
```

DALES 11G:
```bash
python train.py experiment=semantic/dales_11g datamodule.lite_preprocessing=True
```

These commands automatically load the appropriate datamodule and model settings based on the experiment configuration.

## Packaging the Repository

To zip the repository while excluding heavy datasets and log files:

```bash
cd /home/moussabendjilali/supergaussian_transformer
zip -r supergaussian_transformer.zip . -x "data/*" "logs/*" "*.log" "src/utils/cpp/*.so" "src/utils/cpp/build/*"
```

Alternatively, using an absolute-path invocation:

```bash
zip -r /home/moussabendjilali/supergaussian_transformer.zip /home/moussabendjilali/supergaussian_transformer -x "/home/moussabendjilali/supergaussian_transformer/data/*" "/home/moussabendjilali/supergaussian_transformer/logs/*" "*.log" "/home/moussabendjilali/supergaussian_transformer/src/utils/cpp/*.so" "/home/moussabendjilali/supergaussian_transformer/src/utils/cpp/build/*"
```

## License

This project inherits the original SPT license.