# GroundLink

This repository has official implementation of GroundLink: A Dataset Unifying Human Body Movement and Ground Reaction Dynamics, SIGGRAPH Asia 2023

## Description
We introduce _GroundLinkNet_, a benchmark model trained with _GroundLink_.

## Getting Started

To reproduce our results, please follow the steps for installation and running the code for training and visualization. 

    .
    ├── Visualization  
    │   ├── aitviewer
    │   └── models              # copy SMPL models to models
    ├── GRF                     # scripts and data for GRF and CoP
    │   ├── checkpoints         # checkpoints and trained models
    │   ├── Data                # motion and force data prior to processing
    │   ├── ProcessedData       # preprocessed data for NN input
    │   ├── scripts             # tools for processing motion and force data
    │   └── train_smpl.py       # training script
    └── NN

### Data Download
We have released the motion and paired GRF/CoP data to our [data download page](https://csr.bu.edu/groundlink/). 
For simplicity, download SMPL-X models from the [page](https://csr.bu.edu/groundlink/), place it to [Visualization/models](./Visualization/models/) and extract. 

### Installation

The neural network structure and model design is inherited from [UnderPressure](https://github.com/InterDigitalInc/UnderPressure). Current setup requires Python 3.9.7 and Pytorch 1.10.2. We use [aitviewer](https://github.com/eth-ait/aitviewer) to visualize and interact with the 3D motion sequence reconstructed from [MoSh++](https://github.com/nghorbani/moshpp).

The code was tested on Ubuntu 20.04. To clone the repo and install dependencies:
```
git clone https://github.com/hanxingjian/GroundLink.git
cd GroundLink
conda create -n GroundLink python=3.9.7 -y
conda activate GroundLink
conda install -c pytorch pytorch=1.10.2 cudatoolkit=11.3 -y
pip install numpy
```



## Running the tests

### Preprocess Input Data

```
cd (root)/GRF/script
python preprocess.py
```

### Test Trained Models

We save our trained models in [GRF/checkpoint](./GRF/checkpoint/). We will release scripts for predictiong with trained models.

### Train with Motion and Force Data

### Set up Viewer

To install aitviewer (locally) with force plate coordinates setup:
```
cd (root)/Visualization
git clone git@github.com:eth-ait/aitviewer.git
mv forceplate.py aitviewer/
```


Navigate to configuration file and change the model location:
```cd aitviewer/aitviewer```

Edit L2 from [aitvconfig.yaml](https://github.com/eth-ait/aitviewer/blob/main/aitviewer/aitvconfig.yaml) to your location of SMPL-X models. If stricly following this repo including the folder structure above:
```L2: smplx_models: "./models/smplx_models"```

Install:
```cd .. && pip install -e .```

We provide sample data located in [./GRF/SampleData/](./GRF/SampleData/) and you can run the viewer with:
```python visualize_target_pred.py ``` 


## Authors


## License

<!-- This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details -->
TBD

## Acknowledgments

TBD
