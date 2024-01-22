# GroundLink

This repository has official implementation of GroundLink: A Dataset Unifying Human Body Movement and Ground Reaction Dynamics, SIGGRAPH Asia 2023. 

This page is still under construction. Will update soon!


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
    │       ├── fbx             # this skeleton data is optional
    │       ├── Force           # force data
    │       ├── moshpp          # reconstructed pose and shape parameters 
    │       ├── AMASS           # TODO: add instruction to download amass data 
    │   ├── ProcessedData       # preprocessed data for NN input
    │   ├── scripts             # tools for processing motion and force data
    │   └── train_smpl.py       # training script
    └── NN

### Data Download
We have released the motion and paired GRF/CoP data to our [data download page](https://csr.bu.edu/groundlink/). 
Please also download SMPL-X models from the [page](https://smpl-x.is.tue.mpg.de/) (navigate to the Download page), place it to [Visualization/models](./Visualization/models/) and extract. 

### Installation

The neural network structure and model design is inherited from [UnderPressure](https://github.com/InterDigitalInc/UnderPressure). Current setup requires Python 3.9.7 and Pytorch 1.10.2. We use [aitviewer](https://github.com/eth-ait/aitviewer) to visualize and interact with the 3D motion sequence reconstructed from [MoSh++](https://github.com/nghorbani/moshpp).

The code was tested on Ubuntu 20.04 and Windows 11. To clone the repo and install dependencies:
```
git clone https://github.com/hanxingjian/GroundLink.git
cd GroundLink
conda create -y -n GroundLink python=3.9.7 numpy -c conda-forge
conda activate GroundLink
conda install -c pytorch pytorch=1.10.2 cudatoolkit=11.3 -y
```



## Running the tests

### Preprocess Input Data
You will need to run [preprocess.ipynb](./GRF/scripts/preprocess.ipynb) located in [/GRF/scripts](./GRF/scripts/) to generate preprocessed data for each participant. This will save the data to ```GRF/ProcessedData```.


TODO: add notebook for preparing external data including downloading dataset from AMASS page.

### Test Trained Models

We save our trained models in [GRF/checkpoint](./GRF/checkpoint/). Two pretrained models are provided including testing subject [s004](./GRF/checkpoint/pretrained_s4_noshape.tar) and [s007](./GRF/checkpoint/pretrained_s7_noshape.tar). We will demonstrate s007, and s004 will be similar:

Once preprocessed the data, navigate to ```GRF/ProcessedData/S7```, and create a testing folder, and copy all the files from [preprocess](./GRF/ProcessedData/S7/preprocessed/) to the new created test folder:

```
cd (root)/ProcessedData/S7
mkdir test
cp preprocessed/*.pth test/
```

Now run [Predict.ipynb](./GRF/scripts/Predict.ipynb) located in [/GRF/scripts](./GRF/scripts/). This step will:

1. Create a folder in ```(root)/GRF/ProcessedData/S7/prediction```
2. Predict GRF and CoP for user specified subject (currently s007) given the model. 



### Train with Motion and Force Data

### Set up Viewer

To install aitviewer (locally) with force plate coordinates setup:
```
cd (root)/Visualization
git clone git@github.com:eth-ait/aitviewer.git
mv forceplate.py aitviewer/
```

For Windows, use ```move```:
```
move forceplate.py aitviewer/
```



Navigate to configuration file and change the model location:
```cd aitviewer/aitviewer```

Edit L2 from [aitvconfig.yaml](https://github.com/eth-ait/aitviewer/blob/main/aitviewer/aitvconfig.yaml) to your location of SMPL-X models. If stricly following this repo including the folder structure above:
```L2: smplx_models: "./models/smplx_models"```

Edit L205 from [smpl.py](./Visualization/aitviewer/aitviewer/renderables/smpl.py) to use SMPL-X models:
```model_type="smplx"```

Install:
```cd .. && pip install -e .```

We provide sample data located in [./GRF/SampleData/](./GRF/SampleData/) and you can run the viewer with:
```python visualize_sample.py ``` 

If you have followed the steps on processing the data and have predicted results, use:
```python visualize_target_pred_s7.py ``` 

You may modify the trials to what you'd like to test.

## Citation
```
@inproceedings{
han2023groundlink,
title = {GroundLink: A Dataset Unifying Human Body Movement and Ground Reaction Dynamics},
author={Han, Xingjian and Senderling, Benjamin and To, Stanley and Kumar, Deepak and Whiting, Emily and Saito, Jun},
booktitle={ACM SIGGRAPH Asia 2023 Conference Proceedings},
year = {2023},
pages = {1--10},
}
```


## Update Logs
2023.12.02 Added scripts for mocap and force data preprocessing

2023.11.30 Added missing forceplate.py file

2023.09.08 Added sample data and visualization instruction 

2023.09.08 Project created


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) for details

## Acknowledgments

This project can be established thanks to the solid foundation laid by the giants before us. We are sincerely grateful for the help and support from the community. A special thanks to all the open source contributors, including but not limited to [UnderPressure](https://github.com/InterDigitalInc/UnderPressure), [AitViewer](https://github.com/eth-ait/aitviewer), [AMASS](https://amass.is.tue.mpg.de/).
