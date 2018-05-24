# Multi-view Consistency as Supervisory Signal for Learning Shape and Pose Prediction

Shubham Tulsiani, Alexei A. Efros, Jitendra Malik.

[Project Page](https://shubhtuls.github.io/mvcSnP/)

<img src="https://shubhtuls.github.io/mvcSnP/resources/images/teaser.png" width="60%">

## Installation

First, you'll need a working implementation of Torch. The subsequent installation steps are:
```
##### Install 3D spatial transformer ######
cd external/stn3d
luarocks make stn3d-scm-1.rockspec

##### Additional Dependencies (json and matio) #####
sudo apt-get install libmatio2
luarocks install matio
luarocks install json
```

## Training and Evaluating
To train or evaluate the (trained/downloaded) models, it is first required to download the [Shapenet dataset (v1)](https://www.shapenet.org/) and [preprocess the data](docs/preprocessing.md) to compute renderings and voxelizations. Please see the detailed README files for [Training](docs/training.md) or [Evaluation](docs/evaluation.md) of models for subsequent instructions.

## Demo and Pre-trained Models
Please check out the [interactive notebook (coming soon)](demo/demo.ipynb) which shows reconstructions using the learned models. You'll need to - 
- Install a working implementation of torch and itorch.
- Download the [pre-trained models (1.5GB)](https://people.eecs.berkeley.edu/~shubhtuls/cachedir/mvcSnp/shapenet.tar.gz)  and extract them to 'cachedir/snapshots/shapenet/'
- Download [canonical frame alignments computed for pre-trained models](https://people.eecs.berkeley.edu/~shubhtuls/cachedir/mvcSnp/snet_alignments.tar.gz) and extract them to 'cachedir/alignment/shapenet/'
- Edit the path to the blender executable in the demo script.

### Citation
If you use this code for your research, please consider citing:
```
@inProceedings{mvcTulsiani18,
  title={Multi-view Consistency as Supervisory Signal
  for Learning Shape and Pose Prediction},
  author = {Shubham Tulsiani
  and Alexei A. Efros
  and Jitendra Malik},
  booktitle={Computer Vision and Pattern Regognition (CVPR)},
  year={2018}
}
```
