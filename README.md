## DeepCLIP

Welcome to DeepCLIP's git repository. 

DeepCLIP is a novel deep learning tool for finding binding-preferences of RNA-binding proteins. Use pre-trained models or train your own at [http://deepclip.compbio.sdu.dk/](http://deepclip.compbio.sdu.dk).

In this repository, you can find all relevant code for running DeepCLIP on your local machine.

Summary of DeepCLIP and its functionalities:
* DeepCLIP is a neural network with shallow convolutional layers connected to a bidirectional LSTM layer.
* DeepCLIP can calculate binding profiles and pseudo position frequency matrices.
* Binding profiles show whether areas of sequences contain possible binding sites or whether they look like random genomic background.
* DeepCLIP outperforms current state-of-the-art RNA-binding protein motif discovery tools on curated CLIP datasets.

## Table of contents
* [Requirements](#requirements)
* [Installation](#installation)
* [Citation](#citation)
* [Contributors](#contributors)

## Requirements
DeepCLIP was designed to run on Linux flavoured operating systems and while it may run on Windows or FreeBSD flavours such as OS-X we do not actively support this.

DeepCLIP requires Python 2.7 along with the latest versions of Theano and Lasagne.
To install requirements for DeepCLIP, please install Theano and then Lasagne, followed by the remaining requirements:
```shell
pip install git+git://github.com/Theano/Theano.git.
pip install https://github.com/Lasagne/Lasagne/archive/master.zip
pip install mkl-service
pip install scikit-learn
pip install matplotlib
pip install biopython
pip install htseq
```

We recommend using conda to install a DeepCLIP specific environment along with DeepCLIP requirements:
```shell
conda create -n deepclip python=2.7 mkl-service numpy scipy scikit-learn biopython htseq matplotlib
conda activate deepclip
pip install git+git://github.com/Theano/Theano.git
pip install https://github.com/Lasagne/Lasagne/archive/master.zip
```

## Installation

DeepCLIP can be run from within its source directory, so to install DeepCLIP simply clone this repository and add it to your path environment variable.
```shell
git clone http://github.com/deepclip/deepclip
```

In order to run DeepCLIP, simply run DeepCLIP.py:
```shell
$install_folder/DeepCLIP.py
```

## Citation
Link to citable paper about DeepCLIP will be posted soon.

## Contributors
Main DeepCLIP development
* Alexander Gulliver Bjørnholt Grønning
* Thomas Koed Dokor

Additional code
* Simon Jonas Larsen
