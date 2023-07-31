# Deep Equilibrium Multimodal Fusion
PyTorch implementation of the paper: Deep Equilibrium Multimodal Fusion [[arXiv](https://arxiv.org/pdf/2306.16645.pdf)].

## Installation

Please clone this repo and use the following command to setup the environment:

```bash
conda create -n deqfusion python==3.8 pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=10.2 -c pytorch
conda activate deqfusion
pip install -r requirements.txt
```

## Usage

### BRCA

The code is modified from the [official implementation of MM-Dynamics](https://github.com/TencentAILabHealthcare/mmdynamics).

Please follow the command below to train a model on BRCA:

```bash
cd experiments/BRCA
python main.py --mode=train -mrna -dna -mirna --f_thres=105 --b_thres=106
```

Please run `python main.py -h` for more details.

### MM-IMDB

The code is modified from [MultiBench](https://github.com/pliang279/MultiBench).

Please first download MM-IMDB dataset from [here](https://archive.org/download/mmimdb/multimodal_imdb.hdf5) and place it under directory `experiments/MM-IMDB`.

There are several example scripts for running the experiments using different fusion strategies. To train a model with our DEQ fusion on MM-IMDB, please run:

```bash
cd experiments/MM-IMDB
python examples/multimedia/mmimdb_deq.py
```

### CMU-MOSI

The code is modified from the [official implementation of Cross-Modal BERT](https://github.com/thuiar/Cross-Modal-BERT).

First download the pre-trained BERT model from [Google Drive](https://drive.google.com/file/d/1dKSzsgXORN7WVaJJYvNzqFPCQbn-aJcb/view?usp=sharing), or from [Baidu Netdisk](https://pan.baidu.com/s/1G3VaV0kqwYkOEFNst2rfVw) with code `fuse`, or use the following command:

```bash
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
```

Run the experiments by:

```bash
cd experiments/CMU-MOSI
python run_classifier.py
```

### Custom Dataset

If you wish to use DEQ fusion in your own dataset, please copy `DEQ_fusion.py`, `solver.py`, and `jacobian.py` into your repo. Then run `from DEQ_fusion import DEQFusion` and use `DEQFusion` for multimodal fusion.

## Acknowledgement

Our work benefits largely from [DEQ](https://github.com/locuslab/deq), [MDEQ](https://github.com/locuslab/mdeq), [MM-Dynamics](https://github.com/TencentAILabHealthcare/mmdynamics), [MultiBench](https://github.com/pliang279/MultiBench), and [Cross-Modal BERT](https://github.com/thuiar/Cross-Modal-BERT).
