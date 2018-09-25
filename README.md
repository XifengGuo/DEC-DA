# Deep Embedded Clustering with Data Augmentation (DEC-DA)
State-of-the-art clustering performance on (duo to Jun 30, 2018):
- MNIST (acc=0.985, nmi=0.960) 
- MNIST-TEST (acc=0.983, nmi=0.958) 
- USPS (acc=987, nmi=962)
- Fashion-MNIST (acc=0.586, nmi=0.636)

Tensorflow implementation for ACML-2018 paper:
* Xifeng Guo, En Zhu, Xinwang Liu, Jianping Yin. Deep Embedded Clustering with Data Augmentation. 
The 10th Asian Conference on Machine Learning (ACML), 2018.

## Abstract
Deep Embedded Clustering (DEC) surpasses traditional clustering algorithms by jointly performing feature learning and cluster assignment. 
Although a lot of variants have emerged, they all ignore a crucial ingredient, \emph{data augmentation}, 
which has been widely employed in supervised deep learning models to improve the generalization. 
To fill this gap, in this paper, we propose the framework of Deep Embedded Clustering with Data Augmentation (DEC-DA). 
Specifically, we first train an autoencoder with the augmented data to construct the initial feature space.
Then we constrain the embedded features with a clustering loss to further learn clustering-oriented features. 
The clustering loss is composed of the target (pseudo label) and the actual output of the feature learning model, 
where the target is computed by using clean (non-augmented) data, 
and the output by augmented data. 
This is analogous to supervised training with data augmentation and expected to facilitate unsupervised clustering too.
Finally, we instantiate five DEC-DA based algorithms.
Extensive experiments validate that incorporating data augmentation can improve the clustering performance by a large margin. 
Our DEC-DA algorithms become the new state of the art on various datasets.

## Highlights
- The first work to introduce data augmentation into unsupervised deep embedded clustering problem.
- Proposed the framework of deep embedded clustering with data augmentation (DEC-DA) and instantiate five specific algorithms.
- The algorithms achieve the state-of-the-art clustering performance on four image datasets: MNIST, MNIST-TEST, USPS, Fashion-MNIST.


## Usage

### 1. Prepare environment

Install [Anaconda](https://www.anaconda.com/download/) with Python 3.6 version (_Optional_).   
Create a new env (_Optional_):   
```
conda create -n decda python=3.6 -y   
source activate decda  # Linux 
#  or 
conda activate decda  # Windows
```
Install required packages:
```
pip install scipy scikit-learn h5py tensorflow-gpu==1.10  
```
### 2. Clone the code and prepare the datasets.

```
git clone https://github.com/XifengGuo/DEC-DA.git DEC-DA
cd DEC-DA
```
Download **usps_train.jf** and **usps_test.jf** to "./data/usps" from 
[Google Drive](https://drive.google.com/open?id=1GfXb-YrRMe874bqCwX7QPjnx_D0y5-vX)
or
[Baidu Disk](https://pan.baidu.com/s/1rc3zxyjdeYYg-p_wvChrBA)   
**MNIST** and **Fashion-MNIST (FMNIST)** can be downloaded automatically when you run the code.

### 3. Run experiments.    

Quick test (_Optional_):
```bash
python run_exp.py --trials 1 --pretrain-epochs 1 --maxiter 150
```
Reproduce the results in Table 3 in the paper (this may take a long time):
```bash
python run_exp.py
```
Other flexible usages:

```
# 1. get uaage help
python main.py -h
# 2. train FcDEC without data augmentation on mnist dataset:
python main.py --method FcDEC --optimizer sgd --lr 0.01 --save-dir results/fcdec 
# 3. train FcDEC with pretrained autoencoder weights:
python main.py --method FcDEC --optimizer sgd --lr 0.01  --pretrained-weights results/fcdec/ae_weights.h5
# 4. test FcDEC using trained model weights:  
python main.py --method FcDEC -t --weights results/fcdec/model_final.h5
# 5. ......   
```
