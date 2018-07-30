# Convolutional Neural Networks for Sentence Classification
Pytorch re-implementation of [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882).

This is an unofficial implementation.There is [the implementation by the authors](https://github.com/yoonkim/CNN_sentence), which is implemented on Theano.

## Results
Dataset: [SNLI](https://nlp.stanford.edu/projects/snli/)

Baseline from the paper

| Model | MR | SST-1 | SST-2 | Subj | TREC |
| ----- | -- | ----- | ----- | ---- | ---- |
| random | 76.1 | 45.0 | 82.7 | 89.6 | 91.2 |
| static | 81.0 | 45.5 | 86.8 | 93.0 | 92.8 |
| non-static | 81.5 | 48.0 | 87.2 | 93.4 | 93.6 |
| multi-channel | 81.1 | 47.4 | 88.1 | 93.2 | 92.2 |

Re-implementation

| Model | MR | SST-1 | SST-2 | Subj | TREC |
| ----- | -- | ----- | ----- | ---- | ---- |
| random | - | 36.1 | 74.5 | - | 87.2 |
| static | - | 47.8 | 85.2 | - | 92.8 |
| non-static | 81.0 | 47.8 | 85.3 | 92.8 | 93.6 |
| multi-channel | 81.0 | 48.1 | 85.2 | 92.2 | 93.8 |



## Development Environment
- OS: Ubuntu 16.04 LTS (64bit)
- Language: Python 3.6.6
- Pytorch: 0.4.0

## Requirements
Please install the following library requirements first.

    nltk==3.3
    tensorboardX==1.2
    torch==0.4.0
    torchtext==0.2.3
    
## Training
> python train.py --help

    usage: train.py [-h] [--batch-size BATCH_SIZE] [--dataset DATASET]
                    [--dropout DROPOUT] [--epoch EPOCH] [--gpu GPU]
                    [--learning-rate LEARNING_RATE] [--word-dim WORD_DIM]
                    [--norm-limit NORM_LIMIT] [--mode MODE]
                    [--num-feature-maps NUM_FEATURE_MAPS]

    optional arguments:
      -h, --help            show this help message and exit
      --batch-size BATCH_SIZE
      --dataset DATASET     available datasets: MR, TREC, SST-1, SST-2, SUBJ
      --dropout DROPOUT
      --epoch EPOCH
      --gpu GPU
      --learning-rate LEARNING_RATE
      --word-dim WORD_DIM
      --norm-limit NORM_LIMIT
      --mode MODE           available models: rand, static, non-static,
                            multichannel
      --num-feature-maps NUM_FEATURE_MAPS

 
 **Note:** 
