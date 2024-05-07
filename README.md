# CS598 Deep Health for Healthcare Project - Team 88
This repository contains code to reproduce the paper 
*TransformEHR: transformer-based encoder-decoder generative model to enhance prediction of disease outcomes using electronic health records*
by Zuo, H., Xu, J., & Jiang, Q. (2023).[1]

Team Members

Anikesh Haran - anikesh2@illinois.edu <br>
Changhua Zhan - zhan36@illinois.edu <br>
Satvik Kulkarni - satvikk2@illinois.edu

This is the repository for CS 598: Deep Learning for Healthcare class. The reproduced paper is TransformEHR: transformer-based encoder-decoder generative model to enhance prediction of disease outcomes using electronic health records. 
The link to the paper is here: [https://pubmed.ncbi.nlm.nih.gov/38030638/].

## Notebooks

The Project submission can be found in the [DL4H_Team_88_Final.ipynb]() notebook.

## How to run

The code can be run locally, or in Google Colab. Complete instructions are contained inside of the notebook.

### Project Quickstart

Note - Pass sample_size [%] as argument to the python script

Let's say, we want to run the model on 20% of data. Argument will be 0.2

Command - python3 train_ehr.py 0.2


## References
1. Yang, Z., Mitra, A., Liu, W. et al. TransformEHR: transformer-based encoder-decoder generative model to enhance prediction of disease outcomes using electronic health records. Nat Commun 14, 7857 (2023). https://doi.org/10.1038/s41467-023-43715-z
2. Vaswani, A. et al. Attention is All you Need. in Advances in Neural Information Processing Systems 30 (eds. Guyon, I. et al.) 5998–6008 (Curran Associates, Inc., 2017).https://arxiv.org/abs/1706.03762
3. Rasmy, L., Xiang, Y., Xie, Z. et al. Med-BERT: pretrained contextualized embeddings on large-scale structured electronic health records for disease prediction. npj Digit. Med. 4, 86 (2021). https://doi.org/10.1038/s41746-021-00455-y
4. Li, Y., Rao, S., Solares, J.R.A. et al. BEHRT: Transformer for Electronic Health Records. Sci Rep 10, 7155 (2020). https://doi.org/10.1038/s41598-020-62922-y