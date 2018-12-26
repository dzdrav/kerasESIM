# Keras ESIM for Natural Language Inference (NLI)


This repo contains **Keras ESIM**, an implementation of Chen et. al. **[Enhanced LSTM for Natural Language Inference](https://arxiv.org/abs/1609.06038)** (originally written in Theano) for the task of natural language inference.

**Natural Language Inference** (also know as Recognizing Textual Entailment) is defined as:
>...the task of deciding, given two text fragments,
whether the meaning of one text is entailed (can be inferred) from another text
(Dagan and Glickman 2004).

---
# Requirements
Code is tested and works with:
- Python 3.6
- Keras 2.2.2
- TensorFlow 1.8.0
- CUDA 8.0
- CuDNN 7.0.5

Should work with TF 1.5

Also, code is tested and works in **[Google Colab](https://colab.research.google.com/
)**.

---
# Usage
## Preprocess data (only once)
Clone repo and `cd` into it.

Fetch datasets and GloVe embeddings (requires about 2.4GB of space):
```
preprocess.sh
```

## Train
```
python compile_model.py
```
Model weights are saved in `.check` file with a timestamp, you can reload them for later use.

## Hyperparameters
Hyperparameters are changed in ```compile_model.py``` by assigning key-value pairs in ```options``` dictionary. Select dataset to train on by assigning a value for ```dataset``` key as such:
- SNLI: ```options['dataset'] = 'snli'```
- MNLI: ```options['dataset'] = 'mnli'```
- MNLI + random 15% subset of SNLI: ```options['dataset'] = 'mnlisnli'```

---
# Features

## MNLI support
Beside training and testing on [SNLI](https://nlp.stanford.edu/projects/snli/) dataset (549K examples), the code supports training and evaluation on newly released [Multi-Genre NLI corpus](https://www.nyu.edu/projects/bowman/multinli/) (MNLI), which contains 433K annotated sentence pairs. Code supports
- single dataset training (SNLI or MNLI)
- same dataset evaluation
- cross-dataset evaluation (train on SNLI, test on MNLI and vice versa)
- evaluation by every MNLI category
- joint dataset training (MNLI + random subset of 15% of SNLI examples)
 - a subset of SNLI is used so that resulting set has approximately same number of examples from each category (about 75-82K from each MNLI category + 82K from SNLI)

## Bayesian optimization
Code supports Bayesian hyperparameter optimization using [Hyperopt](https://github.com/hyperopt/hyperopt) with Tree Parzen Estimator.

Start Bayesian optimization by running
```
python hyper.py
```

---

# Credits
Implementation was adapted from [this repository](https://github.com/dzdrav/SNLI-Keras). Mentioned implementation didn't have preprocessing for SNLI data. Also, model structure and hyperparameters were different from [original ESIM](https://arxiv.org/abs/1609.06038), which resulted in subpar performance (82.8% compared to 88.0% of original ESIM, tested on SNLI).

My Keras ESIM is compliant with original structure and hyperparameters. Layers and hyperparameters are adjusted to match those in [original code](https://github.com/lukecq1231/nli), helping this model achieve **85.5%** on SNLI test set.

# About
This repository contains code used in my Master's thesis, **_"Recognizing Textual Entailment using  Deep Learning Methods"_**. If you use this code in your work, please provide a link to this repository.

If you need assistance or pre-trained weights, contact me via **dinko.zdravac** at Google mail (**gmail**).
