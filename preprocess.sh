#!/bin/sh

wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
rm glove.840B.300d.zip

wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip
unzip snli_1.0.zip
rm snli_1.0.zip

wget https://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip
unzip multinli_1.0.zip
rm multinli_1.0.zip

python preprocess_SNLI.py
python preprocess_MNLI.py
