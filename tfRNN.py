import json
import os
import re
import time
import random
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from RITutils import f1_score, recall, precision, w_categorical_crossentropy

import keras
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.layers import merge, recurrent, Dense, Input, Dropout, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Lambda
from keras.layers.wrappers import Bidirectional
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l2
from keras.utils import np_utils
from keras.layers.recurrent import GRU,LSTM
from keras.layers import CuDNNGRU, CuDNNLSTM, Activation # CuDNN podržana implementacija LSTM i GRU-a
from keras.backend.tensorflow_backend import set_session
from keras.engine.topology import Layer
from keras.utils.vis_utils import plot_model # vizualizacija modela
from json_tricks import dump

import sys


def time_count(fn):
  # Funtion wrapper used to memsure time consumption
  def _wrapper(*args, **kwargs):
    start = time.clock()
    returns = fn(*args, **kwargs)
    print("[time_count]: %s cost %fs" % (fn.__name__, time.clock() - start))
    return returns
  return _wrapper


class AttentionAlignmentModel:
# accepts options dict with hyperparameters
  def __init__(self, options, annotation ='biGRU', dataset = 'snli'):
    # 1, Set Basic Model Parameters
    self.Layers = 1
    self.EmbeddingSize = 300 # size of projected embeddings
    self.BatchSize = options['BatchSize'] if 'BatchSize' in options else 128
    self.Patience = 7 # original Chen et.al.
    self.MaxEpoch = 25
    self.SentMaxLen = 42 if dataset == 'snli' else 50
    self.DropProb = 0.5 # original Chen et.al.
    self.L2Strength = options['L2Strength'] if 'L2Strength' in options else 0.0
    self.Activate = 'relu'
    self.GradientClipping = options['GradientClipping'] if 'GradientClipping' in options else 10.0
    # self.Optimizer = 'rmsprop' # originalna vrijednost
    self.LearningRate = options['LearningRate'] if 'LearningRate' in options else 4e-4
    if 'Optimizer' not in options or options['Optimizer'] == 'nadam':
        self.Optimizer = keras.optimizers.Nadam(lr = self.LearningRate,
            clipnorm = self.GradientClipping)
    elif options['Optimizer'] == 'adam': # u radu naveden Adam, orig. rmsprop
        self.Optimizer = keras.optimizers.Adam(lr = self.LearningRate,
            clipnorm = self.GradientClipping)
    elif options['Optimizer'] == 'rmsprop':
        self.Optimizer = keras.optimizers.RMSprop(lr = self.LearningRate,
            clipnorm = self.GradientClipping)
    self.rnn_type = annotation
    self.dataset = dataset

    # whether to change tokens to lowercase before training
    self.LowercaseTokens = options['LowercaseTokens'] if 'LowercaseTokens' in options else True
        # changing this value requires setting RetrainEmbeddings to True
    self.RetrainEmbeddings = options['RetrainEmbeddings'] if 'RetrainEmbeddings' in options else True
    self.LoadExistingWeights = False # True: loading existing model weights
    self.TrainableEmbeddings = True # True: update word embeddings during training
    # True: last dropout layer has 1/2 of dropout factor
    self.LastDropoutHalf = options['LastDropoutHalf'] if 'LastDropoutHalf' in options else False
    self.OOVWordInit = options['OOVWordInit'] if 'OOVWordInit' in options else 'zeros'

    # 2, Define Class Variable
    self.Options = options
    self.Options['Timestamp'] = time.strftime('%Y%m%d%H%M', time.localtime()) if 'ConfigTimestamp' not in options else options['ConfigTimestamp']
    self.Timestamp = self.Options['Timestamp']
    self.ResultFilepath = 'models/' + self.Timestamp + '_model/'
    self.History = None
    # self.Verbose = 0
    self.Vocab = 0
    self.model = None
    self.GloVe = defaultdict(np.array)
    self.glove_path = self.ResultFilepath + self.Timestamp + '_GloVe_' + self.dataset + '.npy'
    self.indexer,self.Embed = None, None
    self.train, self.validation, self.test = [],[],[]
    self.Labels = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
    self.rLabels = {0:'contradiction', 1:'neutral', 2:'entailment'}

  # writes a report containing hyperparameters and learning details
  def format_report(self):
      outFile = self.ResultFilepath + self.Timestamp + '_ESIM_' + self.dataset.upper() + '_report.json'
      with open(outFile, 'w', encoding='utf-8') as outFile:
          dump(self.Options, outFile, indent=2)

  # helper function for data preprocessing - pads sentences to same length
  def padd(self, x):
    def padding(x, MaxLen):
      return pad_sequences(sequences=self.indexer.texts_to_sequences(x), maxlen=MaxLen)
    def pad_data(x):
      return padding(x[0], self.SentMaxLen), padding(x[1], self.SentMaxLen), x[2]
    return pad_data(x)

  # loads data depending on declared dataset
  def load_data(self):
    self.model_mkdir()
    if self.dataset == 'snli':
      trn = json.loads(open('snli_train.json', 'r').read())
      vld = json.loads(open('snli_validation.json', 'r').read())
      tst = json.loads(open('snli_test.json', 'r').read())
    elif self.dataset == 'rte':
      trn = json.loads(open('RTE_train.json', 'r').read())
      vld = json.loads(open('RTE_valid.json', 'r').read())
      tst = json.loads(open('RTE_test.json', 'r').read())
    elif self.dataset == 'mnli': # validating with matched validation set
      trn = json.loads(open('mnli_train.json', 'r').read())
      vld = json.loads(open('mnli_validation_matched.json', 'r').read())
      tst = json.loads(open('mnli_validation_mismatched.json', 'r').read())
    elif self.dataset == 'mnlisnli':
      trn = json.loads(open('mnli_train.json', 'r').read())
      trn2 = json.loads(open('snli_train.json', 'r').read())
      vld = json.loads(open('mnli_validation_matched.json', 'r').read())
      vld2 = json.loads(open('snli_validation.json', 'r').read())
      # sljedeću liniju NIKAKO ne brisati, o tome ovisi evaluate_on_set()
      tst = json.loads(open('mnli_validation_mismatched.json', 'r').read())

      """
      Since joint training randomly picks 15% of SNLI examples, to evaluate such
      trained model, you need to provide timestamp of JSON file containing
      which random examples were picked for training so that same embeddings
      could be loaded into the model for evaluation. Why? Because embeddings are
      trainable which makes them model parameters, which requires same
      embeddings to be loaded when evaluating model as when that same model was
      trained. Provide timestamp of trained model in options['ConfigTimestamp'].
      If not provided, random examples are picked and saved to JSON for later
      reproducibility.
      """
      # validation set
      subset = [[],[],[]]
      if 'ConfigTimestamp' in self.Options: # if timestamp exists, load examples
        indices = json.load(open(self.ResultFilepath + self.Options['ConfigTimestamp'] + '_validconfig.json', 'r'))
      else: # else pick randomly
        indices = random.sample(range(len(vld2[0])), 2000)
        json.dump(indices, open(self.ResultFilepath + self.Timestamp + '_validconfig.json', 'w'))
      # merging with MNLI
      for index in indices:
        for i in range(3):
          subset[i].append(vld2[i][index])
      for i in range(3):
        vld[i].extend(subset[i])
      # train set
      subset = [[],[],[]]
      if 'ConfigTimestamp' in self.Options:
        indices = json.load(open(self.ResultFilepath + self.Options['ConfigTimestamp'] + '_trainconfig.json', 'r'))
      else:
        indices = random.sample(range(len(trn2[0])), int(0.15 * len(trn2[0]) ) )
        json.dump(indices, open(self.ResultFilepath + self.Timestamp + '_trainconfig.json', 'w'))
      for index in indices:
        for i in range(3):
          subset[i].append(trn2[i][index])
      for i in range(3):
        trn[i].extend(subset[i])
    else:
      raise ValueError('Unknown Dataset')

    trn[2] = np_utils.to_categorical(trn[2], 3)
    vld[2] = np_utils.to_categorical(vld[2], 3)
    tst[2] = np_utils.to_categorical(tst[2], 3)

    return trn, vld, tst

  def model_mkdir(self):
    if not os.path.exists('models/'):
      os.mkdir('models/')
    if not os.path.exists(self.ResultFilepath):
      os.mkdir(self.ResultFilepath)

  @time_count
  def prep_data(self):
    # 1, Read raw Training,Validation and Test data
    self.train,self.validation,self.test = self.load_data()
    # 2, Prep Word Indexer: assign each word a number
    self.indexer = Tokenizer(lower = self.LowercaseTokens, filters = '') # nova linija
    # indexer fitamo nad training podacima
    self.indexer.fit_on_texts(self.train[0] + self.train[1])
    # self.Vocab je veličina vokabulara
    self.Vocab = len(self.indexer.word_counts) + 1
    print('Vocabulary size:', self.Vocab)

    # 3, Convert each word in set to num and zero pad
    self.train = self.padd(self.train)
    self.validation = self.padd(self.validation)
    self.test = self.padd(self.test)

  def load_GloVe(self):
    # Create an embedding matrix for word2vec (use GloVe)
    # embedding matrix contains word embeddings for each word
    embed_index = {}
    for line in open('glove.840B.300d.txt','r'):
        value = line.split(' ') # Warning: Can't use split()! I don't know why...
        word = value[0]
        embed_index[word] = np.asarray(value[1:],dtype='float32')
    # embed matrix is of size 300*(no. of vocabulary words)
    # hence it CANNOT be reloaded when changing dataset
    if self.dataset == 'mnlisnli' and os.path.exists(self.glove_path):
        embed_matrix = np.load(self.glove_path)
    elif self.OOVWordInit == 'random':
        embed_matrix = np.random.randn(self.Vocab,self.EmbeddingSize)
    elif self.OOVWordInit == 'zeros':
        embed_matrix = np.zeros((self.Vocab,self.EmbeddingSize))

    unregistered = []
    for word,i in self.indexer.word_index.items():
        vec = embed_index.get(word)
        # if word with index 'vec' not in word_index, add it to out-of-voc list
        if vec is None: unregistered.append(word)
        # else save it in embedding matrix on its position
        else: embed_matrix[i] = vec
    np.save(self.glove_path, embed_matrix)
    open('unregisterd_word.txt','w').write(str(unregistered))

  def load_GloVe_dict(self):
    for line in open('glove.840B.300d.txt','r'):
      value = line.split(' ') # Warning: Can't use split()! I don't know why...
      word = value[0]
      self.GloVe[word] = np.asarray(value[1:],dtype='float32')

  @time_count
  def prep_embd(self):
    # Add Embed Layer to convert word index to vector
    if self.dataset != 'mnlisnli':
      self.glove_path = self.ResultFilepath + 'GloVe_' + self.dataset + '.npy'
    if 'ConfigTimestamp' in self.Options:
      self.glove_path = self.ResultFilepath + self.Options['ConfigTimestamp'] + '_GloVe_' + self.dataset + '.npy'

    # with joint training, we always delete previous embedding matrix
    # if self.dataset == 'mnlisnli' and os.path.exists(glove_path) and 'ConfigTimestamp' not in self.Options:
        # os.remove(glove_path)
    if not os.path.exists(self.glove_path) or self.RetrainEmbeddings:
      self.load_GloVe()
    # loading freshly made embedding matrix
    embed_matrix = np.load(self.glove_path)
    self.Embed = Embedding(input_dim = self.Vocab,
                           output_dim = self.EmbeddingSize,
                           input_length = self.SentMaxLen,
                           trainable = self.TrainableEmbeddings,
                           weights = [embed_matrix],
                           name = 'embed_' + self.dataset.upper())


  # Enhanced LSTM Attention model by Qian Chen et al. 2016
  def create_enhanced_attention_model(self):
    # 0, (Optional) Set the upper limit of GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    # 1, Embedding the input and project the embeddings
    premise = Input(shape=(self.SentMaxLen,), dtype='int32')
    hypothesis = Input(shape=(self.SentMaxLen,), dtype='int32')
    embed_p = self.Embed(premise)  # [batchsize, Psize, Embedsize]
    embed_h = self.Embed(hypothesis)  # [batchsize, Hsize, Embedsize]

    # 2, Encoder words with its surrounding context
    # initialization of LSTM input matrix with random Gauss distr
    Encoder = Bidirectional(CuDNNLSTM(units=300, return_sequences=True, kernel_initializer='RandomNormal')) # nova linija - CuDNNLSTM

    embed_p = Dropout(self.DropProb)(embed_p) # firstly dropout
    embed_h = Dropout(self.DropProb)(embed_h) # firstly dropout
    embed_p = Encoder(embed_p) # then BiLSTM encoding
    embed_h = Encoder(embed_h) # then BiLSTM encoding

    # 2, Score each words and calc score matrix Eph.
    F_p, F_h = embed_p, embed_h
    Eph = keras.layers.Dot(axes=(2, 2))([F_h, F_p])  # [batch_size, Hsize, Psize]
    Eh = Lambda(lambda x: keras.activations.softmax(x))(Eph)  # [batch_size, Hsize, Psize]
    Ep = keras.layers.Permute((2, 1))(Eph)  # [batch_size, Psize, Hsize)
    Ep = Lambda(lambda x: keras.activations.softmax(x))(Ep)  # [batch_size, Psize, Hsize]

    # 4, Normalize score matrix, encoder premesis and get alignment
    PremAlign = keras.layers.Dot((2, 1))([Ep, embed_h]) # [-1, Psize, dim]
    HypoAlign = keras.layers.Dot((2, 1))([Eh, embed_p]) # [-1, Hsize, dim]
    mm_1 = keras.layers.Multiply()([embed_p, PremAlign])
    mm_2 = keras.layers.Multiply()([embed_h, HypoAlign])
    sb_1 = keras.layers.Subtract()([embed_p, PremAlign])
    sb_2 = keras.layers.Subtract()([embed_h, HypoAlign])

    # concat [a_, a~, a_ * a~, a_ - a~], isto za b_, b~
    PremAlign = keras.layers.Concatenate()([embed_p, PremAlign, sb_1, mm_1,])  # [batch_size, Psize, 2*unit]
    HypoAlign = keras.layers.Concatenate()([embed_h, HypoAlign, sb_2, mm_2])  # [batch_size, Hsize, 2*unit]
    # ff layer w/RELU activation
    Compresser = TimeDistributed(Dense(300,
                                       kernel_regularizer=l2(self.L2Strength),
                                       bias_regularizer=l2(self.L2Strength),
                                       activation='relu'),
                                 name='Compresser')
    PremAlign = Compresser(PremAlign)
    HypoAlign = Compresser(HypoAlign)

    # 5, Final biLST < Encoder + Softmax Classifier
    Decoder = Bidirectional(CuDNNLSTM(units=300, return_sequences=True, kernel_initializer='RandomNormal'),
                            name='finaldecoder')  # [-1,2*units]
    PremAlign = Dropout(self.DropProb)(PremAlign)
    HypoAlign = Dropout(self.DropProb)(HypoAlign)
    final_p = Decoder(PremAlign)
    final_h = Decoder(HypoAlign)

    AveragePooling = Lambda(lambda x: K.mean(x, axis=1)) # outs [-1, dim]
    MaxPooling = Lambda(lambda x: K.max(x, axis=1)) # outs [-1, dim]
    avg_p = AveragePooling(final_p)
    avg_h = AveragePooling(final_h)
    max_p = MaxPooling(final_p)
    max_h = MaxPooling(final_h)
    # concat of avg and max pooling for hypothesis and premise
    Final = keras.layers.Concatenate()([avg_p, max_p, avg_h, max_h])
    # dropout layer
    Final = Dropout(self.DropProb)(Final)
    # ff layer w/tanh activation
    Final = Dense(300,
                  kernel_regularizer=l2(self.L2Strength),
                  bias_regularizer=l2(self.L2Strength),
                  name='dense300_' + self.dataset,
                  activation='tanh')(Final)

    # last dropout factor
    factor = 1
    if self.LastDropoutHalf:
        factor = 2
    Final = Dropout(self.DropProb / factor)(Final)

    # softmax classifier
    Final = Dense(2 if self.dataset == 'rte' else 3,
                  activation='softmax',
                  name='judge300_' + self.dataset)(Final)
    self.model = Model(inputs=[premise, hypothesis], outputs=Final)


  @time_count
  def compile_model(self):
    """ Load Possible Existing Weights and Compile the Model """
    self.model.compile(optimizer=self.Optimizer,
                       loss=w_categorical_crossentropy if self.dataset == 'rte'
                       else 'categorical_crossentropy',
                       metrics=['accuracy' , precision, recall, f1_score]
                       if self.dataset == 'rte' else ['accuracy'])
    self.model.summary()
    fn = self.rnn_type + '_' + self.dataset + '.check'
    if os.path.exists(fn) and self.LoadExistingWeights:
        self.model.load_weights(fn, by_name=True)
        print('--------Load Weights Successful!--------')

  # returns history of train/val loss/acc values
  def start_train(self):
    """ Starts to Train the entire Model Based on set Parameters """
    # 1, Prep
    callback = [EarlyStopping(patience=self.Patience, verbose=2),
                ReduceLROnPlateau(patience=5, verbose=1),
                CSVLogger(filename=self.rnn_type+'log.csv'),
                ModelCheckpoint(filepath = self.ResultFilepath + self.Timestamp
                                            + '_' + self.dataset
                                            + 'weights.{epoch:02d}-{val_loss:.2f}.check',
                                save_best_only=False,
                                save_weights_only=True)]
    # 2, Train
    self.History = self.model.fit(x = [self.train[0],self.train[1]],
                   y = self.train[2],
                   batch_size = self.BatchSize,
                   epochs = self.MaxEpoch,
                   validation_data=([self.validation[0], self.validation[1]], self.validation[2]),
                   callbacks = callback)
    self.Options['History'] = self.History.history
    self.format_report()
    return self.History

  # eval_set: forward exact filename upon which to test (without file extension)
  def evaluate_on_set(self, eval_set = 'snli_test', weights_file = None):
    # checks validity of eval_set name
    assert eval_set in ['snli_validation', 'snli_test',
                         'mnli_validation_matched',
                         'mnli_validation_mismatched']
    dataset = None

    if weights_file is not None: # loads weights from given file
        self.model.load_weights(self.ResultFilepath + weights_file)
    # or tries to load default named weights
    elif os.path.exists(self.rnn_type + '_' + self.dataset + '.check'):
        self.model.load_weights(self.rnn_type + '_' + self.dataset + '.check') # revert to the best model
    else: # or initializes weights to random
        print('No weights found for model!')
        print('Using random initialized weights...')
    # if testing on same set, load it from member variables...
    if (self.dataset == 'snli' and 'snli' in eval_set) or (self.dataset == 'mnli' and 'mnli' in eval_set):
        if eval_set == 'snli_validation' or eval_set == 'mnli_validation_matched':
            dataset = self.validation
        elif eval_set == 'snli_test' or eval_set == 'mnli_validation_mismatched':
            dataset = self.test
    # if cross-testing, load dataset from file...
    else:
        print('Loading ' + eval_set + ' data...')
        dataset = json.loads(open(eval_set + '.json', 'r').read())
        dataset[2] = np_utils.to_categorical(dataset[2], 3)
        dataset = self.padd(dataset)
    # evaluation
    loss, acc = self.model.evaluate([dataset[0], dataset[1]],
                                    dataset[2], batch_size=self.BatchSize)

    print('Trained on: ' + self.dataset.upper())
    print('Evaluated on: ' + eval_set.title() + ": loss = {:.5f}, acc = {:.4f}%".format(loss, acc))
    return (loss, acc)

  def evaluate_on_test_sets(self, weights_file = None):
    results = {}
    score = {}
    # evaluates on all test sets
    for set in ['snli_test', 'mnli_validation_matched','mnli_validation_mismatched']:
      results[set] = {}
      results[set]['loss'], results[set]['acc'] = self.evaluate_on_set(eval_set = set, weights_file = weights_file)
    # dumping results to a file
    dump(results, open(self.ResultFilepath + self.Timestamp + '_test_results.json', 'w'), indent = 2)

  # evaluates model on ALL MNLI categories: matched & mismatched
  def evaluate_on_all_mnli_categories(self, weights_file = None):
    if weights_file is not None: # loads weights from given file
        self.model.load_weights(self.ResultFilepath + weights_file)
    sets = ['mnli_validation_matched',
            'mnli_validation_mismatched']
    results = {}
    for eval_set in sets:
        print('Loading ' + eval_set + ' data...')
        dataset = json.loads(open(eval_set + '.json', 'r').read())
        all_categories = list(set( dataset[3] ))

        for category in all_categories:
            subdataset = [[],[],[]]
            print('Category: ' + category)
            for i in range(len(dataset[0])):
                if dataset[3][i] == category:
                    for j in range(3):
                        subdataset[j].append(dataset[j][i])
            subdataset[2] = np_utils.to_categorical(subdataset[2], 3)
            subdataset = self.padd(subdataset)
            result = {}
            result['loss'], result['acc'] = self.model.evaluate([subdataset[0], subdataset[1]],
                                            subdataset[2],
                                            batch_size=self.BatchSize)
            results[category] = result
    dump(results, open(self.ResultFilepath + self.Timestamp + '_categoric_results.json', 'w'), indent = 2)


  @staticmethod
  def plotHeatMap(df, psize=(8,8), filename='Heatmap'):
    ax = sns.heatmap(df, vmax=.85, square=True, cbar=False, annot=True)
    plt.xticks(rotation=40), plt.yticks(rotation=360)
    fig = ax.get_figure()
    fig.set_size_inches(psize)
    fig.savefig(filename)
    plt.clf()

  def interactive_predict(self, test_mode = False):
    """[ONLY WORK FOR SNLI] The model must be compiled before execuation """
    prep_alfa = lambda X: pad_sequences(sequences=self.indexer.texts_to_sequences(X),
                                        maxlen=self.SentMaxLen)
    while True:
      prem = input("Please input the premise:\n")
      hypo = input("Please input another sent:\n")
      unknown = set([word for word in list(filter(lambda x: x and x != ' ',
                                                  re.split(r'(\W)',prem) + re.split(r'(\W)',hypo)))
                          if word not in self.indexer.word_counts.keys()])
      if unknown:
        print('[WARNING] {}s Unregistered Words:{}'.format(len(unknown),unknown))
      prem_pad, hypo_pad = prep_alfa([prem]), prep_alfa([hypo])
      if test_mode:
        ans = self.model.predict(x=[prem_pad, hypo_pad], batch_size=1)
        Ep, Eh = np.array(ans[0]).reshape(36,36), np.array(ans[1]).reshape(36,36) # [P,H] [H,P]
        Ep = Ep[-len(prem.split(' ')):,-len(hypo.split(' ')):] # [P,H]
        Eh = Eh[-len(hypo.split(' ')):,-len(prem.split(' ')):] # [H,P]
        self.plotHeatMap(pd.DataFrame(Ep,columns=hypo.split(' '),index=prem.split(' ')),
                         psize=(7, 10), filename='Ep')
        self.plotHeatMap(pd.DataFrame(Eh,columns=prem.split(' '),index=hypo.split(' ')),
                         psize=(10,7), filename='Eh')
        ans = np.reshape(ans[2], -1)
      else:
        ans = np.reshape(self.model.predict(x=[prem_pad, hypo_pad],batch_size=1),-1) # PREDICTION
      print('\n Contradiction     {:.1f}%\n'.format(float(ans[0]) * 100),
            'Neutral         {:.1f}%\n'.format(float(ans[1]) * 100),
            'Entailment     {:.1f}%\n'.format(float(ans[2]) * 100))

  def label_test_file(self):
    outfile = open("pred_vld.txt","w")
    prep_alfa = lambda X: pad_sequences(sequences=self.indexer.texts_to_sequences(X),
                                        maxlen=self.SentMaxLen)
    vld = json.loads(open('validation.json', 'r').read())
    for prem, hypo, label in zip(vld[0], vld[1], vld[2]):
      prem_pad, hypo_pad = prep_alfa([prem]), prep_alfa([hypo])
      ans = np.reshape(self.model.predict(x=[prem_pad, hypo_pad], batch_size = 1), -1)  # PREDICTION
      if np.argmax(ans) != label:
        outfile.write(prem + "\n" + hypo + "\n")
        outfile.write("Truth: " + self.rLabels[label] + "\n")
        outfile.write('Contradiction     {:.1f}%\n'.format(float(ans[0]) * 100) +
                      'Neutral         {:.1f}%\n'.format(float(ans[1]) * 100) +
                      'Entailment     {:.1f}%\n'.format(float(ans[2]) * 100))
        outfile.write("-"*15 + "\n")
    outfile.close()
