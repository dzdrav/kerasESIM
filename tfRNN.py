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

class ResampleSnli(keras.callbacks.Callback):
  def __init__(self, model):
    self.ParentModel = model
  def on_epoch_end(self, batch, logs):
    self.ParentModel.merge_snli_into_mnli()
  # def on_train_begin(self, logs):
    # self.ParentModel.merge_snli_into_mnli()

class AttentionAlignmentModel:
# prima options dictionary koji inicijalizira hiperparametre
  # def __init__(self, annotation ='biGRU', dataset = 'snli'):
  def __init__(self, options, annotation ='biGRU', dataset = 'snli'):
    # 1, Set Basic Model Parameters
    self.Layers = 1
    self.EmbeddingSize = 300
    self.BatchSize = options['BatchSize'] if 'BatchSize' in options else 128
    # patience za early stopping
    # self.Patience = 8 # originalna vrijednost
    self.Patience = 7 # u izvornom kodu Chen et.al.
    # self.Patience = 6 # vlastiti izbor
    self.MaxEpoch = 25
    self.SentMaxLen = 42 if dataset == 'snli' else 50 # snli: 42, mnli: 50 default
    # self.SentMaxLen = 100 # uočeno u izvornom kodu Chen et.al.
    # self.DropProb = 0.4 # originalna vrijednost
    self.DropProb = 0.5 # navedeno u radu Chen et.al.
    # self.L2Strength = 1e-5 # originalna linija
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
    
    # modifikacije treniranja modela
    # pretvaramo li tokene u lowercase prije obrade
    self.LowercaseTokens = options['LowercaseTokens'] if 'LowercaseTokens' in options else True
        # promjena LowercaseTokens zahtijeva postavljanje RetrainEmbeddings na True 
    # RetrainEmbeddings=True: ponovno treniramo word embeddinge prije treninga modela
    self.RetrainEmbeddings = options['RetrainEmbeddings'] if 'RetrainEmbeddings' in options else False
    self.LoadExistingWeights = False # True: učitavamo postojeće weighte za model
    self.TrainableEmbeddings = True # True: model ažurira word embeddinge tijekom treninga
    # True: zadnji dropout layer ima 1/2 faktor dropouta
    self.LastDropoutHalf = options['LastDropoutHalf'] if 'LastDropoutHalf' in options else False
    self.OOVWordInit = options['OOVWordInit'] if 'OOVWordInit' in options else 'zeros'

    # 2, Define Class Variable
    self.Options = options
    self.Options['Timestamp'] = time.strftime('%Y%m%d%H%M', time.localtime())
    self.Timestamp = self.Options['Timestamp']
    self.History = None
    # self.Verbose = 0
    self.Vocab = 0
    self.model = None
    self.GloVe = defaultdict(np.array)
    self.glove_path = self.Timestamp + '_GloVe_' + self.dataset + '.npy'
    self.indexer,self.Embed = None, None
    self.train, self.validation, self.test = [],[],[]
    # služe za istovremeno treniranje SNLI+MNLI
    self.mnli_train, self.snli_train = [], []
    self.Labels = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
    self.rLabels = {0:'contradiction', 1:'neutral', 2:'entailment'}

  # zapisuje report s hiperparametrima i statistikom učenja
  def format_report(self):
      outFile = self.Timestamp + '_ESIM_' + self.dataset.upper() + '_report.json'
      with open(outFile, 'w', encoding='utf-8') as outFile:
          dump(self.Options, outFile, indent=2)

  def padd(self, x):
    def padding(x, MaxLen):
      return pad_sequences(sequences=self.indexer.texts_to_sequences(x), maxlen=MaxLen)
    def pad_data(x):
      return padding(x[0], self.SentMaxLen), padding(x[1], self.SentMaxLen), x[2]
    return pad_data(x)
  
  def load_data(self):
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
      ako želimo evaluirati model s postojećim weightima, treba učitati iste 
      train i val primjere. Treba dati Timestamp u options['ConfigTimestamp']
      json datoteke gdje se te konfiguracije nalaze.
      Ako options['ConfigTimestamp'] ne postoji, biraju se random primjeri
      i spremaju u json datoteku za kasnije korištenje
      """
      subset = [[],[],[]]
      if 'ConfigTimestamp' in self.Options: # ako postoji timestamp u Options
        indices = json.load(open(self.Options['ConfigTimestamp'] + '_validconfig.json', 'r'))
      else: # inače generiraj nasumično
        indices = random.sample(range(len(vld2[0])), 2000)
        json.dump(indices, open(self.Timestamp + '_validconfig.json', 'w'))
      # spajanje s MNLI val skupom
      for index in indices:
        for i in range(3):
          subset[i].append(vld2[i][index])
      for i in range(3):
        vld[i].extend(subset[i])

      subset = [[],[],[]]
      if 'ConfigTimestamp' in self.Options: # ako postoji timestamp u Options
        indices = json.load(open(self.Options['ConfigTimestamp'] + '_trainconfig.json', 'r'))
      else: # inače generiraj nasumično
        indices = random.sample(range(len(trn2[0])), int(0.15 * len(trn2[0]) ) )
        json.dump(indices, open(self.Timestamp + '_trainconfig.json', 'w'))
      # spajanje s MNLI train skupom
      for index in indices:
        for i in range(3):
          subset[i].append(trn2[i][index])
      for i in range(3):
        trn[i].extend(subset[i])
    else:
      raise ValueError('Unknown Dataset') # ispravljen typo, nova linija
    
    # ako želimo RTE dataset, drugi argument: (3 if self.dataset == 'snli' else 2)
    trn[2] = np_utils.to_categorical(trn[2], 3)
    vld[2] = np_utils.to_categorical(vld[2], 3)
    tst[2] = np_utils.to_categorical(tst[2], 3)
    
    return trn, vld, tst

  # učitava oba dataseta iz datoteke u članske varijable (NE VRAĆA ništa)
  def load_smnli_data(self):
    snli = json.loads(open('snli_train.json', 'r').read())
    mnli = json.loads(open('mnli_train.json', 'r').read())
        
    # snli[2] = np_utils.to_categorical(snli[2], 3)
    # mnli[2] = np_utils.to_categorical(mnli[2], 3)
    
    self.snli_train, self.mnli_train = snli, mnli

  # pretvara train skup u MNLI + 0.15*SNLI
  def prep_snli_train_data(self):
    # 0, praznimo trening skup
    del self.train
    self.train = [[],[],[]]
    subset = [[],[],[]]

    # 1, uzimamo random sample 15% indeksa SNLI-a
    indices = random.sample(range(len(self.snli_train[0])), int(0.15 * len(self.snli_train[0]) ) )
    # prolazimo kroz izabrane indekse i spremamo parove u novu listu
    for index in indices:
        for i in range(3):
            subset[i].append(self.snli_train[i][index])
    
    # 2, spremamo nove rečenice u trening skup
    for i in range(3):
        self.train[i].extend(subset[i])
        self.train[i].extend(self.mnli_train[i])
      
    # 3, treniramo indexer nad novim riječima
    # self.indexer.fit_on_texts(subset[0] + subset[1])
    del self.indexer
    self.indexer = Tokenizer(lower = self.LowercaseTokens, filters = '') # nova linija
    self.indexer.fit_on_texts(self.train[0] + self.train[1])
    self.Vocab = len(self.indexer.word_counts) + 1
    # želimo da se vel. vokabulara ne mijenja između epoha (pri retreniranju self.train)
    # self.Vocab = 180000 #TODO privremeno rješenje bazirano na n=175528, 175644, smisliti bolje
    
    self.train[2] = np_utils.to_categorical(self.train[2], 3)
    self.train = self.padd(self.train)

  # priprema train set + indexer za novu epohu
  # @time_count
  # def merge_snli_into_mnli(self):
    # self.prep_snli_train_data()
    # self.prep_embd()

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
    # if self.dataset == 'mnlisnli':
      # self.load_smnli_data()
    
  def load_GloVe(self):
    # Creat a embedding matrix for word2vec(use GloVe)
    # embed matrica sadrži embeddinge za riječi
    embed_index = {}
    for line in open('glove.840B.300d.txt','r'):
        value = line.split(' ') # Warning: Can't use split()! I don't know why...
        word = value[0]
        embed_index[word] = np.asarray(value[1:],dtype='float32')
        # ovdje se embed matrica inicjializira na nule (originalno)
    # embed_matrix = np.zeros((self.Vocab,self.EmbeddingSize)) # originalna linija
        # trebala bi se inicijalizirati na normalnu nasumičnu distribuciju
    # NOVI KOD -- POCETAK
    # NAPOMENA TODO: embed matrica je dimenzija 300x(broj riječi u vokabularu) - ne može se reloadati kada se promijeni dataset
    if self.dataset == 'mnlisnli' and os.path.exists(self.glove_path):
        embed_matrix = np.load(self.glove_path)
    elif self.OOVWordInit == 'random':
        embed_matrix = np.random.randn(self.Vocab,self.EmbeddingSize)
    elif self.OOVWordInit == 'zeros':
        embed_matrix = np.zeros((self.Vocab,self.EmbeddingSize))
    # NOVI KOD -- KRAJ
    
    unregistered = []
    for word,i in self.indexer.word_index.items():
        vec = embed_index.get(word)
        # ako riječ vec indeksa [word] nije u word_indexu, dodaj ju na popis OOV riječi
        if vec is None: unregistered.append(word)
        # inače ju spremi u embed matricu na njenu poziciju
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
    # Add a Embed Layer to convert word index to vector
    if self.dataset != 'mnlisnli':
      self.glove_path = 'GloVe_' + self.dataset + '.npy'
    if 'ConfigTimestamp' in self.Options:
      self.glove_path = self.Options['ConfigTimestamp'] + '_GloVe_' + self.dataset + '.npy'
    
    # kod joint treniranja uvijek brišemo prethodnu embed matricu ako postoji i 
    # ako ju ne pokušavamo učitati radi evaluacije
    # if self.dataset == 'mnlisnli' and os.path.exists(glove_path) and 'ConfigTimestamp' not in self.Options:
        # os.remove(glove_path)
    if not os.path.exists(self.glove_path) or self.RetrainEmbeddings:
      self.load_GloVe()
    # učitavamo upravo napravljenu embed matricu
    embed_matrix = np.load(self.glove_path)
    self.Embed = Embedding(input_dim = self.Vocab,
                           output_dim = self.EmbeddingSize,
                           input_length = self.SentMaxLen,
                           # originalna linija: False, određuje trenirabilnost word embeddinga
                           trainable = self.TrainableEmbeddings,
                           weights = [embed_matrix],
                           name = 'embed_' + self.dataset.upper())

  # TODO Decomposable Attention Model by Ankur P. Parikh et al. 2016
  def create_standard_attention_model(self, test_mode = False):
    ''' This model is Largely based on [A Decomposable Attention Model, Ankur et al.] '''
    # 0, (Optional) Set the upper limit of GPU memory
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    set_session(tf.Session(config=config))

    # 1, Embedding the input and project the embeddings
    premise = Input(shape=(self.SentMaxLen,), dtype='int32')
    hypothesis = Input(shape=(self.SentMaxLen,), dtype='int32')
    embed_p = self.Embed(premise) # [batchsize, Psize, Embedsize]
    embed_h = self.Embed(hypothesis) # [batchsize, Hsize, Embedsize]
    EmbdProject = TimeDistributed(Dense(200,
                                   activation='relu',
                                   kernel_regularizer=l2(self.L2Strength),
                                   bias_regularizer=l2(self.L2Strength)))
    embed_p = Dropout(self.DropProb)(EmbdProject(embed_p)) # [batchsize, Psize, units]
    embed_h = Dropout(self.DropProb)(EmbdProject(embed_h)) # [batchsize, Hsize, units]

    # 2, Score each embeddings and calc score matrix Eph.
    F_p, F_h = embed_p, embed_h
    for i in range(2): # Applying Decomposable Score Function
      scoreF = TimeDistributed(Dense(200,
                                     activation='relu',
                                     kernel_regularizer=l2(self.L2Strength),
                                     bias_regularizer=l2(self.L2Strength)))
      F_p = Dropout(self.DropProb)(scoreF(F_p)) # [batch_size, Psize, units]
      F_h = Dropout(self.DropProb)(scoreF(F_h)) # [batch_size, Hsize, units]
    Eph = keras.layers.Dot(axes=(2, 2))([F_p, F_h]) # [batch_size, Psize, Hsize]

    # 3, Normalize score matrix and get alignment
    Ep = Lambda(lambda x:keras.activations.softmax(x))(Eph) # [batch_size, Psize, Hsize]
    Eh = keras.layers.Permute((2, 1))(Eph) # [batch_size, Hsize, Psize)
    Eh = Lambda(lambda x:keras.activations.softmax(x))(Eh) # [batch_size, Hsize, Psize]
    PremAlign = keras.layers.Dot((2, 1))([Ep, embed_h])
    HypoAlign = keras.layers.Dot((2, 1))([Eh, embed_p])

    # 4, Concat original and alignment, score each pair of alignment
    PremAlign = keras.layers.concatenate([embed_p, PremAlign]) # [batch_size, PreLen, 2*Size]
    HypoAlign = keras.layers.concatenate([embed_h, HypoAlign])# [batch_size, Hypo, 2*Size]
    for i in range(2):
      scoreG = TimeDistributed(Dense(200,
                                     activation='relu',
                                     kernel_regularizer=l2(self.L2Strength),
                                     bias_regularizer=l2(self.L2Strength)))
      PremAlign = scoreG(PremAlign) # [batch_size, Psize, units]
      HypoAlign = scoreG(HypoAlign) # [batch_size, Hsize, units]
      PremAlign = Dropout(self.DropProb)(PremAlign)
      HypoAlign = Dropout(self.DropProb)(HypoAlign)

    # 5, Sum all these scores, and make final judge according to sumed-score
    SumWords = Lambda(lambda X: K.reshape(K.sum(X, axis=1, keepdims=True), (-1, 200)))
    V_P = SumWords(PremAlign) # [batch_size, 512]
    V_H = SumWords(HypoAlign) # [batch_size, 512]
    final = keras.layers.concatenate([V_P, V_H])
    for i in range(2):
      final = Dense(200,
                    activation='relu',
                    kernel_regularizer=l2(self.L2Strength),
                    bias_regularizer=l2(self.L2Strength))(final)
      final = Dropout(self.DropProb)(final)
      final = BatchNormalization()(final)

    # 6, Prediction by softmax
    final = Dense(2 if self.dataset == 'rte' else 3,
                  activation='softmax')(final)
    if test_mode: self.model = Model(inputs=[premise,hypothesis],outputs=[Ep, Eh, final])
    else: self.model = Model(inputs=[premise, hypothesis], outputs=final)

  # TODO Enhanced LSTM Attention model by Qian Chen et al. 2016
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
    # inicijalizacija težina ulazne matrice LSTM-a random Gauss distribucijom (kao u paper kodu)
    # Encoder = Bidirectional(LSTM(units=300, return_sequences=True))  # originalna linija
    # Encoder = Bidirectional(LSTM(units=300, return_sequences=True, kernel_initializer='RandomNormal')) # (nova linija)
    Encoder = Bidirectional(CuDNNLSTM(units=300, return_sequences=True, kernel_initializer='RandomNormal')) # nova linija - CuDNNLSTM

        # originalno, dropout je išao NAKON BiLSTM enkodanja
    # embed_p = Dropout(self.DropProb)(Encoder(embed_p)) # originalna linija
    # embed_h = Dropout(self.DropProb)(Encoder(embed_h)) # originalna linija
        # u paper kodu dropout ide PRIJE BiLSTM encodanja: sljedeće 4 linije
    embed_p = Dropout(self.DropProb)(embed_p) # najprije dropout (nova linija)
    embed_h = Dropout(self.DropProb)(embed_h) # najprije dropout (nova linija)
    embed_p = Encoder(embed_p) # potom BiLSTM enkodiranje (nova linija)
    embed_h = Encoder(embed_h) # potom BiLSTM enkodiranje (nova linija)

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
    # sb_1 = Lambda(lambda x: tf.subtract(x, PremAlign))(embed_p) # originalna linija, subtract layer
    # sb_2 = Lambda(lambda x: tf.subtract(x, HypoAlign))(embed_h) # originalna linija, subtract layer
    sb_1 = keras.layers.Subtract()([embed_p, PremAlign]) # u suštini trebalo bi biti isto (nova linija)
    sb_2 = keras.layers.Subtract()([embed_h, HypoAlign]) # u suštini trebalo bi biti isto (nova linija)

    # konkatenacija [a_, a~, a_ * a~, a_ - a~], isto za b_, b~
    PremAlign = keras.layers.Concatenate()([embed_p, PremAlign, sb_1, mm_1,])  # [batch_size, Psize, 2*unit]
    HypoAlign = keras.layers.Concatenate()([embed_h, HypoAlign, sb_2, mm_2])  # [batch_size, Hsize, 2*unit]
        # u paperu nema ovog dropouta pa je zakomentiran
    # PremAlign = Dropout(self.DropProb)(PremAlign) # originalna linija
    # HypoAlign = Dropout(self.DropProb)(HypoAlign) # originalna linija
    # ff layer sa RELU aktivacijama
    Compresser = TimeDistributed(Dense(300,
                                       kernel_regularizer=l2(self.L2Strength),
                                       bias_regularizer=l2(self.L2Strength),
                                       activation='relu'),
                                 name='Compresser')
    PremAlign = Compresser(PremAlign)
    HypoAlign = Compresser(HypoAlign)

    # 5, Final biLST < Encoder + Softmax Classifier
    # Decoder = Bidirectional(LSTM(units=300, return_sequences=True), # originalna linija 
		# inicijalizacija težina ulazne matrice LSTM-a random Gauss distribucijom (kao u paper kodu)
    # Decoder = Bidirectional(LSTM(units=300, return_sequences=True, kernel_initializer='RandomNormal'),
    Decoder = Bidirectional(CuDNNLSTM(units=300, return_sequences=True, kernel_initializer='RandomNormal'), # nova linija: CuDNNLSTM
                            name='finaldecoder')  # [-1,2*units] # originalno: name='finaldecoer'
        # originalne 2 linije: originalno dropout ide POSLIJE primjene dekodera nad PremAlign i HypoAlign
    # final_p = Dropout(self.DropProb)(Decoder(PremAlign)) # originalna linija
    # final_h = Dropout(self.DropProb)(Decoder(HypoAlign)) # originalna linija
        # u paperu, dropout ide direktn na izlaz ff layera, dakle PRIJE dekodera, ovako (sljedeće 4 linije):
    PremAlign = Dropout(self.DropProb)(PremAlign) # nova linija
    HypoAlign = Dropout(self.DropProb)(HypoAlign) # nova linija
    final_p = Decoder(PremAlign) # nova linija
    final_h = Decoder(HypoAlign) # nova linija

    AveragePooling = Lambda(lambda x: K.mean(x, axis=1)) # outs [-1, dim]
    MaxPooling = Lambda(lambda x: K.max(x, axis=1)) # outs [-1, dim]
    avg_p = AveragePooling(final_p)
    avg_h = AveragePooling(final_h)
    max_p = MaxPooling(final_p)
    max_h = MaxPooling(final_h)
    # konkatenacija avg i max poolinga za hipotezu i premisu
    Final = keras.layers.Concatenate()([avg_p, max_p, avg_h, max_h])
    # dropout layer
    Final = Dropout(self.DropProb)(Final)
    # ff layer sa tanh aktivacijama
    Final = Dense(300,
                  kernel_regularizer=l2(self.L2Strength),
                  bias_regularizer=l2(self.L2Strength),
                  name='dense300_' + self.dataset,
                  activation='tanh')(Final)
                  
    # ovaj layer ne postoji u originalnom kodu
    # ff layer s linearnim aktivacijama  # nova linija
    # Final = Activation(activation = 'linear')(Final) # nova linija
    
    # dropout layer
        # originalno, dropout je s pola vjerojatnosti, u paper kodu nema takve modifikacije
    factor = 1
    if self.LastDropoutHalf:
        factor = 2
    # Final = Dropout(self.DropProb / 2)(Final) # originalna linija
    Final = Dropout(self.DropProb / factor)(Final) # (nova linija)

    # softmax klasifikator
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
    if os.path.exists(fn) and self.LoadExistingWeights: # nova linija: dodana provjera za self.LoadExistingWeights
        self.model.load_weights(fn, by_name=True)
        print('--------Load Weights Successful!--------')
         
  # returns history of train/val loss/acc values
  def start_train(self):
    """ Starts to Train the entire Model Based on set Parameters """
    # 1, Prep
    callback = [EarlyStopping(patience=self.Patience, verbose=2),
                ReduceLROnPlateau(patience=5, verbose=1),
                CSVLogger(filename=self.rnn_type+'log.csv'),
                # ModelCheckpoint(filepath=self.rnn_type + '_' + self.dataset + '.check',
                ModelCheckpoint(filepath=self.Timestamp + '_' + self.dataset + 'weights.{epoch:02d}-{val_loss:.2f}.check',
                                save_best_only=False,
                                save_weights_only=True)]
    # 2, Train
    self.History = self.model.fit(x = [self.train[0],self.train[1]],
                   y = self.train[2],
                   batch_size = self.BatchSize,
                   epochs = self.MaxEpoch,
                   # validation_data=([self.test[0], self.test[1]], self.test[2]), # originalno
                   # promijenjena linija jer se self.test već koristi u evaluate_on_test()
                   validation_data=([self.validation[0], self.validation[1]], self.validation[2]),
                   callbacks = callback)
    self.Options['History'] = self.History.history
    self.format_report()
    return self.History
    # 3, Evaluate
    # self.evaluate_on_test() # originalna linija
    # self.evaluate_on_set(set = 'validation')
    # self.evaluate_on_set(set = 'test')

  # def evaluate_on_test(self):
  # eval_set: proslijediti točno ime datoteke nad kojom želimo testirati (bez ekstenzije)
  def evaluate_on_set(self, eval_set = 'snli_test', filename = None):
    # provjerava valjanost eval_set argumenta
    assert eval_set in ['snli_validation', 'snli_test', 
                         'mnli_validation_matched',
                         'mnli_validation_mismatched']
    dataset = None
    
    if filename is not None: # učitava weighte iz dane datoteke
        self.model.load_weights(filename)
    # ili pokušava defaultne weighte
    elif os.path.exists(self.rnn_type + '_' + self.dataset + '.check'):
        self.model.load_weights(self.rnn_type + '_' + self.dataset + '.check') # revert to the best model
    else: # inače inicijalizira random
        print('No weights found for model!')
        print('Using random initialized weights...')
    # ako testiramo nad istim datasetom, učitaj ga iz članskih varijabli
    if (self.dataset == 'snli' and 'snli' in eval_set) or (self.dataset == 'mnli' and 'mnli' in eval_set):
        if eval_set == 'snli_validation' or eval_set == 'mnli_validation_matched':
            dataset = self.validation
        elif eval_set == 'snli_test' or eval_set == 'mnli_validation_mismatched':
            dataset = self.test
    # ako testiramo unakrsno, učitaj odgovarajući dataset iz datoteke
    else:
        print('Loading ' + eval_set + ' data...')
        dataset = json.loads(open(eval_set + '.json', 'r').read())
        dataset[2] = np_utils.to_categorical(dataset[2], 3)
        dataset = self.padd(dataset)
    # evaluacija
    loss, acc = self.model.evaluate([dataset[0], dataset[1]], 
                                    dataset[2], batch_size=self.BatchSize)
    
    # print("Test: loss = {:.5f}, acc = {:.3f}%".format(loss, acc)) # originalna linija
    print('Trained on: ' + self.dataset.upper())
    print('Evaluated on: ' + eval_set.title() + ": loss = {:.5f}, acc = {:.4f}%".format(loss, acc))
    return (loss, acc)
    # TODO ovaj dalje dio funkcije (RTE) nije održavan, treba ga uskladiti sa SNLI dijelom
    # elif self.dataset == 'rte':
      # true_posi, real_true, pred_true = 0, 0, 0
      # count, left, stime = 0, len(self.test[0]), time.time()
      # for prem, hypo, truth in zip(self.test[0], self.test[1], self.test[2]):
      # for prem, hypo, truth in zip(self.train[0], self.train[1], self.train[2]):
        # prem = np.expand_dims(np.reshape(prem, -1), 0)
        # hypo = np.expand_dims(np.reshape(hypo, -1), 0)
        # predict = np.reshape(self.model.predict(x=[prem, hypo], batch_size=1), -1)
        # predict, truth = np.argmax(predict), np.argmax(truth)
        # if predict == truth and truth == 1:
          # true_posi += 1
        # if predict == 1:
          # pred_true += 1
        # if truth == 1:
          # real_true += 1
        # count += 1
        # if len(self.test[0]) - left >= 1024: break
        # if time.time() - stime > 1:
          # stime = time.time()
          # left -= count
          # print("{}/s | {}/{} | {:.0f} | p = {:.3f} | r = {:.3f}".format(count, len(self.test[0]) - left,
                                                                     # len(self.test[0]), left / count,
                                                                     # true_posi / pred_true,
                                                                     # true_posi / real_true))
          # count = 0
      # print("true_posi = {}, real_true = {}, pred_true = {}".format(true_posi, real_true, pred_true))
      # p, r = true_posi/pred_true, true_posi/real_true
      # print("prec = {:.4f}, recall = {:.4f}, 2pr/(p+r) = {:.4f}".format(p, r, 2*p*r/(p+r)))

  def evaluate_rte_by_snli_model(self, threshold = 0.5):
    assert self.dataset == 'snli'
    def padding(x, MaxLen):
      return pad_sequences(sequences=self.indexer.texts_to_sequences(x), maxlen=MaxLen)
    def pad_data(x):
      return padding(x[0], self.SentMaxLen), padding(x[1], self.SentMaxLen), x[2]
    test_data = pad_data(json.loads(open('RTE_test.json', 'r').read()))
    true_posi, real_true, pred_true = 0, 0, 0
    count, left, stime = 0, len(self.test[0]), time.time()
    for prem, hypo, truth in zip(test_data[0], test_data[1], test_data[2]):
      prem = np.expand_dims(prem, 0)
      hypo = np.expand_dims(hypo, 0)
      ans = np.reshape(self.model.predict(x=[prem, hypo], batch_size=1), -1)  # PREDICTION
      ans = np.delete(ans, 1, 0) # delete 'neutral' output
      e_x = np.exp(ans - np.max(ans))
      ans = e_x / e_x.sum() # reapply softmax
      pred_label = 1 if ans[1] > threshold else 0
      if pred_label == 1: pred_true += 1
      if truth == 1: real_true += 1
      if pred_label == truth and truth == 1 : true_posi += 1
      count += 1
      if time.time() - stime > 1:
        stime = time.time()
        left -= count
        print("{}/s | {}/{} | {:.0f} | p = {:.3f} | r = {:.3f}".format(count, len(self.test[0]) - left,
                                                                       len(self.test[0]), left / count,
                                                                       true_posi / pred_true,
                                                                       true_posi / real_true))
        count = 0
    print("true_posi = {}, real_true = {}, pred_true = {}".format(true_posi, real_true, pred_true))
    p, r = true_posi / pred_true, true_posi / real_true
    print("prec = {:.4f}, recall = {:.4f}, 2pr/(p+r) = {:.4f}".format(p, r, 2 * p * r / (p + r)))

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
