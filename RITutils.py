from keras import metrics
from itertools import product
from functools import partial
from keras.losses import mean_squared_error

"""
najprije imamo funkcije za računanje uspjeha (precision, recall, f1score)
potom slijede funkcije iz RITutils.py za handlanje RTE dataseta
"""

def w_categorical_crossentropy(y_true, y_pred):
    weights = np.array([[1., 5.],  # misclassify N -> Y
                        [10., 1.]])# misclassify Y -> N
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.expand_dims(y_pred_max, 1)
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (
        K.cast(weights[c_t, c_p], K.floatx()) *
        K.cast(y_pred_max_mat[:, c_p],
        K.floatx()) *
        K.cast(y_true[:, c_t],K.floatx()))
    return K.categorical_crossentropy(y_pred, y_true) * final_mask

def precision(y_true, y_pred):
    y_true, y_pred = K.argmax(y_true, axis=1), K.argmax(y_pred, axis=1)
    y_true, y_pred = K.cast(y_true, 'float32'), K.cast(y_pred, 'float32')
    TP = K.sum(K.clip(y_true * y_pred, 0, 1)) # how many
    predicted_positives = K.sum(K.clip(y_pred, 0, 1))
    return TP / (predicted_positives + K.epsilon())

def recall(y_true, y_pred):
    y_true, y_pred = K.argmax(y_true, axis=1), K.argmax(y_pred, axis=1)
    y_true, y_pred = K.cast(y_true, 'float32'), K.cast(y_pred, 'float32')
    TP = K.sum(K.clip(y_true * y_pred, 0, 1))  # how many
    # TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    # possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    possible_positives = K.sum(K.clip(y_true, 0, 1))
    return TP / (possible_positives + K.epsilon())

def f1_score(y_true, y_pred):
    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    fscore = 2 * (p * r) / (p + r + K.epsilon())
    return fscore


"""
uzima podatke iz filename.txt i vraća 3 liste: listu premisa, hipoteza, labela
"""
def data_preprocessing(filename):
	# lines = lista redaka (obrnutim redoslijedom) datoteke filename
	lines = open(filename, 'r', encoding='utf-8').read().split('\n')
	# TODO odkomentirati ovo ako ne radi
	lines.reverse()
	prems, hypos, label = [], [], []
	# svaki element liste razdvojimo po razmacima (zanemarujući prvi simbol) te
	# listu tih riječi spremamo u _varijablu
	# _prem sadrži listu premisa
	while lines:
		try:
			_prem = lines.pop().split()[1:]
			_hypo = lines.pop().split()[1:]
			_label = lines.pop().split()[0]
		except IndexError:
			# print("---\nIndex Error")
			break
		# potom u listu rečenica dodajemo
		prems.append(' '.join(_prem))
		hypos.append(' '.join(_hypo))
		# appenda 'Y' ako je _label = 1 ili 'N' ako je _label = 0
		label.append({'Y':1,'N':0}[_label])
	return prems, hypos, label

"""
sprema "train" podatke
zapravo stvara 2 datoteke: RTE_train.json i RTE_valid.json (train/validation)
argument filename je .txt datoteka s podacima za OBA dataseta
"""
def save_train_data(filename):
    prems, hypos, label = data_preprocessing(filename)
    # premisa, hipoteza i labela mora biti jednak broj
    assert len(prems) == len(hypos) == len(label)
    # spt = granica razgraničenja train i validation seta
    spt = int(len(prems) * 0.9)
    # dumpa tri liste (p, h, l) u JSON datoteku
    open('RTE_train.json', 'w').write(json.dumps([prems[:spt], hypos[:spt], label[:spt]]))
    open('RTE_valid.json', 'w').write(json.dumps([prems[spt:], hypos[spt:], label[spt:]]))
    # ispisuje jedan primjer iz seta (u ovom slučaju 88-i po redu)
    print('train example:\n', prems[88], '\n', hypos[88], '\n', label[88])

"""
sprema "test" podatke
argument filename je .txt datoteka s testnim skupom
dumpa ih u JSON datoteku kao u metodi save_train_data()
"""
def save_test_data(filename):
    prems, hypos, label = data_preprocessing(filename)
    assert len(prems) == len(hypos) == len(label)
    open('RTE_test.json', 'w').write(json.dumps([prems, hypos, label]))
    print('test example:\n', prems[88], '\n', hypos[88], '\n', label[88])

"""
spaja formatirani RTE dataset sa SNLI datasetom tako što cijelom SNLI datasetu
spaja labele 'neutral' i 'contradiction' u isti label
jer RTE ima samo 2 labela: 1 (entailment) i 0 (no entailment)
"""
def merge_data_with_snli():
    strn = json.loads(open('train.json', 'r').read())
    rtrn = json.loads(open('RTE_train.json', 'r').read())

    for i, label in enumerate(strn[2]):
        if label == 1: continue
        if label == 2: label = 1
        rtrn[0].append(strn[0][i])
        rtrn[1].append(strn[1][i])
        rtrn[2].append(label)

    open('RTE_train.json', 'w').write(json.dumps(rtrn))


if  __name__ == '__main__':
    """
    preprocess RTE (2nd part)
    """
    save_train_data('RTE/RTE_train.txt')
    save_test_data('RTE/RTE_test.txt')
    #merge_data_with_snli()
    pass
