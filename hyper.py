import tfRNN
import os, time, math, pickle, csv
import numpy as np
from hyperopt import hp, tpe, Trials, fmin, STATUS_OK
from functools import partial
from timeit import default_timer as timer
from json_tricks import dump

# ciljna funkcija optimizacije parametara - vrti model
def objective(params, outFile = 'trials.csv'):
    # inicijalizacija modela parametrima
    eval_timestamp = time.strftime("%Y%m%d%H%M", time.localtime())
    print("Parameters for this run: ", params)
    start = timer()
    md = tfRNN.AttentionAlignmentModel(options = params, annotation = 'EAM', dataset = 'snli')
    md.prep_data()
    md.prep_embd()
    md.create_enhanced_attention_model()
    md.compile_model()
    history = md.start_train() # treniranje modela
    
    # evaluiramo uspjeh modela
    val_score = md.evaluate_on_set(set = 'validation') # vraća (loss, acc)
    run_time = timer() - start

    # informacije o ovoj evaluaciji zapisujemo odmah u datoteku
    dir = 'trials/'
    if not os.path.exists(dir):
        os.mkdir(dir)
    f_eval_data = dir + eval_timestamp + '_trial.json'
    with open(f_eval_data, 'w') as f:
        dump(params, f, indent=2)
        # json.dump(val_score, f, indent=2)
        dump(history.history, f, indent=2)
        
    # vraćamo dict relevantnih vrijednosti
    return {'eval_timestamp': eval_timestamp, 'run_time': run_time,
            'loss': val_score[0], 'acc': val_score[1], 
            # 'val_loss': val_score[0],
            'params': params,
            'history': history.history,
            'status': STATUS_OK}


""" ################################################# """

max_trials = 4
trials_step = 10

 # optimizacijski algoritam = Tree Parzen Estimator
tpe_algo = algo=partial(tpe.suggest, n_startup_jobs = max_trials)
f_trials_binary = 'trials_binary.hyperopt' # binary output za resumanje triala

# definiramo prostor hiperparametara (search space)
space = {
        # prvi run optimizacije--
    # 'BatchSize': hp.choice('BatchSize', [128,512]),
    # 'Optimizer': hp.choice('Optimizer', ['adam', 'rmsprop', 'nadam']),
    # 'LearningRate': hp.choice('LearningRate', [1e-4, 4e-4, 1e-3]),
    # 'L2Strength': hp.choice('L2Strength', [0.0, 1e-5]),
    # 'GradientClipping': hp.uniform('GradientClipping', 0.0, 15.0),
    # 'LastDropoutHalf': hp.choice('LastDropoutHalf', [True, False])
        # do ovdje---------------
    'GradientClipping': hp.uniform('GradientClipping', 3.0, 10.0),
    'LastDropoutHalf': hp.choice('LastDropoutHalf', [True, False]),
    'LowercaseTokens': hp.choice('LowercaseTokens', [True, False]),
    'OOVWordInit': hp.choice('OOVWordInit', ['random', 'zeros'])
    }

try:  # try to load an already saved trials object, and increase the max
    trials = pickle.load(open('trials/' + f_trials_binary, "rb"))
    print("Found saved Trials! Loading " + f_trials_binary + "...")
    max_trials = len(trials.trials) + trials_step
    print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))
except:  # create a new trials object and start searching
    print('Starting new Trials')
    trials = Trials()

# minimizacijska funkcija fmin - core optimizacije
best = fmin(
    fn = objective, 
    space = space,
    algo = tpe_algo,
    max_evals = max_trials,
    trials = trials,
    rstate = np.random.RandomState(50)
    )

# ispis rezultata u datoteku
f_trials_results = 'trials/' + time.strftime("%Y%m%d%H%M_", time.localtime()) + '_trials_summary.json'
trial_results = sorted(trials.results, key = lambda x: x['loss'])

with open(f_trials_results, 'w', encoding = 'utf-8') as outFile:
    dump(trial_results, outFile, indent = 2)

# eksportanje dosadašnjih triala u datoteku
pickle.dump(trials, open('trials/' + f_trials_binary, "wb"))
