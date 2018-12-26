import tfRNN
import os, time, math, pickle, csv
import numpy as np
from hyperopt import hp, tpe, Trials, fmin, STATUS_OK
from functools import partial
from timeit import default_timer as timer
from json_tricks import dump

def objective(params, outFile = 'trials.csv'):
    eval_timestamp = time.strftime("%Y%m%d%H%M", time.localtime())
    print("Parameters for this run: ", params)
    start = timer()
    # initializing model parameters
    md = tfRNN.AttentionAlignmentModel(options = params, annotation = 'EAM', dataset = 'snli')
    md.prep_data()
    md.prep_embd()
    md.create_enhanced_attention_model()
    md.compile_model()
    # history contains details about each training run
    history = md.start_train() # training the model

    # evaluation of model
    val_score = md.evaluate_on_set(set = 'validation') # returns (loss, acc)
    run_time = timer() - start

    # evaluation data is written to a file
    dir = 'trials/'
    if not os.path.exists(dir):
        os.mkdir(dir)
    f_eval_data = dir + eval_timestamp + '_trial.json'
    with open(f_eval_data, 'w') as f:
        dump(params, f, indent=2)
        dump(history.history, f, indent=2)

    # returns dict with data from this evaluation
    return {'eval_timestamp': eval_timestamp, 'run_time': run_time,
            'loss': val_score[0], 'acc': val_score[1],
            # 'val_loss': val_score[0],
            'params': params,
            'history': history.history,
            'status': STATUS_OK}


""" ################################################# """

max_trials = 4
trials_step = 10

 # probability model = Tree Parzen Estimator
tpe_algo = algo=partial(tpe.suggest, n_startup_jobs = max_trials)
f_trials_binary = 'trials_binary.hyperopt' # binary output for resuming hyperopt

"""
defining search space
keys are hyperparameters, values are allowed values
syntax for values can be found on Hyperopt Github page
"""
space = {
    'BatchSize': hp.choice('BatchSize', [128,512]),
    'Optimizer': hp.choice('Optimizer', ['adam', 'rmsprop', 'nadam']),
    'LearningRate': hp.choice('LearningRate', [1e-4, 4e-4, 1e-3]),
    'L2Strength': hp.choice('L2Strength', [0.0, 1e-5]),
    'GradientClipping': hp.uniform('GradientClipping', 0.0, 15.0),
    'LastDropoutHalf': hp.choice('LastDropoutHalf', [True, False])
    }

# if existing trials object is found, resume optimization from it
try:
    trials = pickle.load(open('trials/' + f_trials_binary, "rb"))
    print("Found saved Trials! Loading " + f_trials_binary + "...")
    max_trials = len(trials.trials) + trials_step
    print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))
# else start new trial
except:
    print('Starting new Trials')
    trials = Trials()

# goal function to minimize
best = fmin(
    fn = objective,
    space = space,
    algo = tpe_algo,
    max_evals = max_trials,
    trials = trials,
    rstate = np.random.RandomState(50)
    )

# writing results to a file
f_trials_results = 'trials/' + time.strftime("%Y%m%d%H%M_", time.localtime()) + '_trials_summary.json'
trial_results = sorted(trials.results, key = lambda x: x['loss'])

with open(f_trials_results, 'w', encoding = 'utf-8') as outFile:
    dump(trial_results, outFile, indent = 2)

# exporting trials to a file
pickle.dump(trials, open('trials/' + f_trials_binary, "wb"))
