import tfRNN
import sys
from keras.models import load_model

"""
if loading existing weights:
    provide weights timestamp as command line argument
    model is evaluated on all sets
    results are saved to 2 files
else:
    train new model
"""
if __name__ == '__main__':
    if len(sys.argv) == 2:
        weights = sys.argv[1]
        weights += '.check'
    else:
        weights = None
    options = {
               'BatchSize': 256,
               'L2Strength': 0.0,
               'GradientClipping': 3.07,
               'LearningRate': 4e-4,
               'Optimizer': 'nadam',
               # 'RetrainEmbeddings': True,
               'Dataset': 'mnlisnli'
               }

    # weights = '201811010243_mnlisnliweights.05-0.68.check'
    if weights is not None:
        options['ConfigTimestamp'] = weights[:12]

    md = tfRNN.AttentionAlignmentModel(options = options,
                                       annotation = 'EAM',
                                       # dataset='snli')
                                       dataset= options['Dataset'])
    md.prep_data()
    md.prep_embd()

    md.create_enhanced_attention_model()
    md.compile_model()
    if weights is None:
        md.start_train()
    else:
        md.evaluate_on_test_sets(weights_file = weights)
        md.evaluate_on_all_mnli_categories(weights_file = weights)
