import tfRNN
from keras.models import load_model


if __name__ == '__main__':
    options = {
               'BatchSize': 128,
               # 'SentMaxLen': 50,
               'L2Strength': 0.0,
               'GradientClipping': 3.07,
               'LearningRate': 4e-4,
               'Optimizer': 'nadam',
               # 'LastDropoutHalf': False,
               # 'OOVWordInit': 'zeros',
               }
    md = tfRNN.AttentionAlignmentModel(options = options, 
                                       annotation='EAM', 
                                       # dataset='snli')
                                       dataset='snli')
    md.prep_data()
    md.prep_embd()
    # _test = False
    # If test_mode is set, then the attention visualization heatmap will also be saved as file
    # md.create_model(test_mode = _test)
    #md.create_standard_attention_model()
    md.create_enhanced_attention_model()
    md.compile_model()
    # md.start_train()
    # md.label_test_file()
    # tfRNN.format_report()
    # md = load_model('ESIM.h5')
    # md.evaluate_on_set(eval_set = 'validation_mismatched')
    # md.evaluate_on_set(eval_set = 'test', filename = 'EAM_snliweights.17-0.41.check')
    weights = 'EAM_snliweights.10-0.38.check'
    md.evaluate_on_set(eval_set = 'snli_test', filename = weights)
    md.evaluate_on_set(eval_set = 'mnli_validation_matched', filename = weights)
    md.evaluate_on_set(eval_set = 'mnli_validation_mismatched', filename = weights)
    

    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
