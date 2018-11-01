import tfRNN
from keras.models import load_model

if __name__ == '__main__':
    options = {
               'BatchSize': 128,
               'L2Strength': 0.0,
               'GradientClipping': 3.07,
               'LearningRate': 4e-4,
               'Optimizer': 'nadam',
               # 'RetrainEmbeddings': True,
               # govori nam iz koje datoteke učitati konfiguraciju train i val skupova
               'ConfigTimestamp': '201811010051', 
               'Dataset': 'mnlisnli'
               }
    md = tfRNN.AttentionAlignmentModel(options = options, 
                                       annotation = 'EAM', 
                                       # dataset='snli')
                                       dataset= options['Dataset'])
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
    weights = '201811010051_mnlisnliweights.06-0.68.check'
    md.evaluate_on_set(eval_set = 'snli_test', filename = weights)
    md.evaluate_on_set(eval_set = 'mnli_validation_matched', filename = weights)
    md.evaluate_on_set(eval_set = 'mnli_validation_mismatched', filename = weights)
    
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))