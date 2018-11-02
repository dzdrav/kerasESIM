import tfRNN
import sys
from keras.models import load_model

if __name__ == '__main__':
    if len(sys.argv) == 2:
        weights = sys.argv[1]
        weights += '.check'
    else:
        weights = None
    options = {
               'BatchSize': 32,
               'L2Strength': 0.0,
               'GradientClipping': 3.07,
               'LearningRate': 4e-4,
               'Optimizer': 'nadam',
               # 'RetrainEmbeddings': True,
               # govori nam iz koje datoteke učitati konfiguraciju train i val skupova
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
        # md.label_test_file()
    else:
        md.evaluate_on_test_sets(weights_file = weights)
        md.evaluate_on_all_mnli_categories(weights_file = weights)
    
    # _test = False
    # If test_mode is set, then the attention visualization heatmap will also be saved as file
    # md.create_model(test_mode = _test)
    #md.create_standard_attention_model()
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))