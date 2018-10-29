import tfRNN


if __name__ == '__main__':
    options = {
               'BatchSize': 128,
               'L2Strength': 0.0,
               'GradientClipping': 10.0,
               'LearningRate': 4e-4,
               'Optimizer': 'nadam',
               'LastDropoutHalf': True,
               'OOVWordInit': 'zeros'
               }
    md = tfRNN.AttentionAlignmentModel(options = options, 
                                       annotation='EAM', 
                                       dataset='snli')
    md.prep_data()
    md.prep_embd()
    # _test = False
    # If test_mode is set, then the attention visualization heatmap will also be saved as file
    # md.create_model(test_mode = _test)
    #md.create_standard_attention_model()
    md.create_enhanced_attention_model()
    md.compile_model()
    md.start_train()
    # md.label_test_file()
    # tfRNN.format_report()
    md.evaluate_on_set(set = 'test')

    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
