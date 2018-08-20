import tfRNN


if __name__ == '__main__':
    md = tfRNN.AttentionAlignmentModel(annotation='EAM', dataset='snli')
    md.prep_data()
    md.prep_embd()
    _test = False
    #md.create_model(test_mode = _test)
    #md.create_standard_attention_model()
    md.create_enhanced_attention_model()
    md.compile_model()
    #md.label_test_file()
    md.start_train()
    md.evaluate_on_test()
    # evaluiranje nad RTE datasetom modela učenog nad SNLI
    md.evaluate_rte_by_snli_model(threshold=0.4)

    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
