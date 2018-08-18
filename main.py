import preprocess_RTE
import preprocess_SNLI
import RITutils
import tfRNN

if  __name__ == '__main__':
    """
    preprocess SNLI
    """
    dataset = 'train' # SNLI train
    in_filepath = 'snli_1.0/snli_1.0_{}.jsonl'.format(dataset)
    prems_arr, hypos_arr, label_arr = load_data(in_filepath)

    out_filepath = '{}.json'.format(dataset)
    open(out_filepath, 'w').write(json.dumps([prems_arr, hypos_arr, label_arr]))


    dataset = 'dev'
    in_filepath = 'snli_1.0/snli_1.0_{}.jsonl'.format(dataset)
    prems_arr, hypos_arr, label_arr = load_data(in_filepath)

    out_filepath = 'validation.json'
    open(out_filepath, 'w').write(json.dumps([prems_arr, hypos_arr, label_arr]))


    dataset = 'test'
    in_filepath = 'snli_1.0/snli_1.0_{}.jsonl'.format(dataset)
    prems_arr, hypos_arr, label_arr = load_data(in_filepath)

    out_filepath = '{}.json'.format(dataset)
    open(out_filepath, 'w').write(json.dumps([prems_arr, hypos_arr, label_arr]))

    """
    preprocess RTE
    """
	input = "dataset.xml"
	output_train = "RTE/RTE_train.txt"
	output_test = "RTE/RTE_test.txt"
	XML_to_TXT(input, output_train, output_test, mini = False)

    """
    preprocess RTE (2nd part)
    """
    save_train_data('RTE/RTE_train.txt')
    save_test_data('RTE/RTE_test.txt')
    #merge_data_with_snli()
    pass
