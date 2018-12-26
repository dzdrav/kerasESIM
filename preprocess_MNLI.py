import preprocess_SNLI
import json

if  __name__ == '__main__':
    """
    preprocess MNLI
    """
    dataset = 'mnli_train'
    in_filepath = 'multinli_1.0/multinli_1.0_train.jsonl'
    prems_arr, hypos_arr, label_arr = preprocess_SNLI.load_data(in_filepath)

    out_filepath = '{}.json'.format(dataset)
    open(out_filepath, 'w').write(json.dumps([prems_arr, hypos_arr, label_arr]))


    dataset = 'mnli_validation_matched'
    in_filepath = 'multinli_1.0/multinli_1.0_dev_matched.jsonl'
    prems_arr, hypos_arr, label_arr, category_arr = preprocess_SNLI.load_data(in_filepath, additional_label = 'genre')

    out_filepath = '{}.json'.format(dataset)
    open(out_filepath, 'w').write(json.dumps([prems_arr, hypos_arr, label_arr, category_arr]))


    dataset = 'mnli_validation_mismatched'
    in_filepath = 'multinli_1.0/multinli_1.0_dev_mismatched.jsonl'
    prems_arr, hypos_arr, label_arr, category_arr = preprocess_SNLI.load_data(in_filepath, additional_label = 'genre')

    out_filepath = '{}.json'.format(dataset)
    open(out_filepath, 'w').write(json.dumps([prems_arr, hypos_arr, label_arr, category_arr]))
