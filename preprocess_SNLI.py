import json

#LABEL_MAP = {"entailment": 0, "neutral": 1,"contradiction": 2 }
labels = {'contradiction': 0, 'neutral': 1, 'entailment': 2}

def load_data(path, additional_label = None):
    # LABEL_MAP = correct labels
    LABEL_MAP = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
    print("Loading", path)
    prems_arr = []
    hypos_arr = []
    label_arr = []
    category_arr = [] # genre
    skipped_examples = []
    with open(path, 'r') as f:
        for line in f:
            # 1 row = 1 example (sentence pair)
            loaded_example = json.loads(line)
            # gold label = consensus label
            # if no consensus = reject example
            if loaded_example["gold_label"] not in LABEL_MAP:
              skipped_examples.append(loaded_example)
              continue
            # add prem/hyp/label to corresponing list
            prems_arr.append(loaded_example["sentence1"])
            hypos_arr.append(loaded_example["sentence2"])
            label_arr.append(LABEL_MAP[loaded_example["gold_label"]])
            if additional_label is not None:
                category_arr.append(loaded_example[additional_label])

    print("Loaded {}/{}/{} Skipped {}".format(len(prems_arr), len(hypos_arr), len(label_arr), len(skipped_examples)))

    if additional_label is not None:
        return (prems_arr, hypos_arr, label_arr, category_arr)
    else:
        return (prems_arr, hypos_arr, label_arr)

if  __name__ == '__main__':
    """
    preprocess SNLI
    """
    dataset = 'snli_train' # SNLI train
    in_filepath = 'snli_1.0/snli_1.0_train.jsonl'
    prems_arr, hypos_arr, label_arr = load_data(in_filepath)

    out_filepath = '{}.json'.format(dataset)
    open(out_filepath, 'w').write(json.dumps([prems_arr, hypos_arr, label_arr]))


    dataset = 'snli_validation'
    in_filepath = 'snli_1.0/snli_1.0_dev.jsonl'
    prems_arr, hypos_arr, label_arr = load_data(in_filepath)

    out_filepath = '{}.json'.format(dataset)
    open(out_filepath, 'w').write(json.dumps([prems_arr, hypos_arr, label_arr]))


    dataset = 'snli_test'
    in_filepath = 'snli_1.0/snli_1.0_test.jsonl'
    prems_arr, hypos_arr, label_arr = load_data(in_filepath)

    out_filepath = '{}.json'.format(dataset)
    open(out_filepath, 'w').write(json.dumps([prems_arr, hypos_arr, label_arr]))
