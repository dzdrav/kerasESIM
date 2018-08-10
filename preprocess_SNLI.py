#LABEL_MAP = {"entailment": 0, "neutral": 1,"contradiction": 2 }
labels = {'contradiction': 0, 'neutral': 1, 'entailment': 2}

def load_data(path):
    # LABEL_MAP = ispravni labeli
    LABEL_MAP = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
    print("Loading", path)
    #examples = []
    prems_arr = []
    hypos_arr = []
    label_arr = []
    skipped_examples = []
    # otvaramo datoteku za čitanje
    with open(path, 'r') as f:
        # čitamo red po red
        for line in f:
            # učitavamo 1 red kao 1 primjer
            loaded_example = json.loads(line)
            # gold label je konsenzus label
            # ako konsenzus nije jedan od ispravna 3 labela, odbaci primjer
            if loaded_example["gold_label"] not in LABEL_MAP:
              skipped_examples.append(loaded_example)
              #print((loaded_example['annotator_labels'], loaded_example["gold_label"]))
              continue
            # u odgovarajuću listu dodajemo p/h/l
            # sintaksa JSON parsera omogućuje nam asocijativni pristup točno
            # onome što nas zanima
            prems_arr.append(loaded_example["sentence1"])
            hypos_arr.append(loaded_example["sentence2"])
            label_arr.append(LABEL_MAP[loaded_example["gold_label"]])

    print("Loaded {}/{}/{} Skipped {}".format(len(prems_arr), len(hypos_arr), len(label_arr), len(skipped_examples)))

    return (prems_arr, hypos_arr, label_arr)


if  __name__ == '__main__':
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
