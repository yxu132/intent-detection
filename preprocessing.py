import json
import os
import numpy as np
import codecs
import tokenization

def parse_json(json_path):
    ret_sents, ret_labels = dict(), dict()
    with codecs.open(json_path, 'r', encoding='ascii') as json_file:
        json_data = json.load(json_file)

    for dataset in json_data:
        ret_sents[dataset] = []
        ret_labels[dataset] = []
        for example in json_data[dataset]:
            ret_sents[dataset].append(example[0].lower())
            ret_labels[dataset].append(example[1].lower())

    return ret_sents, ret_labels

def get_squad_queries(squad_path, tokenizer):

    queries = []

    with codecs.open(squad_path, 'r', encoding='utf=8') as json_file:
        json_data = json.load(json_file)

    for examples in json_data['data']:
        for paragraph in examples['paragraphs']:
            for qa in paragraph['qas']:
                question = qa['question'].lower()
                tokens = tokenizer.tokenize(question)
                if len(tokens) >= 150:
                    continue
                queries.append(question)


    print('squad has '+str(len(queries))+' queries')

    np.random.seed(0)
    shuffled_ids = np.arange(len(queries))
    queries = np.array(queries)[shuffled_ids]

    return queries[:10000]


def write_to_oos_bin_oversampled(output_dir, ret_sents, ret_labels, squad_queries):
    # write for binary classification: is [in-scope], oos [out-of-scope]
    if not os.path.exists(output_dir + ''):
        os.mkdir(output_dir + '')

    train_sents = np.array(ret_sents['train'] + ret_sents['oos_train'] + list(squad_queries))
    train_labels = np.array(['is'] * len(ret_sents['train']) + ['oos'] * len(ret_sents['oos_train'])
                            + ['oos'] * len(squad_queries))
    np.random.seed(0)
    shuffled_ids = np.arange(len(train_sents))
    np.random.shuffle(shuffled_ids)
    train_sents = train_sents[shuffled_ids]
    train_labels = train_labels[shuffled_ids]

    print(str(np.sum(train_labels == 'oos')))
    print(str(np.sum(train_labels == 'is')))

    with open(output_dir + '/sentences.train.in', 'w') as output:
        output.write('\n'.join(train_sents))
    with open(output_dir + '/sentences.train.out', 'w') as output:
        for label in train_labels:
            output.write('_'.join(label.split()) + '\n')

    with open(output_dir + '/sentences.eval.in', 'w') as output:
        output.write('\n'.join(ret_sents['val'] + ret_sents['oos_val']))
    with open(output_dir + '/sentences.eval.out', 'w') as output:
        for label in ['is'] * len(ret_sents['val']) + ['oos'] * len(ret_sents['oos_val']):
            output.write('_'.join(label.split()) + '\n')

    with open(output_dir + '/sentences.test.in', 'w') as output:
        output.write('\n'.join(ret_sents['test'] + ret_sents['oos_test']))
    with open(output_dir + '/sentences.test.out', 'w') as output:
        for label in ['is'] * len(ret_sents['test']) + ['oos'] * len(ret_sents['oos_test']):
            output.write('_'.join(label.split()) + '\n')

def write_to_oos_bin(output_dir, ret_sents, ret_labels):
    # write for binary classification: is [in-scope], oos [out-of-scope]
    if not os.path.exists(output_dir + ''):
        os.mkdir(output_dir + '')

    train_sents = np.array(ret_sents['train'] + ret_sents['oos_train'])
    train_labels = np.array(['is'] * len(ret_sents['train']) + ['oos'] * len(ret_sents['oos_train']))
    np.random.seed(0)
    shuffled_ids = np.arange(len(train_sents))
    np.random.shuffle(shuffled_ids)
    train_sents = train_sents[shuffled_ids]
    train_labels = train_labels[shuffled_ids]

    with open(output_dir + '/sentences.train.in', 'w') as output:
        output.write('\n'.join(train_sents))
    with open(output_dir + '/sentences.train.out', 'w') as output:
        for label in train_labels:
            output.write('_'.join(label.split()) + '\n')

    with open(output_dir + '/sentences.eval.in', 'w') as output:
        output.write('\n'.join(ret_sents['val'] + ret_sents['oos_val']))
    with open(output_dir + '/sentences.eval.out', 'w') as output:
        for label in ['is'] * len(ret_sents['val']) + ['oos'] * len(ret_sents['oos_val']):
            output.write('_'.join(label.split()) + '\n')

    with open(output_dir + '/sentences.test.in', 'w') as output:
        output.write('\n'.join(ret_sents['test'] + ret_sents['oos_test']))
    with open(output_dir + '/sentences.test.out', 'w') as output:
        for label in ['is'] * len(ret_sents['test']) + ['oos'] * len(ret_sents['oos_test']):
            output.write('_'.join(label.split()) + '\n')

def write_to_oos_bin_downsampled(output_dir, ret_sents, ret_labels):
    # write for binary classification: is [in-scope], oos [out-of-scope]
    if not os.path.exists(output_dir + ''):
        os.mkdir(output_dir + '')

    train_sents = np.array(ret_sents['train'] + ret_sents['oos_train'])
    train_labels_orig = np.array(ret_labels['train'] + ret_labels['oos_train'])
    np.random.seed(0)
    shuffled_ids = np.arange(len(train_sents))
    np.random.shuffle(shuffled_ids)
    train_sents = train_sents[shuffled_ids]
    train_labels_orig = train_labels_orig[shuffled_ids]

    down_sample_per_class = 6

    is_labels = set(ret_labels['train'])
    sampled_num = {label: 0 for label in is_labels}
    downsampled_train_sents, downsampled_train_labels = [], []
    for ind, sent in enumerate(train_sents):
        if train_labels_orig[ind] in sampled_num:
            if sampled_num[train_labels_orig[ind]] < down_sample_per_class:
                downsampled_train_sents.append(sent)
                downsampled_train_labels.append('is')
                sampled_num[train_labels_orig[ind]] += 1
        else:
            downsampled_train_sents.append(sent)
            downsampled_train_labels.append('oos')

    np.random.seed(0)
    shuffled_ids = np.arange(len(downsampled_train_sents))
    np.random.shuffle(shuffled_ids)
    downsampled_train_sents = np.array(downsampled_train_sents)[shuffled_ids]
    downsampled_train_labels = np.array(downsampled_train_labels)[shuffled_ids]

    with open(output_dir + '/sentences.train.in', 'w') as output:
        output.write('\n'.join(downsampled_train_sents))
    with open(output_dir + '/sentences.train.out', 'w') as output:
        for label in downsampled_train_labels:
            output.write('_'.join(label.split()) + '\n')

    with open(output_dir + '/sentences.eval.in', 'w') as output:
        output.write('\n'.join(ret_sents['val'] + ret_sents['oos_val']))
    with open(output_dir + '/sentences.eval.out', 'w') as output:
        for label in ['is'] * len(ret_sents['val']) + ['oos'] * len(ret_sents['oos_val']):
            output.write('_'.join(label.split()) + '\n')

    with open(output_dir + '/sentences.test.in', 'w') as output:
        output.write('\n'.join(ret_sents['test'] + ret_sents['oos_test']))
    with open(output_dir + '/sentences.test.out', 'w') as output:
        for label in ['is'] * len(ret_sents['test']) + ['oos'] * len(ret_sents['oos_test']):
            output.write('_'.join(label.split()) + '\n')


def write_to_oos_inscope(output_dir, ret_sents, ret_labels):
    if not os.path.exists(output_dir + ''):
        os.mkdir(output_dir + '')

    train_sents = np.array(ret_sents['train'])
    train_labels = np.array(ret_labels['train'])
    np.random.seed(0)
    shuffled_ids = np.arange(len(train_sents))
    np.random.shuffle(shuffled_ids)
    train_sents = train_sents[shuffled_ids]
    train_labels = train_labels[shuffled_ids]

    with open(output_dir + '/sentences.train.in', 'w') as output:
        output.write('\n'.join(train_sents))
    with open(output_dir + '/sentences.train.out', 'w') as output:
        for label in train_labels:
            output.write('_'.join(label.split()) + '\n')

    with open(output_dir + '/sentences.eval.in', 'w') as output:
        output.write('\n'.join(ret_sents['val']))
    with open(output_dir + '/sentences.eval.out', 'w') as output:
        for label in ret_labels['val']:
            output.write('_'.join(label.split()) + '\n')

    with open(output_dir + '/sentences.test.in', 'w') as output:
        output.write('\n'.join(ret_sents['test']))
    with open(output_dir + '/sentences.test.out', 'w') as output:
        for label in ret_labels['test']:
            output.write('_'.join(label.split()) + '\n')


def write_to_oos_train(output_dir, ret_sents, ret_labels, label_set_oos_train_to_ids):
    if not os.path.exists(output_dir + ''):
        os.mkdir(output_dir + '')

    train_sents = np.array(ret_sents['train'] + ret_sents['oos_train'])
    train_labels = np.array(ret_labels['train'] + ret_labels['oos_train'])
    np.random.seed(0)
    shuffled_ids = np.arange(len(train_sents))
    np.random.shuffle(shuffled_ids)
    train_sents = train_sents[shuffled_ids]
    train_labels = train_labels[shuffled_ids]

    with open(output_dir + '/sentences.train.in', 'w') as output:
        output.write('\n'.join(train_sents))
    with open(output_dir + '/sentences.train.out', 'w') as output:
        for label in train_labels:
            output.write('_'.join(label.split()) + '\n')

    with open(output_dir + '/sentences.eval.in', 'w') as output:
        output.write('\n'.join(ret_sents['val'] + ret_sents['oos_val']))
    with open(output_dir + '/sentences.eval.out', 'w') as output:
        for label in ret_labels['val'] + ret_labels['oos_val']:
            output.write('_'.join(label.split()) + '\n')

    with open(output_dir + '/sentences.test.in', 'w') as output:
        output.write('\n'.join(ret_sents['test'] + ret_sents['oos_test']))
    with open(output_dir + '/sentences.test.out', 'w') as output:
        for label in ret_labels['test'] + ret_labels['oos_test']:
            output.write('_'.join(label.split()) + '\n')

    with open(output_dir + '/intent_to_ids.txt', 'w') as output:
        for intent in label_set_oos_train_to_ids:
            output.write(str('_'.join(intent.split())) + '\n')

    sent_lens = []
    for dataset in ret_sents:
        sent_lens.extend([len(sent) for sent in ret_sents[dataset]])

    print('max sent length: ' + str(max(sent_lens)))


if __name__ == '__main__':

    ret_sents, ret_labels = parse_json('data/data_imbalanced.json')

    label_set = []
    for dataset in ret_labels:
        label_set.extend(ret_labels[dataset])

    label_set = set(label_set)
    label_set_oos_train = list(label_set) + ['oos']
    label_to_ids = {label: id for id, label in enumerate(list(label_set))}
    label_set_oos_train_to_ids = {label: id for id, label in enumerate(label_set_oos_train)}

    # write_to_oos_train('data/oos_train', ret_sents, ret_labels, label_set_oos_train_to_ids)
    #
    # write_to_oos_bin('data/oos_binary/bin', ret_sents, ret_labels)
    # write_to_oos_inscope('data/oos_binary/in_scope', ret_sents, ret_labels)
    # write_to_oos_bin_downsampled('data/oos_binary/down_sampled_bin', ret_sents, ret_labels)

    tokenizer = tokenization.FullTokenizer(vocab_file='/Users/yxu132/data/bert/uncased_L-12_H-768_A-12/vocab.txt', do_lower_case=True)
    squad_queries = get_squad_queries('/Users/yxu132/pub-repos/decaNLP/data/squad/train-v1.1.json', tokenizer)
    write_to_oos_bin_oversampled('data/oos_binary/over_sampled_bin', ret_sents, ret_labels, squad_queries)