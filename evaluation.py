import numpy as np

def readlines(input_file):
    ret = []
    for line in open(input_file, 'r'):
        ret.append(line.strip())
    return ret

def _clss_accuracy(labels, predicts):
    """Compute accuracy for classification"""
    total_count = 0.
    match = 0.0
    for ind, label in enumerate(labels):
      max_pred_index = np.argmax(np.array(predicts[ind]))
      if max_pred_index == label:
          match += 1
      total_count += 1
    return match / total_count

def _clss_accuracy_is(labels, predicts, oos_label_id=47):
    """Compute accuracy for classification"""
    total_count = 0.
    match = 0.0
    for ind, label in enumerate(labels):
      max_pred_index = np.argmax(np.array(predicts[ind]))
      if label != oos_label_id:
          if max_pred_index == label:
              match += 1
          total_count += 1
    return match / total_count

def _clss_prec_recall(labels, predicts, oos_label_id=47):
    """Compute accuracy for classification"""
    total_count = 0.
    tp, fp, fn = 0.0, 0.0, 0.0
    for ind, label in enumerate(labels):
      max_pred_index = np.argmax(np.array(predicts[ind]))
      if label == oos_label_id:
          if max_pred_index == label:
              tp += 1
          else:
              fn += 1
          total_count += 1
      else:
          if max_pred_index == label:
              fp += 1
    rec = (tp / (tp+fn)) if total_count > 0 else -1.0
    pre = (tp / (tp+fp)) if (tp+fp)>0 else -1.0
    return pre, rec


def eval(label_path, prediction_path, label_id_path, oos_id=-1):

    label_lines = readlines(label_path)
    prediction_lines = readlines(prediction_path)
    label_ids = readlines(label_id_path)
    label_map = dict()
    for intent in label_ids:
        comps = intent.split('\t')
        label_map[comps[0]] = int(comps[1])

    labels = []
    for example in label_lines:
        labels.append(label_map[example])
    predictions = []
    for example in prediction_lines:
        predictions.append([float(a) for a in example.split('\t')])
    labels = labels[:len(predictions)]

    accuracy = _clss_accuracy(labels, predictions)
    is_acc = _clss_accuracy_is(labels, predictions, oos_label_id=oos_id)
    oos_pre, oos_rec = _clss_prec_recall(labels, predictions, oos_label_id=oos_id)

    print('Test performance: acc='+'{0:.1f}'.format(accuracy*100.0)
          +', is_acc='+'{0:.1f}'.format(is_acc*100.0)
          +', oos_pre='+'{0:.1f}'.format(oos_pre*100.0)
          +', oos_rec='+'{0:.1f}'.format(oos_rec*100.0))

if __name__ == '__main__':
    eval('data/oos_train/sentences.test.out', 'results/test_results-oos_train-ls0.2.tsv',
         'results/label_ids-oos_train-ls0.2.txt', oos_id=11)