# %% Import packages

from sklearn import metrics

# %%

def confusion_matrix(y_true, y_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return tn, fp, fn, tp

# %%

def f1(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred)
    return (2 * tp) / (2 * tp + fp +fn)

# %%

def error_metrics(y_true, y_pred):
    metrics_dict = dict()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred)
    metrics_dict['tn'] = tn # True negative
    metrics_dict['fp'] = fp # False positive
    metrics_dict['fn'] = fn # False negative
    metrics_dict['tp'] = tp # True positive

    metrics_dict['p'] = metrics_dict['tp'] + metrics_dict['fn'] # Positive (true condition)
    metrics_dict['n'] = metrics_dict['tn'] + metrics_dict['fp'] # Negative (true condition)

    # tpr: true positive rate, sensitity, recall (probability of detection)
    metrics_dict['tpr'] = metrics_dict['tp'] / metrics_dict['p']
    # fnr: false negative rate (miss rate)
    metrics_dict['fnr'] = 1 - metrics_dict['tpr']
    # tnr: true negative rate, specificity, selectivity
    metrics_dict['tnr'] = metrics_dict['tn'] / metrics_dict['n']
    # fpr: false positive rate (probability of false alarm)
    metrics_dict['fpr'] = 1 - metrics_dict['tnr']
    # ppv: positive predictive value, precision
    metrics_dict['ppv'] = metrics_dict['tp'] / (metrics_dict['tp'] + metrics_dict['fp'])
    # acc: accuracy
    metrics_dict['acc'] = (metrics_dict['tp'] + metrics_dict['tn']) / (metrics_dict['p'] + metrics_dict['n'])
    # f1: F1 score
    metrics_dict['f1'] = 2 * metrics_dict['ppv'] * metrics_dict['tpr'] / (metrics_dict['ppv'] + metrics_dict['tpr'])

    return metrics_dict
