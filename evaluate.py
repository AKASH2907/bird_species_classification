import numpy as np
from keras import backend as K
from os.path import isfile, join
from os import rename, listdir, rename, makedirs
import tensorflow as tf
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    print(true_positives)
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    print(predicted_positives)
    precision = true_positives / (predicted_positives + K.epsilon())
    print(precision)
    return precision


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1(y_true, y_pred):
    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision


    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    print(precision, recall)
    return (2*((precision*recall)/(precision+recall+K.epsilon())))

y_true = np.load('Y_test.npy')
y_pred = np.load('Y_pred.npy')

print(y_true)
print(y_pred)
print(len(y_true))
args = []
preds = []
for i in range(len(y_true)):
    # print(i)
    y_true_arg = np.argmax(y_true[i])
    args.append(y_true_arg)
    preds_arg = np.argmax(y_pred[i])
    preds.append(preds_arg)
    # print(y_true_arg)

print(args)
print(preds)


# average_precision = average_precision_score(args, preds, average='micro')
# print(average_precision)
precision = precision_score(args, preds)
print(precision)
recall = recall_score(args, preds)
print(recall)

f1 = f1_score(args, preds)

print(f1)
# y_true = tf.convert_to_tensor(y_true, np.float32)
# y_pred = tf.convert_to_tensor(y_pred, np.float32)
# print(y_true, y_pred)
# score_precision = precision(y_true, y_pred)

# print(score_precision)