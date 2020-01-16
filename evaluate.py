import numpy as np
from sklearn.metrics import confusion_matrix


def precision(matrix):
    avg_precise = 0
    for i in range(16):
        tp = matrix[i][i]
        fp = np.sum(matrix[:, i])
        if fp != 0:

            avg_precise += tp / fp

    return avg_precise / 16


def recall(matrix):
    avg_recall = 0
    for i in range(16):
        tp = matrix[i][i]
        fn = np.sum(matrix[i, :])
        avg_recall += tp / fn

    return avg_recall / 16


def f1_score(precision, recall):
    return (2 * precision * recall) / (precision + recall)


if __name__ == "__main__":

    y_true = np.load("prediction_files/Y_test.npy")
    y_pred = np.load("prediction_files/Y_pred_inception_resnet.npy")

    ground_truth = []
    preds = []

    for i in range(len(y_true)):
        y_true_arg = np.argmax(y_true[i])
        ground_truth.append(y_true_arg)
        preds_arg = np.argmax(y_pred[i])
        preds.append(preds_arg)

    confusion_matrix = confusion_matrix(ground_truth, preds)
    # print(confusion_matrix)

    precise = precision(confusion_matrix)
    recall = recall(confusion_matrix)
    f1 = f1_score(precise, recall)

    print(precise, recall, f1)
