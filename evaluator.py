from sklearn.metrics import precision_score, classification_report, recall_score
import matplotlib.pyplot as plt

def get_roc_curve(y_true, y_pred, model_name):
    thresholds = [0.0,  0.3, 0.6, 1.0, 1.3, 1.5, 1.8]
    tpr_list, fpr_list = list(), list()

    for thresh in thresholds:
        temp_predictions = [1  if x > thresh else 0 for x in y_pred]


        temp_tpr = tpr(y_true, temp_predictions)
        temp_fpr = fpr(y_true, temp_predictions)

        tpr_list.append(temp_tpr)
        fpr_list.append(temp_fpr)
        print("Threshold", thresh)
        print(classification_report(y_true=y_true,   y_pred=temp_predictions))

    plot_roc(fpr_list, tpr_list, model_name)


def plot_roc(fpr_list, tpr_list, name ="", save=True):
    plt.figure(figsize=(7, 7))
    plt.fill_between(fpr_list, tpr_list, alpha = 0.4)
    plt.plot(fpr_list, tpr_list, lw = 3)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.rcParams["font.size"] = "20"

    if name !="" and save:
        plt.savefig(name + '.png')

    plt.show()


def tpr(y_true, y_pred):
        return recall_score(y_true=y_true, y_pred=y_pred)

def fpr(y_true, y_pred):
    fp = false_pos(y_true, y_pred)
    tn =   true_neg(y_true, y_pred)
    return fp/ (tn+fp)

def true_pos(y_true, y_pred):
    tp = 0
    for yt, yp in zip(y_true, y_pred):
        if  yt == 1 and  yp == 0:
            tp += 1
    return tp

def true_neg(y_true, y_pred):
    tn = 0
    for yt, yp in zip(y_true, y_pred):
        if  yt == 0 and  yp == 0:
            tn += 1
    return tn

def false_pos(y_true, y_pred):
    fp = 0
    for yt, yp in zip(y_true, y_pred):
        if  yt == 0 and  yp  == 1:
            fp += 1
    return fp

def false_neg(y_true, y_pred):
    fn = 0
    for yt, yp in zip(y_true, y_pred):
        if  yt == 1 and  yp == 0:
            fn += 1
    return fn