import numpy as np
import pandas as pd
from CPC import extract_cc_statistics
import os
import sys


def compute_class_conditional_q_hat(predictions, labels, alpha):
    # for each instance, make a prediction and get the softmax score of the true label of that instance
    class_1 = np.array(predictions)
    class_0 = 1 - class_1
    conformal_scores_0 = []
    conformal_scores_1 = []

    for i in range(len(predictions)):
        if labels[i] == 0:
            pred_score = class_0[i]
            conformal_scores_0.append(1 - pred_score)
        else:
            pred_score = class_1[i]
            conformal_scores_1.append(1 - pred_score)

    n_0 = len(conformal_scores_0)
    n_1 = len(conformal_scores_1)

    #compute quantile
    q_level_0 = np.ceil((n_0 + 1) * (1 - alpha)) / n_0
    qhat_0 = np.quantile(conformal_scores_0, q_level_0, method='higher')

    q_level_1 = np.ceil((n_1 + 1) * (1 - alpha)) / n_1
    qhat_1 = np.quantile(conformal_scores_1, q_level_1, method='higher')
    #print("quantile value for class 0: "+ str(qhat_0))
    #print("quantile value for class 1: "+ str(qhat_1))

    return qhat_0, qhat_1


def comput_set_correctness(set_0, set_1, tl):
    correct_0 = -1
    correct_1 = -1
    if (len(set_0) != 0 and tl == 0):
        correct_0 = True
    elif (len(set_0) != 0 and tl == 1):
        correct_0 = -1
    else:
        correct_0 = False

    if (len(set_1) != 0 and tl == 1):
        # or (len(set_1) == 0 and tl == 0)
        correct_1 = True
    elif (len(set_1) != 0 and tl == 0):
        correct_1 = -1
    else:
        correct_1 = False

    return correct_0, correct_1

def class_cond_CP(t_predictions, t_true_labels, quantile_0, quantile_1, alpha, c_threshold, i=0):
    if isinstance(t_predictions, pd.Series) or isinstance(t_predictions, pd.DataFrame):
        t_predictions = t_predictions.tolist()
    cc_res = []
    for tp in t_predictions:
        cc_t_prob = get_classes_prob(tp)
        set_0, set_1 = compute_class_cond_CP_set(quantile_0, quantile_1, cc_t_prob)
        # computes m_CC-CP: class conditional CP that classifies instances as either Clean or Fault-prone, but not both
        set_0_sp, set_1_sp = compute_class_cond_CP_set(quantile_0, quantile_1, cc_t_prob, c_threshold)
        tl = t_true_labels[t_predictions.index(tp)]

        # compute class-specific precision, i.e., how many of the sets contain the correct label
        correct_0, correct_1 = comput_set_correctness(set_0, set_1, tl)
        correct_0_sp, correct_1_sp = comput_set_correctness(set_0_sp, set_1_sp, tl)

        cc_res.append((cc_t_prob, tl, set_0, len(set_0), correct_0, set_1, len(set_1), correct_1, set_0_sp, len(set_0_sp), correct_0_sp, set_1_sp, len(set_1_sp, correct_1_sp) ))
    cc_df = pd.DataFrame(cc_res, columns=['softmax','true label', 'set_0','nr_set_0','correct_0', 'set_1', 'nr_set_1', 'correct_1', 'set_0_single_pred','nr_set_0_sp','correct_0_sp', 'set_1_single_pred', 'nr_set_1_sp', 'correct_1_sp'])
    cc_df.to_excel('normal_and_CC-CP_with_ALPHA' + str(alpha) + '_' + str(i) + '.xlsx')
    cp_validity_c0 = (float)(len(cc_df[cc_df['correct_0'] == True]) / len(cc_df[cc_df['nr_set_0'] ==1]))
    cp_validity_c1 = (float)(len(cc_df[cc_df['correct_1'] == True]) / len(cc_df[cc_df['nr_set_1'] ==1]))
    cp_validity_c0_sp = (float)(len(cc_df[cc_df['correct_0_sp'] == True]) / len(cc_df[cc_df['nr_set_0_sp'] == 1]))
    cp_validity_c1_sp = (float)(len(cc_df[cc_df['correct_1_sp'] == True]) / len(cc_df[cc_df['nr_set_1_sp'] == 1]))
    label_defect = len(cc_df[cc_df['true label'] == 1])

    extract_cc_statistics(len(t_predictions), label_defect, cp_validity_c0, len(cc_df[cc_df['nr_set_0'] ==1]), len(cc_df[cc_df['correct_0'] == True]), cp_validity_c1, len(cc_df[cc_df['nr_set_1'] ==1]), len(cc_df[cc_df['correct_1'] == True]), cp_validity_c0_sp, len(cc_df[cc_df['nr_set_0_sp'] == 1]), len(cc_df[cc_df['correct_0_sp'] == True]), cp_validity_c1_sp, len(cc_df[cc_df['nr_set_1_sp'] == 1]), len(cc_df[cc_df['correct_1_sp'] == True]))



def get_classes_prob(prediction):
    # for each instance, make a prediction and get the softmax score of the true label of that instance
    class_1 = prediction
    class_0 = 1 - class_1
    return np.array([class_0, class_1])

def compute_class_cond_CP_set(qhat_0, qhat_1, predict_scores):
    prediction_set_0 = []
    prediction_set_1 = []
    set = 0

    if predict_scores[0] >= (1-qhat_0):
        prediction_set_0.append((predict_scores[0], 0))
        set = set +1
    if predict_scores[1] >= (1-qhat_1):
        prediction_set_1.append((predict_scores[1], 1))
        set = set + 1

    return prediction_set_0, prediction_set_1, set

def get_predicted_label(predict_scores, c_threshold):
    if (predict_scores[1] >= c_threshold):
        return 1
    else:
        return 0

def compute_class_cond_CP_set(qhat_0, qhat_1, predict_scores, c_threshold):
    "m_CC-CP: This function is similar to the compute_class_cond_CP_set() with the only difference that it does not compute prediction sets for both possible labels but only for the label that is predicted"
    prediction_set_0 = []
    prediction_set_1 = []
    c0 = predict_scores[0]
    c1 = predict_scores[1]
    if (c1 >= c_threshold):
        if (c1 >= (1 - qhat_1)):
            print("prediction probability for class = 1: ", c1)
            prediction_set_1.append((c1, 1))
        else:
            print("prediction probability for class = 1 DOES NOT SURPASS THRESHOLD ", c1)

    elif (c0 >= c_threshold):
           if c0 >= (1 - qhat_0):
                prediction_set_0.append((c0, 0))
           else:
               print("prediction probability for class = 0 DOES NOT SURPASS THRESHOLD ", c0)
    return prediction_set_0, prediction_set_1
