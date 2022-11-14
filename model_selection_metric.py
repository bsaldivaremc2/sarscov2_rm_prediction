from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
def static_sens(y_true,y_predicted,sample_weight=None,sens_th=0.8,lower_better=True,show_debug=False):
    """
    This metric does not count specificity until sensitivity reaches 0.8. 
    Afterwards, specificity is added to the maximum allowed sensitivity (sens_th=0.8).
    This metric was done for compatibility with MLjar.

    """
    if isinstance(y_true, pd.DataFrame):
        y_true = np.array(y_true)
    if isinstance(y_predicted, pd.DataFrame):
        y_predicted = np.array(y_predicted)
    #
    if len(y_predicted.shape) == 2 and y_predicted.shape[1] == 1:
        y_predicted = y_predicted.ravel()
    #
    if len(y_predicted.shape) == 1:
        y_predicted = (y_predicted > 0.5).astype(int)
    else:
        y_predicted = np.argmax(y_predicted, axis=1)
    
    #t = y_true.round()
    #p = y_predicted.round()
    tn, fp, fn, tp = confusion_matrix(y_true,y_predicted).ravel()
    sens = tp/(tp+fn)
    spec = tn/(tn+fp)
    best_sens = min(sens_th,sens)
    gate = sens//sens_th
    valid_spec = gate*spec
    if show_debug:
        m = "sens: {}. spec: {}. best_sens: {}. gate: {}. valid_spec: {}".format(sens,spec,best_sens,gate,valid_spec)
        print(m)
    metric =  best_sens + valid_spec
    if lower_better:
        metric =  metric*-1
    return metric/(1+sens_th)
