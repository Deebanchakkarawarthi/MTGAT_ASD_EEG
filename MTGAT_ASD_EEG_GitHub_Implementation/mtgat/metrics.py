import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, average_precision_score, classification_report

def classification_metrics(y_true, probabilities):
    y = np.asarray(y_true); p = np.asarray(probabilities); pred = p.argmax(1)
    out = {"accuracy": float(accuracy_score(y,pred)),
           "balanced_accuracy": float(balanced_accuracy_score(y,pred)),
           "macro_f1": float(f1_score(y,pred,average="macro",zero_division=0)),
           "weighted_f1": float(f1_score(y,pred,average="weighted",zero_division=0)),
           "precision_macro": float(precision_score(y,pred,average="macro",zero_division=0)),
           "sensitivity_macro": float(recall_score(y,pred,average="macro",zero_division=0)),
           "confusion_matrix": confusion_matrix(y,pred).tolist(),
           "classification_report": classification_report(y,pred,zero_division=0,output_dict=True)}
    if p.shape[1] == 2 and len(np.unique(y)) == 2:
        out["auroc"] = float(roc_auc_score(y,p[:,1]))
        out["auprc"] = float(average_precision_score(y,p[:,1]))
    return out
