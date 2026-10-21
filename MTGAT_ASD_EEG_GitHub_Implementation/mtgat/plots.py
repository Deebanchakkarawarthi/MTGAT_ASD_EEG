import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def plot_training_history(csv_path, output_path):
    import pandas as pd
    d = pd.read_csv(csv_path); fig, ax = plt.subplots()
    ax.plot(d.epoch,d.train_loss,label="Training loss")
    ax.plot(d.epoch,d.val_loss,label="Validation loss")
    ax.set(xlabel="Epoch",ylabel="Cross-entropy loss",title="Training history")
    ax.legend(); fig.tight_layout(); fig.savefig(output_path,dpi=300); plt.close(fig)

def plot_roc_pr(y_true, probabilities, prefix):
    y=np.asarray(y_true); p=np.asarray(probabilities)[:,1]
    fpr,tpr,_=roc_curve(y,p); fig,ax=plt.subplots()
    ax.plot(fpr,tpr,label=f"AUROC={auc(fpr,tpr):.3f}"); ax.plot([0,1],[0,1],"--")
    ax.set(xlabel="False-positive rate",ylabel="True-positive rate",title="ROC curve")
    ax.legend(); fig.tight_layout(); fig.savefig(str(prefix)+"_roc.png",dpi=300); plt.close(fig)
    precision,recall,_=precision_recall_curve(y,p); fig,ax=plt.subplots()
    ax.plot(recall,precision,label=f"AUPRC={average_precision_score(y,p):.3f}")
    ax.set(xlabel="Recall",ylabel="Precision",title="Precision-recall curve")
    ax.legend(); fig.tight_layout(); fig.savefig(str(prefix)+"_pr.png",dpi=300); plt.close(fig)
