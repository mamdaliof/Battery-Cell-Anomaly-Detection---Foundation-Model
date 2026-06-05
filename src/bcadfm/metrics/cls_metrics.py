from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix


@dataclass
class ClassificationMetricsConfig:
    """Configuration placeholder for classification metrics.

    Currently no options, but kept for future extensions (e.g. averaging method).
    """

    average: str = "binary"  # averaging method for precision/recall/F1 in binary classification


def compute_cls_metrics(eval_pred) -> Dict[str, float]:
    """Compute classification metrics for binary classification.

    Expects eval_pred to be a tuple (logits, labels) as provided by HF Trainer.
    Returns a dict with keys that Trainer will prefix with `eval_` during evaluation.
    """

    logits, labels = eval_pred
    # logits: (num_examples, num_classes)
    # labels: (num_examples,)
    preds = np.argmax(logits, axis=-1)

    # Basic metrics
    acc = accuracy_score(labels, preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )

    # AUROC: only defined when there are at least two classes present in labels
    try:
        # Use probability/logit for the positive class (assumed to be class 1)
        if logits.shape[1] == 2:
            pos_scores = logits[:, 1]
        else:
            # Fall back to max logit as score if more than 2 classes
            pos_scores = np.max(logits, axis=1)
        auroc = roc_auc_score(labels, pos_scores)
    except ValueError:
        auroc = float("nan")

    # Confusion matrix: TN, FP, FN, TP
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }
