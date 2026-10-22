"""Explicit metrics for two binary operating policies and three decision states."""
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, precision_recall_curve, auc


def safe_div(a, b):
    return float(a/b) if b else None


def evaluate(y, scores, decisions, policy="strict"):
    y = np.asarray(y, dtype=int)
    decisions = np.asarray(decisions)
    if policy not in ("strict", "alert") or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid policy or labels")
    pred = decisions == "Suspicious" if policy == "strict" else decisions != "Trusted"
    tn, fp, fn, tp = [int(n) for n in confusion_matrix(y, pred, labels=[0, 1]).ravel()]
    precision, recall = safe_div(tp, tp+fp), safe_div(tp, tp+fn)
    result = dict(policy=policy, n=len(y), attack_windows=int(y.sum()), benign_windows=int((y==0).sum()),
                  tp=tp, fp=fp, tn=tn, fn=fn, precision=precision, recall=recall,
                  f1=safe_div(2*tp, 2*tp+fp+fn), fpr=safe_div(fp, fp+tn),
                  accuracy=safe_div(tp+tn, len(y)), uncertainty_rate=float(np.mean(decisions=="Uncertain")),
                  benign_uncertainty=safe_div(((y==0)&(decisions=="Uncertain")).sum(), (y==0).sum()),
                  attack_uncertainty=safe_div(((y==1)&(decisions=="Uncertain")).sum(), (y==1).sum()),
                  decided_coverage=float(np.mean(decisions!="Uncertain")), roc_auc=None, average_precision=None, pr_auc_trapezoid=None)
    if len(np.unique(y)) == 2:
        p, r, _ = precision_recall_curve(y, scores)
        result.update(roc_auc=float(roc_auc_score(y, scores)), average_precision=float(average_precision_score(y, scores)),
                      pr_auc_trapezoid=float(auc(r, p)))
    return result


def episode_delays(predictions):
    """Consecutive positive-label episodes among scored windows, not unobserved attack start."""
    result = []
    for (session, pseudo), g in predictions.groupby(["session", "pseudonym"], sort=False):
        g = g.sort_values("rx_time", kind="stable")
        episode = []
        def emit(rows):
            if not rows:
                return
            first = rows[0]
            detected = next((r for r in rows if r["decision"] == "Suspicious"), None)
            result.append(dict(session=session, pseudonym=pseudo, variant=first["variant"],
                first_scored_attack_time=first["rx_time"], scored_attack_windows=len(rows),
                detected=detected is not None,
                delay_s=None if detected is None else detected["rx_time"]-first["rx_time"]))
        for row in g.to_dict("records"):
            if row["label"] == 1:
                if episode and row["rx_time"]-episode[-1]["rx_time"] > 5:
                    emit(episode)
                    episode = []
                episode.append(row)
            else:
                emit(episode)
                episode = []
        emit(episode)
    return result
