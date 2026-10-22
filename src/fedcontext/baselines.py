import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


def features(data):
    return np.concatenate([data["target"], data["movement"].std(1), data["timing"].mean(1),
                           data["timing"].std(1), data["context"]], axis=1)


def fit_baselines(data, meta, cfg, seed, out):
    x = features(data)
    train = np.flatnonzero(meta.partition.eq("train"))
    calibration = np.flatnonzero(meta.partition.eq("calibration"))
    rng = np.random.default_rng(seed)
    ids = rng.choice(train, min(len(train), cfg["baseline_max_samples"]), replace=False)
    models = {"isolation_forest": IsolationForest(n_estimators=200, max_samples="auto", random_state=seed, n_jobs=1),
              "one_class_svm": OneClassSVM(kernel="rbf", gamma="scale", nu=.05)}
    result = {}
    for name, model in models.items():
        use = ids[:5000] if name == "one_class_svm" else ids
        model.fit(x[use])
        scores = -model.score_samples(x)
        low, high = np.quantile(scores[calibration], [cfg["q_low"], cfg["q_high"]], method="higher")
        decision = np.where(scores > high, "Suspicious", np.where(scores > low, "Uncertain", "Trusted"))
        joblib.dump(model, out/(name+".joblib"))
        result[name] = dict(scores=scores, decisions=decision, train_samples=len(use), low=float(low), high=float(high))
    return result
