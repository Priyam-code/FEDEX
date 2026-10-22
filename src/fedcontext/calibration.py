import itertools
import numpy as np

STREAMS = ("movement", "timing", "context")
SUBSETS = [tuple(s) for length in (1, 2, 3) for s in itertools.combinations(range(3), length)]


def subset_name(indices):
    return "+".join(STREAMS[i] for i in indices)


class Calibrator:
    def __init__(self, reference=None, thresholds=None):
        self.reference = reference
        self.thresholds = thresholds or {}

    def percentile(self, errors):
        if not np.isfinite(errors).all():
            raise ValueError("Non-finite anomaly errors")
        errors = np.asarray(errors)
        return np.column_stack([np.searchsorted(self.reference[:, i], errors[:, i], side="right")/(len(self.reference)+1)
                                for i in range(errors.shape[1])])

    def fit(self, reference, calibration, q_low, q_high):
        if len(reference) < 2 or len(calibration) < 2:
            raise ValueError("Need distinct score-reference and threshold-calibration windows")
        self.reference = np.sort(np.asarray(reference), axis=0)
        ranked = self.percentile(calibration)
        for indices in SUBSETS:
            values = ranked[:, indices].mean(1)
            self.thresholds[subset_name(indices)] = [float(np.quantile(values, q_low, method="higher")),
                                                     float(np.quantile(values, q_high, method="higher"))]
        return self

    def predict(self, errors, indices=(0, 1, 2)):
        ranked = self.percentile(errors)
        fused = ranked[:, indices].mean(1)
        low, high = self.thresholds[subset_name(indices)]
        # Strict > is deliberate; high ECDF ties may reduce attainable alert rates.
        decision = np.where(fused > high, "Suspicious", np.where(fused > low, "Uncertain", "Trusted"))
        return fused, decision, ranked

    def to_dict(self):
        return dict(reference=self.reference.tolist(), thresholds=self.thresholds,
                    note="Empirical ranks, not probabilities. Equal fusion weights; strict > thresholds.")

    @classmethod
    def from_dict(cls, obj):
        return cls(np.asarray(obj["reference"]), obj["thresholds"])
