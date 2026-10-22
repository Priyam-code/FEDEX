# Sources and attribution

The package implements the team's proposed three-stream design; it is not an official implementation of the related-work papers. Any comparisons with published systems require faithful reimplementation and comparable data/splits, not a comparison of unrelated reported accuracy numbers.

## Dataset/schema

- VeReMi NextGen official repository: https://github.com/VeReMi-dataset/VeReMi-NextGen
- Official message/schema documentation: https://github.com/VeReMi-dataset/VeReMi-NextGen/blob/main/Documentation/README.md
- Dataset release used for the compatibility pilot: https://zenodo.org/records/19665762
- Dataset citation: Hermann, A., Remmers, J. N., Eisermann, D., Erb, B., and Kargl, F. “VeReMi NextGen: A Dataset for Evaluating Misbehavior Detection Systems in VANETs.” IEEE Vehicular Networking Conference, 2026. Confirm the final publisher bibliographic entry before submission.

The official baseline archive examined here omits the attacker field. The attack archive contains explicit labels. Positions can be comma-separated strings rather than JSON arrays. Timing values are in nanoseconds. The importer was exercised against those released bytes rather than only an invented schema.

## Algorithms and software

- McMahan et al., “Communication-Efficient Learning of Deep Networks from Decentralized Data,” AISTATS 2017: https://proceedings.mlr.press/v54/mcmahan17a.html — FedAvg motivation and aggregation.
- PyTorch GRU: https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html
- Scikit-learn average precision: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
- Scikit-learn Isolation Forest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
- Scikit-learn one-class SVM: https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html

The neural model sizes, feature selection, calibration protocol and equal-weight fusion are explicit prototype design choices. They are not attributed as inventions of the above libraries or claimed as verbatim algorithms from the team's related-work papers.

## Related-work links supplied by the team

These links document the literature background; this package does not reproduce or verify every prior prose claim about their future-work sections.

- HCA-MDS: https://irep.ntu.ac.uk/id/eprint/38326/1/1216848_Al-Hadhrami.pdf
- Physics Joins AI: https://ans.unibs.it/docs/assets/res/plexeMDS.pdf
- PAMPOS: https://arxiv.org/abs/2605.06833
- Congestion-context flooding detection: https://doi.org/10.1016/j.vehcom.2026.101045
- Federated VAE–GMM: https://link.springer.com/article/10.1007/s10207-025-01000-8

Use the papers' own bibliographic records when preparing the manuscript. Do not call Trusted/Uncertain/Suspicious decisions a solution to attack-type classification: those are different tasks.
