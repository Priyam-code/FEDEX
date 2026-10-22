# Explaining the implementation simply

## What problem are we solving?

Vehicles exchange messages about their position, speed and movement. Some messages may be false, delayed or repeated. We are testing whether combining three kinds of evidence can identify suspicious behaviour with fewer mistaken alerts.

## What does each model do?

The movement GRU looks at a sender's recent messages and predicts its next movement state. A large prediction error is evidence that the new message does not fit the recent pattern.

The timing GRU autoencoder learns to reconstruct normal patterns of message gaps, delays and repetition. A large reconstruction error indicates unusual communication timing.

The context MLP autoencoder learns normal combinations of neighbour count, speed differences, distances and optional road-map consistency. It asks whether the message makes sense relative to the observed surroundings.

## Why train only on normal data?

We want to learn normal behaviour rather than memorize a list of attacks. New attacks may cause a deviation from that normal pattern. This does not guarantee that every unseen attack will be detected; a believable attack may still look normal.

## What is federated learning here?

We divide training windows among virtual roadside units. Each client trains locally and the server averages its model parameters with the others. Our code simulates this on one computer. It does not yet deploy separate machines or implement secure aggregation.

## Which ML algorithms are actually used?

GRU prediction, GRU reconstruction and MLP reconstruction are the neural models. Adam updates their weights. MSE measures prediction/reconstruction errors. FedAvg combines client parameters. Empirical ranks and quantiles calibrate decisions. Isolation Forest and one-class SVM are comparison baselines. Precision/recall/F1 and AUC are evaluation measures, not additional trained models.

## How does the congestion idea help?

Traffic conditions can change what normal looks like. We therefore include observed neighbour count and neighbour speeds instead of judging every timing anomaly in isolation. This is an adaptation of a context-aware principle, not a claim that we reproduced a particular congestion paper's exact classifier.

## Why three decisions?

Some cases fall between clearly normal-looking and strongly abnormal-looking. We call that middle score range Uncertain. Evaluation shows what happens if we count those cases as alerts and what happens if we do not. We are not pretending Uncertain is a ground-truth attack category.

## What can we claim as progress?

We can show the code structure, passing tests, an end-to-end demo, saved models, actual CSV reports and a small real-data pilot. These demonstrate implementation progress. We cannot say the full idea is validated or outperforms other papers until adequate real-data experiments support that statement.

## Why might a simple baseline win?

A neural network is not automatically better. The feature quality, amount of benign data, training budget, threshold calibration and distribution shift all matter. The supplied reports keep poor or negative results. That helps us identify which component needs work.

## What is still pending?

Full multi-scenario/attack experiments, rigorous map validation, congestion-specific controlled evaluation, longer training and sensitivity analysis, independent-run uncertainty estimates, secure aggregation/privacy analysis, and benchmarking on actual target OBU hardware.
