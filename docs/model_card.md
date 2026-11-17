# Model card: fraudguard champion

Figures on this page come from the run published in `docs/evaluation_report.md`. Retraining changes the registry version and may change the champion, so check that report for the current numbers.

## Model details

| Item | Value |
| --- | --- |
| Registered model | `fraudguard`, alias `champion`, version 1 |
| Algorithm | XGBoost gradient-boosted trees: 400 trees, depth 5, learning rate 0.05, row and column subsampling 0.8 |
| Imbalance handling | `scale_pos_weight` of about 598, the ratio of legitimate to fraud rows in the training split |
| Inputs | `time_seconds`, `v1` to `v28`, and `amount`, as described in `docs/feature_dictionary.md` |
| Preprocessing | Engineered amount and hour features, median imputation, standard scaling, all fitted on the training split |
| Output | A score between 0 and 1, and a label: fraud when the score is at least the decision threshold of 0.120 |
| Training data | `creditcard.csv`, SHA-256 `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89` |
| Code licence | MIT |

The score is not a calibrated probability. Class weighting shifts scores upwards: the unweighted control model reaches a similar operating point at a threshold of 0.010, against 0.120 for the champion. Decisions should use the label, not the raw score.

## Intended use

- **Intended:** showing an end-to-end fraud scoring workflow, from validated data to a monitored API, on a public dataset. In a real system, a model like this would rank card transactions for manual review or for a step-up check such as a one-time code.
- **Users:** engineers and reviewers studying the pipeline.
- **Out of scope:**
  - Decisions about real customers. The model was trained on two days of 2013 data and has not been validated on anything newer.
  - Any data other than the anonymised components of this dataset. The components come from a transformation that is not public, so the model cannot score raw transaction data from another source.
  - Fully automatic declines or account closures without human review and a way for customers to appeal.

## Data

| Split | Transactions | Fraud cases | Use |
| --- | --- | --- | --- |
| Train | 170,235 | 284 | Fitting preprocessing and models |
| Validation | 56,745 | 94 | Model selection and the decision threshold |
| Test | 56,746 | 95 | One final evaluation of the champion |

The split is random and stratified by label after removing 1,081 duplicate rows. It is not split by time, so the test result says nothing about how the model holds up on later transactions.

## Metrics

- **PR-AUC** (average precision) on the validation split selects the model, because it measures how well fraud is ranked above legitimate transactions across all thresholds.
- **Precision, recall, false-positive rate, and false-negative rate** at the decision threshold.
- **Expected cost per 10,000 transactions**, with a missed fraud costing 150 and a false alert costing 5.
- **Brier score and a calibration curve.**
- **95% bootstrap intervals** from 500 resamples of the test split.

Accuracy is not used. On this data a model that flags nothing scores 99.83%.

## Results on the test split

| Metric | Value | 95% interval |
| --- | --- | --- |
| PR-AUC | 0.866 | 0.800 to 0.929 |
| ROC-AUC | 0.978 | 0.953 to 0.994 |
| Precision at threshold | 0.853 | 0.780 to 0.922 |
| Recall at threshold | 0.853 | 0.783 to 0.922 |
| False-positive rate | 0.025% | |
| Cost per 10,000 transactions | 382 | 199 to 568 |

With no model, the cost is 2,511 per 10,000 transactions. On the validation split the same model reached PR-AUC 0.803, precision 0.712, and recall 0.787. The test figures are higher, but the validation PR-AUC lies inside the test interval, so the difference is within sampling noise. With fewer than 100 fraud cases per split, any single number here is uncertain.

The decision threshold was chosen on the validation split: the lowest-cost threshold whose false-positive rate stays within a review budget of 0.1% of legitimate transactions. The full threshold analysis is in the evaluation report.

## Explanations

SHAP values on validation data rank `v14`, `v4`, `v12`, `v10`, and `v3` as the most influential features, and permutation importance broadly agrees. The hour of day is the most useful engineered feature. Because `v1` to `v28` are anonymised, an explanation can name the components behind a score but cannot translate them into reasons a customer or analyst would recognise.

## Ethical considerations

- **False positives harm people.** A wrongly declined payment can leave a customer stranded at a till, block a time-critical purchase, or make them feel accused. At the chosen threshold about 1 legitimate transaction in 4,000 is flagged; in a large card portfolio that is many people every day.
- **Unusual is not the same as fraudulent.** The model learns what typical transactions look like. Customers whose behaviour differs from the majority, such as people who travel, have irregular incomes, or rarely use their card, are more likely to be flagged. These groups can overlap with protected characteristics.
- **Fairness cannot be measured here.** The dataset has no demographic, geographic, or account attributes, so false-positive rates cannot be compared across groups. Proxies may still be present inside the anonymised components. Before any real deployment, error rates must be compared across relevant groups using properly governed data, and disparities must be addressed.
- **Keep a human in the loop.** Flags should lead to review or a step-up check, not a final decision, and customers need a clear way to contest a decision.
- **Personal data.** The public dataset is anonymised, but production inputs would be personal data. The API's prediction log stores full inputs and would need access control, a retention limit, and a lawful basis under rules such as the GDPR.

## Caveats and recommendations

- Validate on later, out-of-time data before relying on any of these numbers.
- Re-derive the threshold whenever the cost assumptions or the review capacity change.
- Retrain when the drift monitor raises alerts, and track precision and recall against fraud labels as they arrive.
- The champion beat the unweighted control by 0.0015 in validation PR-AUC, which is well within noise. A larger validation set, or cross-validation, is needed before preferring either.
- Calibrate the scores, for example with isotonic regression on a separate calibration split, before any downstream system treats them as probabilities.
