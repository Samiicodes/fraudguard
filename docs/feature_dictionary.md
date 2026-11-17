# Feature dictionary

Every feature is computed from the columns of `creditcard.csv`. No external data is joined. The code is in `src/fraudguard/feature_engineering.py`.

## Input columns

The raw file is renamed to snake_case on load. The API accepts the same names.

| Column | Source column | Type | Description |
| --- | --- | --- | --- |
| `time_seconds` | `Time` | float, seconds | Seconds between this transaction and the first transaction in the file. Runs from 0 to 172,792, about 48 hours. |
| `v1` to `v28` | `V1` to `V28` | float | Principal components of transaction attributes that the data owner did not release. Centred near 0. Their original meaning is not public. |
| `amount` | `Amount` | float | Transaction amount, not transformed by the data owner. The source does not state the currency. |
| `is_fraud` | `Class` | int, 0 or 1 | Target. 1 marks a transaction labelled as fraud. Not available at prediction time. |

## Model features

The model sees 32 features: the 28 components and 4 engineered features.

| Feature | Definition | Reason |
| --- | --- | --- |
| `v1` to `v28` | Unchanged | The main signal in the dataset. |
| `amount_log` | `ln(1 + amount)` | Amounts are heavily skewed: the median is 22 and the maximum is 25,691. The log stops a few large purchases from dominating the linear baseline. Zero amounts map to 0. |
| `amount_bucket` | Ordinal bucket 0 to 6 for amounts below 1, 1 to 10, 10 to 50, 50 to 100, 100 to 500, 500 to 1,000, and 1,000 or more. An amount equal to an edge falls into the bucket above it. | Fraud is over-represented at very small amounts, a sign of card testing: 13.8% of fraud cases are below 1, against 5.9% of legitimate transactions. A bucket lets the linear baseline treat that range separately. Edges are set in `config/config.yaml`. |
| `hour_sin` | `sin(2 * pi * h / 24)` | Fraud is more common at night: in hours 1 to 6 the fraud rate is 0.62%, against 0.14% for the rest of the day. |
| `hour_cos` | `cos(2 * pi * h / 24)` | Paired with `hour_sin` so that 23:00 and 00:00 end up close together. |

Here `h = (time_seconds mod 86,400) / 3,600`, the hour of day as a decimal.

After these features are built, two fitted steps follow. Both are fitted on the training split only and then applied unchanged to validation, test, and live requests.

| Step | Fitted statistic | Effect |
| --- | --- | --- |
| Median imputation | Median of each feature in the training split | Replaces missing values. The public file has none, but the API path stays safe if a value is ever missing upstream. |
| Standard scaling | Mean and standard deviation of each feature in the training split | Puts features on one scale for logistic regression. Tree models are unaffected. Explanation values returned by the API are in these standardised units. |

## Columns that are deliberately not used

| Column | Reason |
| --- | --- |
| `time_seconds` as a raw value | It is a counter over the collection period. A model that learned from it would describe those two days, not future transactions. Only the hour of day is kept. |
| Day index | The file covers two days, too few to learn anything about weekdays. |

## Assumption about time

The hour features assume the file starts at midnight. The source does not say so, but the data supports it: hours 1 to 6 hold 7.1% of all transactions, the night-time dip expected if `Time = 0` is midnight. Clients of the API must use the same convention and send `time_seconds` counted from a midnight.

## Features that cannot be built from this dataset

| Feature | Missing information |
| --- | --- |
| Transactions per card in the last hour or day | No card or customer identifier |
| Amount compared with the card's usual spending | No card identifier |
| Time since the previous transaction on the card | No card identifier |
| Merchant, country, device, or channel risk | No such columns |

In production fraud systems these behavioural features are usually among the strongest signals. Their absence is the main limitation of this dataset, and none of them is approximated here.

## Drift monitoring

The API compares `amount` and `v1` to `v28` in recent requests with their training distributions. `time_seconds` is excluded because a counter always looks drifted. See `docs/architecture.md` for the method.
