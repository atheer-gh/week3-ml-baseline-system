# Model Card

## Unit of Analysis
* Individual Customer

## Target Column
* churn

## Positive Class (Binary)
* 1

## Decision Enabled
* Send a retention offer to the customer

## One Constraint
* Marketing budget limit
## Evaluation Plan
- **Baseline Model**: We use a `DummyClassifier` that always predicts the most frequent class (most_frequent strategy).
- **Split Strategy**: Data is split into 80% training and 20% holdout sets using stratified sampling to maintain class balance.
- **Primary Metrics**: Accuracy and F1-score are used to evaluate the baseline's performance.