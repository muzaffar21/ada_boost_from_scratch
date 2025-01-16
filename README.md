ADA Boost Implementation from scratch
======================================

This is ada boost implementation from scratch in python.
Just for learning purposes.

Note:- This is not efficient implementation in terms of computations and not for production use.
The purpose is to just understand ada boost from scratch.

Training Process
The AdaBoost algorithm consists of the following steps:

Step 1: Initialize Weights
Each training sample is assigned an equal weight:

python
Copy
Edit
df__['weights'] = 1 / df__.shape[0]
Step 2: Train a Weak Classifier
A decision stump is trained on the weighted dataset:

python
Copy
Edit
pred, dt = self._train_classifier(X, y)
Step 3: Calculate Alpha
The weight of the weak classifier is calculated based on its error:

python
Copy
Edit
alpha = 0.5 * np.log((1 - error) / error + 0.000001)
Step 4: Update Sample Weights
Weights are updated to emphasize misclassified samples:

python
Copy
Edit
df_['updated_weights'] = df_['weights'] * np.exp(alpha * (df_['y'] != df_['y_pred']).apply(lambda x: 1 if x else -1))
Step 5: Normalize Weights
Weights are normalized so that they sum up to 1:

python
Copy
Edit
df_['normalized_weights'] = df_['updated_weights'] / df_['updated_weights'].sum()
Step 6: Resample Dataset
Samples are resampled based on their updated weights to create a new training dataset:

python
Copy
Edit
selected_indices = self._create_new_dataset_indices(df_)
Step 7: Repeat
Repeat Steps 2–6 for the specified number of weak classifiers (num_stumps).

Step 8: Final Prediction
The final prediction is computed as a weighted sum of the weak classifiers' predictions:

python
Copy
Edit
res += alpha * y_pred
return np.sign(res)
