import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def my_map(challenges):
    n_samples = challenges.shape[0]
    feature_vectors = np.zeros((n_samples, 64))
    
    for i in range(32):
        feature_vectors[:, 2*i] = 1 - challenges[:, i]
        feature_vectors[:, 2*i + 1] = challenges[:, i]
    
    return feature_vectors

def my_fit(challenges, response0, response1):
    # Map the challenges to new feature vectors
    mapped_features = my_map(challenges)

    # Train logistic regression models
    model0 = LogisticRegression(max_iter=1000)
    model0.fit(mapped_features, response0)

    model1 = LogisticRegression(max_iter=1000)
    model1.fit(mapped_features, response1)

    # Extract weights and biases
    W0, b0 = model0.coef_[0], model0.intercept_[0]
    W1, b1 = model1.coef_[0], model1.intercept_[0]

    return W0, b0, W1, b1

# # Load the training data (example format)
# train_data = np.loadtxt('/content/public_trn.txt')
# test_data = np.loadtxt('/content/public_tst.txt')

# # Separate the training data
# Z_trn = train_data
# Z_tst = test_data

# # Fit the model using training data
# W0, b0, W1, b1 = my_fit(Z_trn[:, :-2], Z_trn[:, -2], Z_trn[:, -1])

# # Map the test challenges to feature vectors
# feat = my_map(Z_tst[:, :-2])

# # Predict the responses for the test challenges
# predicted_response0 = (feat @ W0 + b0 > 0).astype(int)
# predicted_response1 = (feat @ W1 + b1 > 0).astype(int)

# actual_response0 = test_data[:, -2]
# actual_response1 = test_data[:, -1]

# # Calculate and print the accuracy for each response
# accuracy0 = accuracy_score(actual_response0, predicted_response0)
# accuracy1 = accuracy_score(actual_response1, predicted_response1)

# print(f'Accuracy for Response 0: {accuracy0:.4f}')
# print(f'Accuracy for Response 1: {accuracy1:.4f}')
