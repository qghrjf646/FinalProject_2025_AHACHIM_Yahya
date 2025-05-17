import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix
import pickle

# Load preprocessed data
# User-item interaction matrix for ALS
als_matrix = pd.read_csv('als_train_matrix.csv', index_col=0)
# Feature matrix for LogisticRegression (content-based)
X_content = pd.read_csv('content_features_X.csv', index_col=0)
y_content = pd.read_csv('content_features_y.csv', index_col=0).values.ravel()

# --- Collaborative Filtering: ALS ---
try:
    from implicit.als import AlternatingLeastSquares
    als_data = csr_matrix(als_matrix.values)
    als_model = AlternatingLeastSquares(factors=50, regularization=0.1, iterations=20)
    als_model.fit(als_data)
    with open('als_model.pkl', 'wb') as f:
        pickle.dump(als_model, f)
    print('ALS model trained.')
except ImportError:
    print('implicit library not installed, skipping ALS training.')

# --- Content-Based Filtering: Logistic Regression ---
lr = LogisticRegression(max_iter=1000)
lr.fit(X_content, y_content)
with open('logreg_model.pkl', 'wb') as f:
    pickle.dump(lr, f)
print('LogisticRegression model trained.')

print('Models trained and saved.')
