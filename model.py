"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    # TODO: Replace every NaN with that column's nan-aware mean...
    X = np.asarray(X, dtype=float).copy()

    means = np.nanmean(X, axis=0)
    means = np.nan_to_num(means, nan=0.0)

    for j in range(X.shape[1]):
        nan_mask = np.isnan(X[:,j])
        X[nan_mask, j] = means[j]

    return X

# Step 2 - compute_iqr_bounds
import numpy as np
def compute_iqr_bounds(X, k=1.5):
    # TODO: Compute per-column lower/upper clip bounds using the IQR rule.
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)

    iqr = q3 - q1

    lower = q1 - k * iqr
    upper = q3 + k * iqr

    return lower, upper

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    # TODO: Clip every entry of a feature matrix to per-column lower/upper bounds.
    return np.clip(X, lower,upper)

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    # TODO: Form a derived ratio feature from two 1-D arrays with safe division.
    numerator = np.array(numerator)
    denominator = np.array(denominator)

    N = numerator / (denominator + eps)

    return N

# Step 5 - append_column
import numpy as np
def append_column(X, col):
    # TODO: Horizontally append one 1-D feature column onto a design matrix.
    return np.column_stack((X, col))

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    # TODO: Convert a 1-D array of categorical labels into a dense binary one-hot matrix.
    labels = np.asarray(labels)

    classes, indices = np.unique(labels, return_inverse=True)

    n = len(labels)
    c = len(classes)

    result = np.zeros((n,c), dtype=float)
    result[np.arange(n), indices] = 1.0

    return result

# Step 7 - fit_standardizer
import numpy as np
def fit_standardizer(X):
    # TODO: Compute per-column mean and std used to standardize features...
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    std[std==0] = 1.0
    
    return mean,std

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    # TODO: Return the scaled matrix (X - mean) / std via broadcasting.
    scaled_matrix = (X - mean) / std

    return scaled_matrix

# Step 9 - add_bias_column
import numpy as np
def add_bias_column(X):
    # TODO: Prepend a column of ones to a 2-D feature matrix X...
    N = X.shape[0]

    ones = np.ones((N, 1))

    return np.column_stack((ones, X))

# Step 10 - make_shuffled_indices
import numpy as np

def make_shuffled_indices(n_samples, seed):
    indices = np.arange(n_samples)

    rng = np.random.default_rng(seed)

    shuffled_indices = rng.permutation(indices)

    return shuffled_indices

# Step 11 - partition_indices
import numpy as np
def partition_indices(indices, train_ratio, val_ratio):
    # TODO: Split a shuffled index array into train, validation, and test index arrays.
    N = len(indices)
    train_end = int(N * train_ratio)
    val_end = train_end + int(N * val_ratio)

    train = indices[:train_end]
    validation = indices[train_end:val_end]
    test = indices[val_end:]


    return train,validation,test

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    # TODO: Select the rows of X and y at the given indices.
    X_sub = X[indices]
    y_sub = y[indices]
    return X_sub, y_sub

# Step 13 - ols_fit
def ols_fit(X, y):
    # TODO: return the ordinary-least-squares weight vector for a linear model.
    A = X.T @ X
    b = X.T @ y

    theta = np.linalg.solve(A, b)

    return theta

# Step 14 - ols_predict
def ols_predict(X, theta):
    # TODO: Predict continuous targets with a fitted linear model.
    predictions = X @ theta
    return predictions

# Step 15 - mean_absolute_error
import numpy as np
def mean_absolute_error(y_true, y_pred):
    # TODO: return the mean absolute error between targets and predictions
    MAE = np.mean(np.abs(y_true - y_pred))
    return MAE

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    # TODO: return the root mean squared error as a Python float

    error = y_true - y_pred
    mse = np.mean(error ** 2)
    rmse = np.sqrt(mse)

    return rmse

# Step 17 - r_squared
import numpy as np

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)

    mean_y = np.mean(y_true)
    ss_tot = np.sum((y_true - mean_y) ** 2)

    if ss_tot == 0:
        return 0.0

    return float(1 - (ss_res / ss_tot))

# Step 18 - residual_summary
import numpy as np

def residual_summary(y_true, y_pred):
    r = y_true - y_pred

    return {
        'mean': float(np.mean(r)),
        'std': float(np.std(r)),
        'median_abs': float(np.median(np.abs(r)))
    }

# Step 19 - prepare_cleaned_features
import numpy as np


def prepare_cleaned_features(X, iqr_k=1.5):
    X_clean = np.asarray(X, dtype=float).copy()

    # 1. Impute NaNs with column means
    for i in range(X_clean.shape[1]):
        column = X_clean[:, i]

        mean = np.nanmean(column)

        # If the entire column is NaN, use 0
        if np.isnan(mean):
            mean = 0.0

        column[np.isnan(column)] = mean
        X_clean[:, i] = column

    # 2. Compute IQR bounds AFTER imputation
    q1 = np.percentile(X_clean, 25, axis=0)
    q3 = np.percentile(X_clean, 75, axis=0)

    iqr = q3 - q1

    lower = q1 - iqr_k * iqr
    upper = q3 + iqr_k * iqr

    # 3. Clip outliers
    X_clean = np.clip(X_clean, lower, upper)

    return X_clean

# Step 20 - assemble_feature_matrix
import numpy as np
def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    # TODO: build an extended feature matrix by appending a derived ratio...
    numerator = X_num[:, ratio_num_idx]
    denominator = X_num[:, ratio_den_idx]

    ratio = make_ratio_feature(numerator, denominator)

    X_extended = append_column(X_num, ratio)

    if cat_labels is not None:
        cat_features = one_hot_encode(cat_labels)
        X_extended = np.hstack((X_extended, cat_features))

    return X_extended

# Step 21 - make_train_val_test
import numpy as np

def make_train_val_test(X, y, train_ratio, val_ratio, seed):
    N = X.shape[0]

    rng = np.random.RandomState(seed)

    indices = np.arange(N)
    rng.shuffle(indices)

    train_end = int(N * train_ratio)
    val_end = int(N * (train_ratio + val_ratio))

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_val = X[val_indices]
    y_val = y[val_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

# Step 22 - standardize_and_add_bias
import numpy as np

def standardize_and_add_bias(splits):
    X_train = splits["X_train"]
    X_val = splits["X_val"]
    X_test = splits["X_test"]

    # Fit statistics ONLY on training features
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    # Avoid division by zero for constant features
    std = np.where(std == 0, 1.0, std)

    # Standardize all feature splits using training statistics
    X_train_std = (X_train - mean) / std
    X_val_std = (X_val - mean) / std
    X_test_std = (X_test - mean) / std

    # Prepend bias column
    train_bias = np.ones((X_train_std.shape[0], 1))
    val_bias = np.ones((X_val_std.shape[0], 1))
    test_bias = np.ones((X_test_std.shape[0], 1))

    X_train_std = np.hstack((train_bias, X_train_std))
    X_val_std = np.hstack((val_bias, X_val_std))
    X_test_std = np.hstack((test_bias, X_test_std))

    std_splits = {
        "X_train": X_train_std,
        "y_train": splits["y_train"],
        "X_val": X_val_std,
        "y_val": splits["y_val"],
        "X_test": X_test_std,
        "y_test": splits["y_test"],
    }

    return std_splits, mean, std

# Step 23 - evaluate_predictions
def evaluate_predictions(y_true, y_pred):
    # TODO: Bundle MAE, RMSE, R^2, and residual summary into one metrics dict.

    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred),
        'r2': r_squared(y_true, y_pred),
        'residual_summary': residual_summary(y_true, y_pred)
    }

    return metrics

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

