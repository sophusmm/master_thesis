import numpy as np

def compute_mve(data):
    """
    Computes the minimum volume ellipsoid for a given time series.
    The location and scatter parameters that define the ellipsoid are
    multivariate high-breakdown estimators of location and scatter.
    
    Parameters:
    data (numpy.ndarray): Input data matrix, where each row is an observation
    
    Returns:
    tuple: (MVE_Location, MVE_Dispersion) - location vector and dispersion matrix
    """
    num_observations = data.shape[0]
    ones = np.ones(num_observations)
    m = np.mean(data, axis=0)
    S = np.cov(data, rowvar=False)
    
    det_S_new = 0
    w = np.ones(num_observations) / num_observations
    keep_loop = True
    
    while keep_loop:
        mahalanobis = np.zeros(num_observations)
        
        for t in range(num_observations):
            x_t = data[t, :]
            mahalanobis[t] = (x_t - m).T @ np.linalg.inv(S) @ (x_t - m)
        
        update = np.where(mahalanobis > 1)[0]
        w[update] = w[update] * mahalanobis[update]
        
        m = (data.T @ w) / np.sum(w)
        
        # Compute centered data matrix
        centered_data = data - np.outer(ones, m)
        
        # Compute weighted covariance matrix
        S = (centered_data.T @ np.diag(w) @ centered_data)
        
        det_S_old = det_S_new
        det_S_new = np.linalg.det(S)
        
        # Check convergence criterion
        keep_loop = (det_S_old / det_S_new < 0.99999) if det_S_old != 0 else True
    
    MVE_Location = m
    MVE_Dispersion = S
    
    return MVE_Location, MVE_Dispersion



def reject_outlier(sample, index=None):
    """
    Finds the "worst" outlier in a time series.
    
    Parameters:
    sample (numpy.ndarray): Input data matrix, where each row is an observation
    index (any): Not used in the function but kept for compatibility with original
    
    Returns:
    int: Index of the rejected (worst) outlier
    """
    # Get number of observations
    T = sample.shape[0]
    
    # Compute mean vector
    m = np.mean(sample, axis=0)
    
    # Center the data
    U = sample - np.outer(np.ones(T), m)
    
    # Compute leverage values (diagonal elements of the hat matrix)
    Lambda = np.diag(U @ np.linalg.inv(U.T @ U) @ U.T)
    
    # Find the index of the maximum leverage value
    rejected = np.argmax(Lambda)
    
    return rejected