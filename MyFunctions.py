import numpy as np
import pandas as pd
from scipy import stats
from scipy import interpolate
from scipy.linalg import cholesky
from typing import Union, List

def CMA_separation(X, p):
    """
    Separation step of Copula-Marginal Algorithm (CMA)
    Python implementation of Meucci A., "New Breed of Copulas for Risk and Portfolio Management", Risk, September 2011
    
    Parameters:
    X (numpy.ndarray): Input data matrix, size J x N
    p (numpy.ndarray): Probability vector, size J
    
    Returns:
    tuple: (x, u, U) where:
        x (numpy.ndarray): Sorted input data, size J x N
        u (numpy.ndarray): Rescaled CDF values, size J x N
        U (numpy.ndarray): Grade matrix, size J x N
    """
    # Preprocess variables
    J, N = X.shape
    l = J / (J + 1)
    p = np.maximum(p, 1/J * 10**(-8))
    p = p / np.sum(p)
    u = np.zeros_like(X)
    U = np.zeros_like(X)
    
    # Core algorithm
    x = np.zeros_like(X)
    Indx = np.zeros_like(X, dtype=int)
    
    for n in range(N):  # For each marginal...
        # Sort
        x[:, n] = np.sort(X[:, n])
        Indx[:, n] = np.argsort(X[:, n])
        
        I = Indx[:, n]
        # Compute CDF
        cum_p = np.cumsum(p[I])
        # Rescale to be <1 at the far right
        u[:, n] = cum_p * l
        
        # Compute ranking of each entry (equivalent to MATLAB's interp1)
        # First create inverse permutation
        inverse_I = np.zeros(J, dtype=int)
        for j in range(J):
            inverse_I[I[j]] = j
            
        # This is equivalent to MATLAB's Rnk = round(interp1(I,1:J,1:J))
        Rnk = inverse_I
        
        # Compute grade
        U[:, n] = cum_p[Rnk] * l
        
    return x, u, U


def CMA_combination(x, u, U):
    """
    Combination step of Copula-Marginal Algorithm (CMA)
    Python implementation of Meucci A., "New Breed of Copulas for Risk and Portfolio Management", Risk, September 2011
    
    Parameters:
    x (numpy.ndarray): Sorted input data, size J x K
    u (numpy.ndarray): Rescaled CDF values, size J x K
    U (numpy.ndarray): Grade matrix, size M x K
    
    Returns:
    numpy.ndarray: Combined data matrix X, size J x K
    """
    J, K = x.shape
    X = np.zeros_like(U)
    
    for k in range(K):
        # Create an interpolation function equivalent to MATLAB's interp1 with 'linear' and 'extrap'
        f = interpolate.interp1d(u[:, k], x[:, k], kind='linear', bounds_error=False, fill_value='extrapolate')
        X[:, k] = f(U[:, k])
    
    return X


def compute_returns_statistics(returns_data, sector_names=None, risk_free=0):
    """
    Compute comprehensive summary statistics for stock returns by sector.
    Assuming weekly log returns as input
    
    Parameters:
    returns_data (numpy.ndarray or pandas.DataFrame): Matrix of returns (observations x sectors)
    sector_names (list, optional): List of sector names. If None, will use generic names
    
    Returns:
    pandas.DataFrame: Summary statistics for each sector
    """
    # Convert numpy array to pandas DataFrame if needed
    if isinstance(returns_data, np.ndarray):
        if sector_names is None:
            sector_names = [f"Sector_{i+1}" for i in range(returns_data.shape[1])]
        returns_data = pd.DataFrame(returns_data, columns=sector_names)
    
    # Create empty DataFrame for statistics
    stats_df = pd.DataFrame(index=sector_names)
    
    # Calculate basic statistics
    stats_df['Mean (%)'] = returns_data.mean() * 100
    stats_df['Median (%)'] = returns_data.median() * 100
    stats_df['Std Dev (%)'] = returns_data.std() * 100
    stats_df['Min (%)'] = returns_data.min() * 100
    stats_df['Max (%)'] = returns_data.max() * 100
    
    # Higher moments
    stats_df['Skewness'] = returns_data.skew()
    stats_df['Excess Kurtosis'] = returns_data.kurtosis()
    
    # VaR at 95%
    stats_df['VaR 95% (%)'] = returns_data.quantile(0.05) * 100
    
    # CVaR at 95% (Expected Shortfall)
    cvar_values = []
    for col in returns_data.columns:
        var_cutoff = np.quantile(returns_data[col], 0.05)
        cvar = returns_data[col][returns_data[col] <= var_cutoff].mean() * 100
        cvar_values.append(cvar)
    stats_df['CVaR 95% (%)'] = cvar_values

    # Annualized statistics (assuming weekly log returns)
    weeks = 52  # Standard assumption for trading days in a year
    stats_df['Ann. Return (%)'] = stats_df['Mean (%)'] * weeks
    stats_df['Ann. Volatility (%)'] = stats_df['Std Dev (%)'] * np.sqrt(weeks)
    stats_df['Ann. Sharpe Ratio'] = (stats_df['Ann. Return (%)'] - risk_free*100) / stats_df['Ann. Volatility (%)']

    # Maximum drawdown
    max_drawdowns = []
    for col in returns_data.columns:
        # Calculate cumulative returns
        cum_returns = returns_data[col].cumsum()
        # Calculate running maximum
        running_max = cum_returns.cummax()
        # Calculate drawdown
        drawdown = np.exp(cum_returns - running_max) - 1
        max_drawdowns.append(drawdown.min() * 100)
    stats_df['Max Drawdown (%)'] = max_drawdowns

    # Jarque-Bera test for normality (p-value)
    jb_pvalues = []
    for col in returns_data.columns:
        jb_stat, jb_pvalue = stats.jarque_bera(returns_data[col].dropna())
        jb_pvalues.append(jb_pvalue)
    stats_df['JB p-value'] = jb_pvalues
    
    
    return stats_df


### Parametric Copula


def simulate_returns_gaussian_copula_t_marginals(U, t_params, n_samples=1000):
    """
    Simulate returns using a Gaussian copula with Student's t marginal distributions
    
    Parameters:
    U (numpy.ndarray): Grades matrix from CMA_separation
    t_params (list of tuples): List of (df, loc, scale) parameters for each stock's t-distribution
    n_samples (int): Number of samples to generate
    
    Returns:
    numpy.ndarray: Simulated returns
    """
    n_vars = U.shape[1]
    
    # Step 1: Convert U (grades) to standard normal variables
    Z = stats.norm.ppf(U)
    
    # Step 2: Compute correlation matrix of Z
    corr_matrix = np.corrcoef(Z, rowvar=False)
    
    # Step 3: Generate samples from multivariate normal with this correlation
    # First, calculate Cholesky decomposition
    try:
        L = cholesky(corr_matrix, lower=True)
    except np.linalg.LinAlgError:
        # If correlation matrix is not positive definite, adjust it
        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-8)  # Ensure positive eigenvalues
        corr_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        # Make sure diagonal elements are 1
        d = np.sqrt(np.diag(corr_matrix))
        corr_matrix = corr_matrix / np.outer(d, d)
        L = cholesky(corr_matrix, lower=True)
    
    # Generate independent standard normal variables
    Z_indep = np.random.standard_normal((n_samples, n_vars))
    
    # Transform to correlated normal variables
    Z_correlated = Z_indep @ L.T
    
    # Step 4: Convert to uniform using normal CDF
    U_simulated = stats.norm.cdf(Z_correlated)
    
    # Step 5: Transform uniform to Student's t for each marginal
    simulated_returns = np.zeros((n_samples, n_vars))
    
    for j in range(n_vars):
        df, loc, scale = t_params[j]
        simulated_returns[:, j] = stats.t.ppf(U_simulated[:, j], df=df, loc=loc, scale=scale)
    
    return simulated_returns

# Example usage:
# Assuming you've fitted t-distributions to each stock and have parameters
# t_params = [(df_1, loc_1, scale_1), (df_2, loc_2, scale_2), ..., (df_30, loc_30, scale_30)]
# where each tuple contains the degrees of freedom, location, and scale parameters

def fit_t_distribution_to_series(data):
    """
    Fit a Student's t distribution to a data series
    
    Parameters:
    data (numpy.ndarray or pandas.Series): Returns data for one stock
    
    Returns:
    tuple: (df, loc, scale) parameters of the fitted t-distribution
    """
    # Use MLE to fit a t-distribution
    params = stats.t.fit(data)
    return params  # (df, loc, scale)

# Fit t-distributions to original returns for each stock
def fit_t_distributions(returns_data):
    """
    Fit t-distributions to each column of returns data
    
    Parameters:
    returns_data (numpy.ndarray or pandas.DataFrame): Matrix of returns
    
    Returns:
    list: List of (df, loc, scale) parameters for each column
    """
    if isinstance(returns_data, pd.DataFrame):
        returns_data = returns_data.values
    
    n_stocks = returns_data.shape[1]
    t_params = []
    
    for j in range(n_stocks):
        params = fit_t_distribution_to_series(returns_data[:, j])
        t_params.append(params)
    
    return t_params

# Full example workflow:
# 1. Fit t-distributions to your original returns
# t_params = fit_t_distributions(log_returns)
# 
# 2. Simulate returns using Gaussian copula with t-marginals
# simulated_returns = simulate_returns_gaussian_copula_t_marginals(U, t_params, n_samples=1000)
# 
# 3. Analyze the simulated returns
# simulated_stats = compute_returns_statistics(simulated_returns)


# PCA function
def calculate_PCA(returns, n_components):
    """
    Extract n principal components from returns data.
    Parameters:
    returns (numpy ndarray or pandas DataFrame)
    n_components (integer): The number of Principal Components to extract
    Returns:
    mean_vector: the mean of each feature
    eig_values: eigenvalues in descending order
    eig_vectors: eigenvectors corresponding to eigenvalues
    PC: principal components
    """
    # 1. Center the data first
    mean_vector = returns.mean()
    returns_centered = returns - mean_vector
    
    # 2. Calculate covariance matrix of centered data
    cov = returns_centered.cov()
    
    # 3. Compute eigenvalues and eigenvectors
    eig_values, eig_vectors = np.linalg.eigh(cov)  # eigh for symmetric matrices
    
    # 4. Sort eigenvalues and eigenvectors in descending order
    idx = eig_values.argsort()[::-1]
    eig_values = eig_values[idx]
    eig_vectors = eig_vectors[:, idx]
    
    # 5. Project centered data onto eigenvectors
    P = returns_centered @ eig_vectors
    
    # 6. Select first n principal components
    PC = P.iloc[:, :n_components]
    
    return mean_vector, eig_values, eig_vectors, PC


def calculate_sortino_ratio(returns, risk_free_rate=0, target_return=0):
    """
    Calculate the Sortino ratio for a series of investment returns.
    
    Parameters:
    returns (numpy.ndarray): Array of periodic investment returns
    risk_free_rate (float): Risk-free rate for the period (default: 0)
    target_return (float): Minimum acceptable return (default: 0)
    
    Returns:
    float: The Sortino ratio
    """
    # Calculate average return
    average_return = np.mean(returns)
    
    # Calculate excess return
    excess_return = average_return - risk_free_rate
    
    # Calculate downside returns (returns below target)
    downside_returns = returns[returns < target_return] - target_return
    
    # If no downside returns, return a large value
    if len(downside_returns) == 0:
        return float('inf')
    
    # Calculate downside deviation
    downside_deviation = np.sqrt(np.mean(np.square(downside_returns)))
    
    # Return Sortino ratio
    return excess_return / downside_deviation


def calculate_starr_ratio(returns, risk_free_rate=0, alpha=0.05):
    """
    Calculate the STARR (Stable Tail Adjusted Return Ratio) for a series of investment returns.
    
    Parameters:
    returns (numpy.ndarray): Array of periodic investment returns
    risk_free_rate (float): Risk-free rate for the period (default: 0)
    alpha (float): The CVaR percentile (default 0.05)
    
    Returns:
    float: The STARR ratio
    """
    # Calculate average return
    average_return = np.mean(returns)
    
    # Calculate excess return
    excess_return = average_return - risk_free_rate
    
    # Calculate VaR at the specified confidence level
    var = np.percentile(returns, alpha)
    
    # Calculate CVaR (Conditional Value at Risk or Expected Shortfall)
    # Mean of returns below VaR threshold
    cvar = -np.mean(returns[returns <= var])
    
    # If CVaR is zero or negative, return a large value or handle appropriately
    if cvar <= 0:
        return float('inf')
    
    # Return STARR ratio (excess return divided by CVaR)
    return excess_return / cvar
