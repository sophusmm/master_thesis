import numpy as np
import cvxpy as cp
from scipy import optimize
from scipy.optimize import minimize
from codelib.portfolio_optimization import risk_metrics as rm
    



def calculate_risk_parity(cov_mat: np.ndarray) -> np.ndarray:  

    """
    Bruder & Roncalli (2012) method
    """
    
    num_assets = cov_mat.shape[0]
    
    w = cp.Variable(num_assets)

    b = 1.0 / num_assets
    c = 0.1

    constraints = [w >= 0.0001,
                   cp.sum(b * cp.log(w)) >= c]

    prob = cp.Problem(cp.Minimize(0.5*cp.quad_form(w, cov_mat)), constraints=constraints)

    prob.solve()

    return w.value / w.value.sum()


    
def calculate_minimum_variance(cov_mat: np.ndarray, max_weight) -> np.ndarray: 
    """
    Calculates the minimum-variance portfolio with a no-shorting constraint and an optinal max_weight constraint.
    max_weight should be expressed as a number [0, 1]
    """

    
    num_assets = cov_mat.shape[0]
    
    # optimization variable
    w = cp.Variable(num_assets)

    # define constraints 
    constraints = [cp.sum(w)==1.0, w>=0]
    constraints.append(w <= max_weight)

    # define problem 
    prob = cp.Problem(cp.Minimize(w @ cov_mat @ w), constraints=constraints)

    # solve problem 
    prob.solve()
    if prob.status == 'optimal': 
        return w.value
    else: 
        prob.solve(solver="SCS")
        return w.value


def calculate_tangency_portfolio(mean, cov_mat, risk_free_rate, max_weight) -> np.ndarray:
    """
    Calculates the tangency portfolio (maximum Sharpe ratio) with a no-shorting constraint
    and an optional max_weight constraint.
    max_weight should be expressed as a number [0, 1]
    Confirmed that function gives same SR as max SR from Fortitude frontier
    """
    
    num_assets = cov_mat.shape[0]
    
    # Use a different approach: maximize return-to-risk ratio using a change of variables
    # The idea is to use y = w/k where k is a scalar
    y = cp.Variable(num_assets)
    k = cp.Variable(1, pos=True)
    
    # Define objective: maximize (y @ mean - k * risk_free_rate)
    # This is equivalent to maximizing the Sharpe ratio
    obj = cp.Maximize(y @ mean - k * risk_free_rate)
    
    # Define constraints
    constraints = [
        cp.sum(y) == k,  # Equivalent to sum(w) = 1
        y >= 0,          # No shorting
        y <= k * max_weight,  # Max weight constraint
        cp.quad_form(y, cov_mat) <= 1  # Standardized risk constraint
    ]
    
    # Define and solve the problem
    prob = cp.Problem(obj, constraints)
    try:
        prob.solve()
    except:
        prob.solve(solver="SCS")
    
    # Check if the problem was solved successfully
    if prob.status in ["optimal", "optimal_inaccurate"]:
        # Convert back to weights
        weights = y.value / k.value
        return weights
    else:
        print(f"Problem status: {prob.status}")
        return None


def calculate_cc_ratio(weights: np.ndarray, cov_mat: np.ndarray):

    """
    Calculates the diversification ratio of Chouefaty and Coignard (2008)

    .. math::

        \\begin{equation}
            \\text{GLR}(w, \\Sigma) = \\frac{\\sum_{i=1}^N w_i \\sigma_i}{\\sqrt{w^{\\top} \\Sigma w}}
        \\end{equation}

    Parameters
    ----------
    weights:
        Portfolio weights.
    cov_mat:
        Covariance matrix.
        
    Returns
    -------
    float
        Diversification ratio.
    """

    port_std = rm.calculate_portfolio_std(weights=weights, cov_mat=cov_mat)

    vol_vec = np.sqrt(np.diag(cov_mat))
    avg_std = np.inner(weights, vol_vec)

    return avg_std / port_std

def calculate_most_diversified_portfolio(cov_mat: np.ndarray, max_weight=1, init_weights=None) -> np.ndarray:

    """
    Implmentation of Chouefaty and Coignard most diversified portfolio
    Finds the Maximum Diversification portfolio subject to sum(weights) = 1, weights bounds = [0, max_weight)
    """
    # define intial values
    n = cov_mat.shape[0]
    if init_weights is None:
        init_weights = np.repeat(1.0 / n, n)
    
    # define sum to one constraint
    eq_constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    
    # perform optimization
    res = optimize.minimize(lambda x: -calculate_cc_ratio(x, cov_mat) * 100 * 100, init_weights,
                            constraints=[eq_constraint,], bounds=[(0, max_weight)]*n)
    
    return res.x



def calculate_minimum_variance_norm(cov_mat: np.ndarray, delta: float) -> np.ndarray: 

    """
    Need look through. Write descr.
    """
    
    num_assets = cov_mat.shape[0]
    
    # optimization variable
    w = cp.Variable(num_assets)

    # define constraints 
    constraints = [cp.sum(w)==1.0, w>=0,
                   cp.quad_form(w, np.eye(num_assets)) <= delta]

    # define problem 
    prob = cp.Problem(cp.Minimize(w @ cov_mat @ w), constraints=constraints)
    
    # solve problem 
    try:  
        prob.solve()
    except: 
        prob.solve(solver="SCS")
    
    return w.value