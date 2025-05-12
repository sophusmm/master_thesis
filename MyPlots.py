import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle




def plot_histograms_with_fits(returns_data, t_params, sector_names=None, n_cols=3, figsize=(15, 20)):
    """
    Plot histograms of returns with fitted Student's t distributions
    
    Parameters:
    returns_data (numpy.ndarray or pandas.DataFrame): Matrix of returns
    t_params (list of tuples): List of (df, loc, scale) parameters for each stock's t-distribution
    sector_names (list, optional): List of sector names. If None, will use generic names
    n_cols (int): Number of columns in the plot grid
    figsize (tuple): Figure size (width, height)
    
    Returns:
    matplotlib.figure.Figure: The figure object containing the plots
    """
    if isinstance(returns_data, pd.DataFrame):
        if sector_names is None:
            sector_names = returns_data.columns.tolist()
        returns_data = returns_data.values
    else:
        if sector_names is None:
            sector_names = [f"Sector_{i+1}" for i in range(returns_data.shape[1])]
    
    n_sectors = returns_data.shape[1]
    n_rows = int(np.ceil(n_sectors / n_cols))
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n_rows, n_cols)
    
    for i in range(n_sectors):
        ax = plt.subplot(gs[i // n_cols, i % n_cols])
        
        # Extract data for this sector
        data = returns_data[:, i]
        
        # Plot histogram with density=True for proper scaling
        n, bins, patches = ax.hist(data, bins=30, density=True, alpha=0.7, 
                                 color='skyblue', edgecolor='black')
        
        # Get t-distribution parameters for this sector
        df, loc, scale = t_params[i]
        
        # Create x values for the PDF
        x = np.linspace(min(data), max(data), 1000)
        
        # Plot the PDF
        pdf = stats.t.pdf(x, df=df, loc=loc, scale=scale)
        ax.plot(x, pdf, 'r-', linewidth=2, label=f't-dist (df={df:.2f})')
        
        # Normal distribution for comparison
        norm_mean = np.mean(data)
        norm_std = np.std(data)
        norm_pdf = stats.norm.pdf(x, loc=norm_mean, scale=norm_std)
        ax.plot(x, norm_pdf, 'g--', linewidth=1.5, label='Normal')
        
        # Add vertical line at mean
        #ax.axvline(x=loc, color='k', linestyle='--', alpha=0.5) # removed this
        
        # Set title and labels
        ax.set_title(f"{sector_names[i]}")
        ax.set_xlabel("Return")
        ax.set_ylabel("Density")
        
        # Add legend
        ax.legend(loc='best', fontsize='small')
        
        # Add text with statistics
        stats_text = f"Mean: {loc:.4f}\nStd: {scale:.4f}\nDF: {df:.2f}"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.5, facecolor='white'))
    
    plt.tight_layout()
    return fig




def plot_lag_scatterplots(data, columns=None, lag=1, figsize=(12, 10), titles=None, overall_title=""):
    """
    Create scatter plots of each variable against its lagged values.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset containing the variables to plot
    columns : list, optional
        List of column names to plot. If None, all numeric columns will be used.
    lag : int, default=1
        The number of time periods to lag
    figsize : tuple, default=(12, 10)
        The size of the figure
    titles : list, optional
        List of custom titles for each plot. If None, default titles will be used.
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure containing the scatter plots
    """
    # If columns is None, use all numeric columns
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Calculate number of subplots needed
    n_cols = min(2, len(columns))
    n_rows = int(np.ceil(len(columns) / n_cols))
    
    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Make axes iterable even if there's only one subplot
    if len(columns) == 1:
        axes = np.array([axes])
    
    # Flatten axes array for easy iteration
    axes = axes.flatten()
    
    # Create scatter plots
    for i, col in enumerate(columns):
        if i < len(axes):
            # Create lagged version of the column
            original = data[col].iloc[lag:]
            lagged = data[col].iloc[:-lag].values
            
            # Create scatter plot without regression line
            axes[i].scatter(lagged, original.values, alpha=0.7)
            
            # Calculate correlation coefficient
            corr = np.corrcoef(original, lagged)[0, 1]
            
            # Set custom title if provided, otherwise use default
            if titles and i < len(titles):
                axes[i].set_title(titles[i])
            else:
                axes[i].set_title(f'{col} vs Lag-{lag}')
            
            # Set x-axis label (only lagged variable name)
            axes[i].set_xlabel(r'$return_{t-1}$')
            
            # Remove y-axis label
            axes[i].set_ylabel(r'$return_t$')
            
            # Add correlation value in a light grey box
            # Position the box in the bottom right corner
            textstr = f'Corr: {corr:.3f}'
            props = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)
            axes[i].text(0.05, 0.95, textstr, transform=axes[i].transAxes,
                         verticalalignment='top', bbox=props)
            
            # Add grid
            axes[i].grid(True, linestyle='--', alpha=0.7)
    
    # Hide any unused subplots
    for j in range(len(columns), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(overall_title, fontsize=20, weight="bold")
    plt.tight_layout()
    return fig


def plot_multiple_acf(data, columns=None, lags=50, figsize=(12, 10), titles=None, 
                      overall_title="", alpha=0.05, fill_color='lightgrey'):
    """
    Create autocorrelation function plots for multiple columns in a dataframe.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset containing the variables to plot
    columns : list, optional
        List of column names to plot. If None, all numeric columns will be used.
    lags : int, default=50
        The number of lags to include in the plot
    figsize : tuple, default=(12, 10)
        The size of the figure
    titles : list, optional
        List of custom titles for each plot. If None, default titles will be used.
    alpha : float, default=0.05
        The significance level for the confidence intervals
    fill_color : str, default='lightgrey'
        The color to use for the confidence interval bands
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure containing the ACF plots
    """
    # If columns is None, use all numeric columns
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Calculate number of subplots needed
    n_cols = min(2, len(columns))
    n_rows = int(np.ceil(len(columns) / n_cols))
    
    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Make axes iterable even if there's only one subplot
    if len(columns) == 1:
        axes = np.array([axes])
    
    # Flatten axes array for easy iteration
    axes = axes.flatten()
    
    # Create ACF plots
    for i, col in enumerate(columns):
        if i < len(axes):
            
            ax = pd.plotting.autocorrelation_plot(data.iloc[:, i], ax=axes[i])
            
            # Set custom title if provided, otherwise use default
            if titles and i < len(titles):
                axes[i].set_title(titles[i])
            else:
                axes[i].set_title(f'Autocorrelation: {col}')
            
            # Set axis labels
            axes[i].set_xlabel('Lag')
            axes[i].set_ylabel('Autocorrelation')
            
            # Set axes limits
            axes[i].set_xlim([0, lags])
            bottom, top = axes[i].get_ylim()
            axes[i].set_ylim(bottom + bottom * 0.5, top + top * 0.5)
            
            # Customize confidence interval area
            if fill_color != 'lightgrey':  # If custom color specified
                # Find the confidence interval lines
                for line in axes[i].get_lines():
                    if line.get_linestyle() == '--':
                        # Get y value of confidence line
                        y_val = line.get_ydata()[0]
                        # Fill between confidence lines
                        axes[i].axhspan(-y_val, y_val, alpha=0.2, color=fill_color)
                        # Remove original confidence lines
                        line.set_visible(False)
            
            # Add grid
            axes[i].grid(True, linestyle='--', alpha=0.7)
    
    # Hide any unused subplots
    for j in range(len(columns), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(overall_title, fontsize=20, weight="bold")
    plt.tight_layout()
    return fig


def plot_comparison_histograms(returns1, returns2, labels=["Empirical", "Simulated"], 
                              title="Weekly Portfolio Returns", bins=50, figsize=(16, 8)):
    """
    Plot two histograms side by side with statistical annotations and consistent axes.
    
    Parameters:
    -----------
    returns1 : numpy.ndarray
        First array of returns to plot
    returns2 : numpy.ndarray
        Second array of returns to plot
    labels : list
        Labels for the two distributions
    title : str
        Plot title
    bins : int
        Number of histogram bins
    figsize : tuple
        Figure size (width, height)
    """
    # Calculate statistics
    mean1, std1 = np.mean(returns1), np.std(returns1)
    mean2, std2 = np.mean(returns2), np.std(returns2)
    
    # Calculate skewness
    skew1 = stats.skew(returns1)
    skew2 = stats.skew(returns2)
    
    # Calculate excess kurtosis (scipy.stats.kurtosis already calculates excess kurtosis by default)
    kurt1 = stats.kurtosis(returns1)
    kurt2 = stats.kurtosis(returns2)
    
    # Create subplot with two panels
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    
    # Plot histograms
    ax[0].hist(returns1, bins=bins, density=True, label=labels[0])
    ax[1].hist(returns2, bins=bins, density=True, label=labels[1])
    
    # Set consistent x-axes
    min_value = min(returns1.min(), returns2.min())
    max_value = max(returns1.max(), returns2.max())
    x_padding = (max_value - min_value) * 0.05
    x_min = min_value - x_padding
    x_max = max_value + x_padding
    ax[0].set_xlim(x_min, x_max)
    ax[1].set_xlim(x_min, x_max)
    
    # Add textboxes with statistics
    textstr1 = f"Mean: {mean1:.4f}\nSD: {std1:.4f}\nSkewness: {skew1:.2f}\nExcess Kurtosis: {kurt1:.2f}"
    textstr2 = f"Mean: {mean2:.4f}\nSD: {std2:.4f}\nSkewness: {skew2:.2f}\nExcess Kurtosis: {kurt2:.2f}"
    
    # Position the text boxes in top left corner
    props = dict(boxstyle='round', facecolor='lightgrey', alpha=0.8)
    ax[0].text(0.05, 0.95, textstr1, transform=ax[0].transAxes, fontsize=11,
              verticalalignment='top', bbox=props)
    ax[1].text(0.05, 0.95, textstr2, transform=ax[1].transAxes, fontsize=11,
              verticalalignment='top', bbox=props)
    
    # Set labels, title and legend
    ax[0].set_ylabel('Density')
    ax[0].set_xlabel('Returns')
    ax[1].set_xlabel('Returns')
    ax[0].legend()
    ax[1].legend()
    
    # Add overall title
    fig.suptitle(title, fontsize=20, weight="bold")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust for the suptitle
    plt.show()
    
    return fig, ax

# Example usage:
# plot_comparison_histograms(ew_returns, ew_returns_simulated, 
#                           title="Weekly Equal-weighted Portfolio Returns")


def create_kde_plots(dataframe, suptitle="Distribution Plot", figsize=(14, 10), n_rows=4, n_cols=2, textbox_position='left', textbox_charts = []):
    """
    Create KDE plots for all columns in a dataframe.
    
    Parameters:
    -----------
    dataframe : pandas.DataFrame
        The dataset containing columns to plot
    suptitle : str
        The super title for the entire figure
    figsize : tuple
        Figure size as (width, height)
    n_rows : int
        Number of rows in the subplot grid
    n_cols : int
        Number of columns in the subplot grid
    textbox_position : str
        Position of statistics textbox ('left', 'middle', 'right')
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plots
    """
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing
    
    # First find the min and max across all columns to standardize x-axis
    min_value = float('inf')
    max_value = float('-inf')
    for column in dataframe.columns:
        min_value = min(min_value, dataframe[column].min())
        max_value = max(max_value, dataframe[column].max())
    
    # Add some padding to the limits
    x_padding = (max_value - min_value) * 0.05
    x_min = min_value - x_padding
    x_max = max_value + x_padding
    
    # Create plots
    for i, column in enumerate(dataframe.columns):
        if i >= len(axes):  # Make sure we don't exceed available axes
            break
            
        data = dataframe[column]
        sns.kdeplot(data, fill=True, alpha=0.5, ax=axes[i], label=column)
        
        # Calculate statistics for textbox
        mean, sd = np.mean(data), np.std(data)
        
        # Add textboxes with statistics
        textstr = f"Mean: {mean:.2f}\nSD: {sd:.2f}"
        
        # Position the text box based on the specified position
        props = dict(boxstyle='round', facecolor='lightgrey', alpha=0.8)
        if i in textbox_charts:
            # Set position and alignment based on textbox_position parameter
            if textbox_position == 'left':
                x_pos, h_align = 0.05, 'left'
            elif textbox_position == 'middle':
                x_pos, h_align = 0.5, 'center'
            elif textbox_position == 'right':
                x_pos, h_align = 0.95, 'right'

        else: # default to left
            x_pos, h_align = 0.05, 'left'

        
            
        axes[i].text(x_pos, 0.95, textstr, transform=axes[i].transAxes, fontsize=11,
                  verticalalignment='top', horizontalalignment=h_align, bbox=props)
        
        # Set consistent x-axis limits
        axes[i].set_xlim(x_min, x_max)
        
        # Add titles
        axes[i].set_title(f"{column}")
        axes[i].set_xlabel("")  # x-axis should be obvious, cleaner to not repeat for all charts
        
        # Only set ylabel for left-side plots (even indices)
        if i % n_cols == 0:
            axes[i].set_ylabel("Density")
        else:
            axes[i].set_ylabel("")
    
    # Hide unused subplots if any
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    # Add overall title
    fig.suptitle(suptitle, fontsize=16, weight="bold")
    plt.tight_layout()
    # Adjust the layout to make room for the suptitle
    plt.subplots_adjust(top=0.9)
    
    return fig

# Example usage:
# fig = create_kde_plots(stress_test_SRs, "Distribution of Sharpe ratios")
# plt.show()
