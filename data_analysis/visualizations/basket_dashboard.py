import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings('ignore')

# Set style and aesthetics for professional visualization
plt.style.use('dark_background')
sns.set_style("darkgrid", {
    'axes.facecolor': '#1a1a1a',
    'figure.facecolor': '#121212',
    'grid.color': '#333333',
    'grid.linestyle': '--',
    'text.color': '#cccccc',
    'axes.labelcolor': '#cccccc',
    'xtick.color': '#cccccc',
    'ytick.color': '#cccccc',
    'axes.edgecolor': '#333333',
})

# Custom colors for visualization
COLORS = {
    'red': '#DC2626',
    'blue': '#1E3A8A',
    'green': '#047857',
    'purple': '#9333EA',
    'orange': '#F59E0B',
    'teal': '#0EA5E9',
    'yellow': '#EAB308',
    'pink': '#EC4899',
    'indigo': '#4F46E5',
}

# Custom color maps
CMAP_RISK = LinearSegmentedColormap.from_list('risk', 
    ['#047857', '#EAB308', '#DC2626'])  # Green to Yellow to Red
CMAP_RETURN = LinearSegmentedColormap.from_list('return', 
    ['#DC2626', '#F59E0B', '#047857'])  # Red to Yellow to Green

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
results_dir = os.path.join(base_dir, "results")
visualizations_dir = current_dir  # Save directly to the visualizations directory

# Create visualizations directory if it doesn't exist
os.makedirs(visualizations_dir, exist_ok=True)

# Function to format percentages
def format_pct(x, pos=0):
    return f'{x:.1%}'

# Function to format currency values
def format_currency(x, pos=0):
    return f'${x:.2f}'

class BasketDashboard:
    """Class for creating a comprehensive dashboard of stock baskets visualizations"""
    
    def __init__(self):
        # Load data - Use the updated factor baskets data
        self.baskets = pd.read_csv(os.path.join(results_dir, "updated_factor_baskets.csv"))
        
        # Check if factor_loadings.csv exists
        factor_loadings_path = os.path.join(results_dir, "factor_loadings.csv")
        if os.path.exists(factor_loadings_path):
            self.factor_loadings = pd.read_csv(factor_loadings_path)
            # Rename the first column in factor_loadings to 'Stock'
            self.factor_loadings = self.factor_loadings.rename(columns={self.factor_loadings.columns[0]: 'Stock'})
        else:
            print("Warning: factor_loadings.csv not found. Creating dummy data.")
            # Create dummy factor loadings data using stocks from baskets
            all_stocks = []
            for idx, row in self.baskets.iterrows():
                all_stocks.extend(row['Stocks'].split(','))
            
            # Create a DataFrame with dummy values
            self.factor_loadings = pd.DataFrame({
                'Stock': all_stocks,
                'Alpha': np.random.uniform(0.05, 0.1, len(all_stocks)),
                'Beta': np.random.uniform(0.5, 1.5, len(all_stocks)),
                'SMB': np.random.uniform(-0.3, 0.5, len(all_stocks)),
                'HML': np.random.uniform(-0.3, 0.3, len(all_stocks)),
                'R2': np.random.uniform(0.7, 0.9, len(all_stocks))
            })
        
        # Extract each basket's stocks
        self.basket_stocks = {}
        for idx, row in self.baskets.iterrows():
            basket_num = row['BasketNumber']
            stocks_list = row['Stocks'].split(',')
            self.basket_stocks[basket_num] = stocks_list
        
        # Set up figure sizes and DPI for high-quality output
        self.figure_size = (14, 10)
        self.dpi = 300
        
        print("BasketDashboard initialized. Ready to create visualizations.")
        
    def create_risk_return_profile(self, save=True, show=False):
        """Create enhanced risk-return profile visualization with efficient frontier"""
        
        plt.figure(figsize=self.figure_size)
        
        # Create gradient colors based on risk ratings
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.baskets)))
        
        # Extract data
        volatilities = self.baskets['PredictedAnnualVolatility'].values
        returns = self.baskets['PredictedAnnualReturn'].values
        basket_nums = self.baskets['BasketNumber'].values
        names = self.baskets['Name'].values
        risk_ratings = self.baskets['RiskRating'].values
        
        # Create risk-return scatter plot
        for i, (vol, ret, basket, name, color) in enumerate(
            zip(volatilities, returns, basket_nums, names, colors)):
            # Size based on risk rating and number of stocks
            marker_size = 400 + risk_ratings[i] * 100
            
            # Plot point
            plt.scatter(vol, ret, s=marker_size, color=color, alpha=0.8, 
                      edgecolor='white', linewidth=2, zorder=5)
            
            # Add basket number and name as annotation
            plt.annotate(f"Basket {basket}\n{name}", 
                       xy=(vol, ret), xytext=(10, 10),
                       textcoords='offset points', 
                       fontsize=12, fontweight='bold',
                       color='white',
                       bbox=dict(boxstyle="round,pad=0.3", fc=color, alpha=0.7))
        
        # Generate a simulated efficient frontier
        vol_range = np.linspace(0, max(volatilities)*1.2, 100)
        # Simplified model of expected returns based on volatility
        frontier_returns = 0.05 + 0.3 * vol_range - 0.5 * vol_range**2
        
        # Plot the frontier
        plt.plot(vol_range, frontier_returns, 'w--', alpha=0.5, linewidth=2, 
                label='Simulated Efficient Frontier')
        
        # Add risk-free rate point
        risk_free_rate = 0.05  # 5% annual
        plt.scatter(0, risk_free_rate, s=150, color='gold', marker='*', 
                  edgecolor='white', linewidth=1, label='Risk-Free Rate (5%)')
        
        # Add benchmark market return point (estimated)
        market_return = 0.08  # 8% annual
        market_vol = 0.15  # 15% annual volatility
        plt.scatter(market_vol, market_return, s=200, color=COLORS['red'], marker='d', 
                  edgecolor='white', linewidth=1, label='Market Benchmark (Est.)')
        
        # Add a zero return line for reference
        plt.axhline(y=0, color='white', linestyle=':', alpha=0.3)
        
        # Style the plot
        plt.title('Risk-Return Profile of Stock Baskets', fontsize=20, pad=20)
        plt.xlabel('Expected Annual Volatility', fontsize=14)
        plt.ylabel('Expected Annual Return', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left', fontsize=12)
        
        # Format axes as percentages
        plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        
        # Set reasonable axis limits
        x_max = max(volatilities) * 1.1
        y_min = min(min(returns) * 1.1, -0.05)  # At least -5%
        y_max = max(max(returns) * 1.1, market_return * 1.1)  # At least 10% above market
        
        plt.xlim(0, x_max)
        plt.ylim(y_min, y_max)
        
        # Add grid
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Add text about baskets and diversification
        plt.figtext(0.02, 0.02, 
                  "Note: Bubble size represents risk level. The model sorts stocks into 5 distinct risk baskets based on Fama-French factors and Beta.",
                  ha="left", fontsize=10, alpha=0.7)
        
        # Save and/or show
        if save:
            output_file = os.path.join(visualizations_dir, "basket_risk_return_profile.png")
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            print(f"Risk-return profile saved to {output_file}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def create_factor_exposure_dashboard(self, save=True, show=False):
        """Create comprehensive factor exposure visualization dashboard"""
        
        # Set up figure with grid layout
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig)
        
        # Title for the entire dashboard
        fig.suptitle('Factor Exposure Analysis by Basket', fontsize=24, y=0.98)
        
        # 1. Beta Exposure - Top left
        ax1 = fig.add_subplot(gs[0, 0])
        sns.barplot(x='BasketNumber', y='AvgBeta', data=self.baskets, 
                   palette='viridis', ax=ax1)
        ax1.set_title('Market Beta (β)', fontsize=14)
        ax1.set_xlabel('')
        ax1.set_ylabel('Average Beta', fontsize=12)
        
        # Add value labels
        for i, p in enumerate(ax1.patches):
            height = p.get_height()
            ax1.annotate(f'{height:.2f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=10, fontweight='bold')
        
        # 2. SMB Exposure - Top middle
        ax2 = fig.add_subplot(gs[0, 1])
        sns.barplot(x='BasketNumber', y='AvgSMB', data=self.baskets, 
                   palette='viridis', ax=ax2)
        ax2.set_title('Size Factor (SMB)', fontsize=14)
        ax2.set_xlabel('')
        ax2.set_ylabel('Average SMB', fontsize=12)
        
        # Add value labels
        for i, p in enumerate(ax2.patches):
            height = p.get_height()
            ax2.annotate(f'{height:.2f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=10, fontweight='bold')
        
        # 3. HML Exposure - Top right
        ax3 = fig.add_subplot(gs[0, 2])
        sns.barplot(x='BasketNumber', y='AvgHML', data=self.baskets, 
                   palette='viridis', ax=ax3)
        ax3.set_title('Value Factor (HML)', fontsize=14)
        ax3.set_xlabel('')
        ax3.set_ylabel('Average HML', fontsize=12)
        
        # Add value labels
        for i, p in enumerate(ax3.patches):
            height = p.get_height()
            ax3.annotate(f'{height:.2f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=10, fontweight='bold')
        
        # 4. Factor Heatmap - Middle row spanning all columns
        ax4 = fig.add_subplot(gs[1, :])
        
        # Prepare data for heatmap
        heatmap_data = self.baskets[['BasketNumber', 'AvgBeta', 'AvgSMB', 'AvgHML']].set_index('BasketNumber')
        
        # Create heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, linewidths=1, ax=ax4)
        ax4.set_title('Factor Exposure Heatmap by Basket', fontsize=16)
        ax4.set_xlabel('')
        ax4.set_ylabel('Basket Number', fontsize=12)
        
        # 5. Factor Distribution Boxplot - Bottom row spanning all columns
        ax5 = fig.add_subplot(gs[2, :])
        
        # Prepare data for boxplot - reshape to long format
        factor_cols = ['Beta', 'SMB', 'HML']
        boxplot_data = pd.DataFrame()
        
        for basket_num in self.baskets['BasketNumber']:
            stocks = self.basket_stocks[basket_num]
            basket_loadings = self.factor_loadings[self.factor_loadings['Stock'].isin(stocks)]
            
            for factor in factor_cols:
                temp_df = pd.DataFrame({
                    'Basket': [basket_num] * len(basket_loadings),
                    'Factor': [factor] * len(basket_loadings),
                    'Value': basket_loadings[factor].values
                })
                boxplot_data = pd.concat([boxplot_data, temp_df])
        
        # Create boxplot
        sns.boxplot(x='Basket', y='Value', hue='Factor', data=boxplot_data, 
                   palette=[COLORS['blue'], COLORS['green'], COLORS['orange']], ax=ax5)
        ax5.set_title('Factor Loading Distribution by Basket', fontsize=16)
        ax5.set_xlabel('Basket Number', fontsize=12)
        ax5.set_ylabel('Factor Loading Value', fontsize=12)
        ax5.legend(title='Factor', loc='upper right')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save and/or show
        if save:
            output_file = os.path.join(visualizations_dir, "basket_factor_exposures.png")
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            print(f"Factor exposure dashboard saved to {output_file}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def create_risk_return_matrix(self, save=True, show=False):
        """Create risk-return rating matrix visualization"""
        
        plt.figure(figsize=(12, 10))
        
        # Create a 5x5 risk-return grid
        grid = np.zeros((5, 5))
        
        # Fill grid with basket numbers
        for _, row in self.baskets.iterrows():
            risk_idx = int(row['RiskRating']) - 1
            return_idx = int(row['ReturnRating']) - 1
            grid[return_idx, risk_idx] = row['BasketNumber']
        
        # Plot heatmap with text
        ax = plt.gca()
        im = ax.imshow(grid, cmap='viridis', aspect='equal')
        
        # Add text annotations
        for i in range(5):
            for j in range(5):
                basket_num = int(grid[i, j])
                if basket_num > 0:
                    # Look up extra information about this basket
                    basket_info = self.baskets[self.baskets['BasketNumber'] == basket_num].iloc[0]
                    num_stocks = basket_info['NumStocks']
                    
                    ax.text(j, i, f"Basket {basket_num}\n({num_stocks} stocks)", 
                           ha="center", va="center", color="white", 
                           fontsize=12, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", fc='black', alpha=0.3))
                else:
                    ax.text(j, i, "—", ha="center", va="center", 
                           color="gray", fontsize=14)
        
        # Set tick labels
        risk_labels = ['Very Low (1)', 'Low (2)', 'Moderate (3)', 'High (4)', 'Very High (5)']
        return_labels = ['Low (1)', 'Moderate-Low (2)', 'Moderate (3)', 'Moderate-High (4)', 'High (5)']
        
        ax.set_xticks(np.arange(len(risk_labels)))
        ax.set_yticks(np.arange(len(return_labels)))
        ax.set_xticklabels(risk_labels, fontsize=10)
        ax.set_yticklabels(return_labels, fontsize=10)
        
        # Rotate x tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add labels and title
        plt.title('Risk-Return Rating Matrix of Stock Baskets', fontsize=18, pad=20)
        plt.xlabel('Risk Rating', fontsize=14, labelpad=15)
        plt.ylabel('Return Rating', fontsize=14, labelpad=15)
        
        # Grid
        ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 5, 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=1, alpha=0.3)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
        cbar.set_label('Basket Number', rotation=270, labelpad=20, fontsize=12)
        
        # Annotation
        plt.figtext(0.5, 0.01, 
                  "Empty cells (—) indicate no baskets with that specific risk-return profile combination.",
                  ha="center", fontsize=10, alpha=0.8)
        
        # Save and/or show
        if save:
            output_file = os.path.join(visualizations_dir, "basket_risk_return_matrix.png")
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            print(f"Risk-return matrix saved to {output_file}")
            
            # Save a second copy with the name used in the HTML
            updated_output_file = os.path.join(visualizations_dir, "updated_risk_return_matrix.png")
            plt.savefig(updated_output_file, dpi=self.dpi, bbox_inches='tight')
            print(f"Updated risk-return matrix saved to {updated_output_file}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def create_basket_characteristics_dashboard(self, save=True, show=False):
        """Create comprehensive basket characteristics dashboard"""
        
        # Set up figure with grid layout
        fig = plt.figure(figsize=(16, 14))
        gs = GridSpec(3, 2, figure=fig)
        
        # Title for the entire dashboard
        fig.suptitle('Stock Basket Characteristics', fontsize=24, y=0.98)
        
        # 1. Expected Returns - Top left
        ax1 = fig.add_subplot(gs[0, 0])
        returns = self.baskets['PredictedAnnualReturn'].values
        
        # Use a diverging color map based on return values
        colors = [(1, 0, 0) if r < 0 else (0, 0.8, 0) for r in returns]
        
        sns.barplot(x='BasketNumber', y='PredictedAnnualReturn', data=self.baskets, 
                   palette=colors, ax=ax1)
        ax1.set_title('Expected Annual Return', fontsize=16)
        ax1.set_xlabel('Basket Number', fontsize=12)
        ax1.set_ylabel('Annual Return', fontsize=12)
        ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax1.axhline(y=0, color='white', linestyle='--', alpha=0.5)
        
        # Add value labels
        for i, p in enumerate(ax1.patches):
            height = p.get_height()
            ax1.annotate(f'{height:.1%}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=10, fontweight='bold')
        
        # 2. Expected Volatility - Top right
        ax2 = fig.add_subplot(gs[0, 1])
        sns.barplot(x='BasketNumber', y='PredictedAnnualVolatility', data=self.baskets, 
                   palette='YlOrRd', ax=ax2)
        ax2.set_title('Expected Annual Volatility', fontsize=16)
        ax2.set_xlabel('Basket Number', fontsize=12)
        ax2.set_ylabel('Annual Volatility', fontsize=12)
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        
        # Add value labels
        for i, p in enumerate(ax2.patches):
            height = p.get_height()
            ax2.annotate(f'{height:.1%}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=10, fontweight='bold')
        
        # 3. Risk-Return Scatter - Middle row, span both columns
        ax3 = fig.add_subplot(gs[1, :])
        
        # Extract data
        volatilities = self.baskets['PredictedAnnualVolatility'].values
        returns = self.baskets['PredictedAnnualReturn'].values
        basket_nums = self.baskets['BasketNumber'].values
        
        # Create scatter plot with custom sizing and coloring
        scatter = ax3.scatter(volatilities, returns, s=300, c=basket_nums, 
                           cmap='viridis', edgecolor='white', linewidth=1)
        
        # Add basket labels
        for i, basket_num in enumerate(basket_nums):
            ax3.annotate(f"Basket {basket_num}",
                        (volatilities[i], returns[i]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=10, fontweight='bold')
        
        # Add theoretical Sharpe ratio lines
        x_vals = np.linspace(0, max(volatilities) * 1.2, 100)
        
        # Risk-free rate
        rf_rate = 0.05
        
        # Plot Sharpe ratio lines
        for sharpe in [0.5, 1.0, 1.5]:
            y_vals = rf_rate + sharpe * x_vals
            ax3.plot(x_vals, y_vals, '--', color='gray', alpha=0.5)
            # Label at the right edge
            ax3.annotate(f"Sharpe = {sharpe}", 
                        (x_vals[-1], y_vals[-1]),
                        xytext=(5, 0),
                        textcoords="offset points",
                        fontsize=8, alpha=0.7)
        
        ax3.set_title('Risk-Return Relationship with Sharpe Ratio Lines', fontsize=16)
        ax3.set_xlabel('Expected Annual Volatility', fontsize=12)
        ax3.set_ylabel('Expected Annual Return', fontsize=12)
        ax3.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax3.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax3.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax3)
        cbar.set_label('Basket Number')
        
        # 4. Stock count by basket - Bottom left
        ax4 = fig.add_subplot(gs[2, 0])
        sns.barplot(x='BasketNumber', y='NumStocks', data=self.baskets, 
                   palette='viridis', ax=ax4)
        ax4.set_title('Number of Stocks per Basket', fontsize=16)
        ax4.set_xlabel('Basket Number', fontsize=12)
        ax4.set_ylabel('Stock Count', fontsize=12)
        
        # Add value labels
        for i, p in enumerate(ax4.patches):
            height = p.get_height()
            ax4.annotate(f'{int(height)}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=10, fontweight='bold')
        
        # 5. Rating comparison - Bottom right
        ax5 = fig.add_subplot(gs[2, 1])
        
        # Create a grouped bar chart for risk and return ratings
        bar_width = 0.35
        x = np.arange(len(self.baskets))
        
        # Plot risk ratings
        risk_bars = ax5.bar(x - bar_width/2, self.baskets['RiskRating'], 
                          bar_width, label='Risk Rating',
                          color=COLORS['red'], alpha=0.8)
        
        # Plot return ratings
        return_bars = ax5.bar(x + bar_width/2, self.baskets['ReturnRating'], 
                            bar_width, label='Return Rating',
                            color=COLORS['green'], alpha=0.8)
        
        # Add labels and styling
        ax5.set_title('Risk vs. Return Ratings', fontsize=16)
        ax5.set_xlabel('Basket Number', fontsize=12)
        ax5.set_ylabel('Rating (1-5 Scale)', fontsize=12)
        ax5.set_xticks(x)
        ax5.set_xticklabels(self.baskets['BasketNumber'])
        ax5.set_yticks(range(1, 6))
        ax5.set_ylim(0, 5.5)
        ax5.legend()
        
        # Add value labels
        for bar in [risk_bars, return_bars]:
            for rect in bar:
                height = rect.get_height()
                ax5.annotate(f'{int(height)}',
                            xy=(rect.get_x() + rect.get_width()/2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=10, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save and/or show
        if save:
            output_file = os.path.join(visualizations_dir, "basket_characteristics_dashboard.png")
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            print(f"Basket characteristics dashboard saved to {output_file}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def create_basket_stock_examples(self, save=True, show=False):
        """Create visualization showing example stocks from each basket"""
        
        # Create a simpler visualization since we don't have full factor data for all stocks
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Title for the entire figure
        fig.suptitle('Top Stocks by Percentage Weight in Each Basket', fontsize=24, y=0.98)
        
        # Color mapping for baskets
        basket_colors = {
            1: '#34C759',  # green for low risk
            2: '#5AC8FA',  # blue for moderate-low risk
            3: '#FFCC00',  # yellow for moderate risk
            4: '#FF9500',  # orange for moderate-high risk
            5: '#FF3B30',  # red for high risk
        }
        
        # For each basket, create a horizontal bar for top stocks
        y_positions = []
        bar_heights = []
        bar_colors = []
        bar_labels = []
        
        # Track the current position
        current_y = 0
        y_ticks = []
        y_labels = []
        
        for basket_num in sorted(self.basket_stocks.keys()):
            basket_info = self.baskets[self.baskets['BasketNumber'] == basket_num].iloc[0]
            stocks = self.basket_stocks[basket_num]
            
            # Add basket header position
            y_ticks.append(current_y)
            y_labels.append(f"Basket {basket_num}: {basket_info['Name']}")
            current_y += 1
            
            # Weights from the basket cards in the website
            weights_map = {
                1: [31.54, 15.69, 11.77, 11.53, 11.35, 9.88, 8.24],  # Basket 1 weights
                2: [18.94, 17.70, 14.80, 13.83, 13.12, 11.18, 10.43],  # Basket 2 weights
                3: [17.17, 17.13, 15.61, 14.61, 12.67, 11.60, 11.21],  # Basket 3 weights
                4: [17.77, 15.80, 14.70, 14.08, 13.65, 12.68, 11.32],  # Basket 4 weights
                5: [17.31, 16.98, 16.85, 13.37, 12.65, 11.88, 10.96],  # Basket 5 weights
            }
            
            weights = weights_map[basket_num]
            
            # Add each stock
            for i, stock in enumerate(stocks):
                y_positions.append(current_y)
                bar_heights.append(weights[i])  # Use actual weights
                bar_colors.append(basket_colors[basket_num])
                bar_labels.append(f"{stock}")
                current_y += 0.7
            
            # Add space between baskets
            current_y += 1.5
        
        # Create horizontal bars
        bars = ax.barh(y_positions, bar_heights, height=0.5, color=bar_colors, alpha=0.8)
        
        # Add labels to each bar
        for i, (bar, label) in enumerate(zip(bars, bar_labels)):
            width = bar.get_width()
            ax.text(width + 1, y_positions[i], f"{label} ({width:.2f}%)", 
                   va='center', fontsize=10, fontweight='bold')
        
        # Style the plot
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=12, fontweight='bold')
        ax.set_xlabel('Weight in Basket (%)', fontsize=14)
        ax.invert_yaxis()  # Make the top basket appear at the top
        ax.grid(axis='x', alpha=0.3)
        
        # Remove y-axis line
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # Description text
        plt.figtext(0.5, 0.01, 
                  "Visualization of top stocks in each basket with their percentage weights.",
                  ha="center", fontsize=10, alpha=0.8)
        
        # Add basket descriptions
        for i, basket_num in enumerate(sorted(self.basket_stocks.keys())):
            basket_info = self.baskets[self.baskets['BasketNumber'] == basket_num].iloc[0]
            description = f"β: {basket_info['AvgBeta']:.2f}, Return: {basket_info['PredictedAnnualReturn']:.1%}, Volatility: {basket_info['PredictedAnnualVolatility']:.1%}"
            ax.text(0, y_ticks[i] - 0.5, description, fontsize=10, alpha=0.8)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save and/or show
        if save:
            output_file = os.path.join(visualizations_dir, "basket_stock_examples.png")
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            print(f"Basket stock examples saved to {output_file}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def create_risk_return_relationship(self, save=True, show=False):
        """Create a visualization showing the relationship between risk and return ratings"""
        
        plt.figure(figsize=(12, 8))
        
        # Extract data
        risk_ratings = self.baskets['RiskRating'].values
        return_ratings = self.baskets['ReturnRating'].values
        returns = self.baskets['PredictedAnnualReturn'].values
        basket_nums = self.baskets['BasketNumber'].values
        
        # Create scatter plot
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.baskets)))
        
        for i, (risk, ret_rating, ret, basket, color) in enumerate(
            zip(risk_ratings, return_ratings, returns, basket_nums, colors)):
            plt.scatter(risk, ret_rating, s=400, color=color, alpha=0.8, 
                      edgecolor='white', linewidth=2, zorder=5)
            
            # Add basket number annotation
            plt.annotate(f"Basket {basket}\n{ret:.1%}", 
                       xy=(risk, ret_rating), xytext=(0, 0),
                       textcoords='offset points', 
                       fontsize=12, fontweight='bold',
                       color='white', ha='center', va='center')
        
        # Add trend line
        z = np.polyfit(risk_ratings, return_ratings, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(0.5, 5.5, 100)
        y_trend = p(x_trend)
        
        plt.plot(x_trend, y_trend, 'w--', alpha=0.5, linewidth=2, 
                label=f'Trend: y = {z[0]:.2f}x + {z[1]:.2f}')
        
        # Style the plot
        plt.title('Risk-Return Rating Relationship', fontsize=20, pad=20)
        plt.xlabel('Risk Rating', fontsize=14)
        plt.ylabel('Return Rating', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left', fontsize=12)
        
        # Set axis limits and ticks
        plt.xlim(0.5, 5.5)
        plt.ylim(0.5, 5.5)
        plt.xticks(range(1, 6))
        plt.yticks(range(1, 6))
        
        # Add risk labels
        risk_labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
        return_labels = ['Low', 'Moderate-Low', 'Moderate', 'Moderate-High', 'High']
        
        for i, label in enumerate(risk_labels):
            plt.text(i+1, 0.7, label, ha='center', fontsize=9, alpha=0.7, rotation=45)
        
        for i, label in enumerate(return_labels):
            plt.text(0.7, i+1, label, va='center', fontsize=9, alpha=0.7, rotation=45)
        
        # Add description
        plt.figtext(0.5, 0.01, 
                  "Relationship between risk ratings (1-5) and return ratings (1-5) across all baskets.",
                  ha="center", fontsize=10, alpha=0.8)
        
        # Save and/or show
        if save:
            output_file = os.path.join(visualizations_dir, "risk_return_relationship.png")
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            print(f"Risk-return relationship chart saved to {output_file}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def create_basket_return_ratings(self, save=True, show=False):
        """Create visualization showing return ratings for each basket"""
        
        plt.figure(figsize=(14, 8))
        
        # Extract data
        basket_nums = self.baskets['BasketNumber'].values
        names = self.baskets['Name'].values
        returns = self.baskets['PredictedAnnualReturn'].values
        return_ratings = self.baskets['ReturnRating'].values
        
        # Define colors based on return ratings
        colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, 5))
        bar_colors = [colors[int(r)-1] for r in return_ratings]
        
        # Create bar chart
        bars = plt.bar(basket_nums, returns, color=bar_colors, alpha=0.8, width=0.6)
        
        # Add value and rating labels
        for i, (bar, ret, rating) in enumerate(zip(bars, returns, return_ratings)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.005, 
                    f"{height:.1%}", ha='center', va='bottom', 
                    fontsize=12, fontweight='bold')
            
            plt.text(bar.get_x() + bar.get_width()/2, height/2, 
                    f"{int(rating)}/5", ha='center', va='center', 
                    fontsize=14, fontweight='bold', color='white')
        
        # Add titles and labels
        plt.title('Expected Returns & Rating Scale', fontsize=20, pad=20)
        plt.xlabel('Basket Number', fontsize=14)
        plt.ylabel('Expected Annual Return', fontsize=14)
        
        # Format y-axis as percentage
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        
        # Set x-ticks
        plt.xticks(basket_nums)
        
        # Add basket names
        for i, (num, name) in enumerate(zip(basket_nums, names)):
            plt.text(num, -0.01, name, ha='center', va='top', fontsize=10, 
                    rotation=45, alpha=0.8)
        
        # Add grid
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add rating scale legend
        legend_labels = [
            'Rating 1/5: Low',
            'Rating 2/5: Moderate-Low',
            'Rating 3/5: Moderate',
            'Rating 4/5: Moderate-High',
            'Rating 5/5: Extremely High'
        ]
        
        for i, label in enumerate(legend_labels):
            plt.plot([0], [0], 'o', color=colors[i], alpha=0.8, label=label)
        
        plt.legend(loc='upper left', fontsize=10)
        
        # Save and/or show
        if save:
            output_file = os.path.join(visualizations_dir, "basket_return_ratings.png")
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            print(f"Basket return ratings chart saved to {output_file}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def create_all_visualizations(self, show=False):
        """Create all visualizations in one go"""
        
        print("Creating comprehensive basket visualizations...")
        
        self.create_risk_return_profile(show=show)
        self.create_factor_exposure_dashboard(show=show)
        self.create_risk_return_matrix(show=show)
        self.create_basket_characteristics_dashboard(show=show)
        self.create_basket_stock_examples(show=show)
        self.create_risk_return_relationship(show=show)
        self.create_basket_return_ratings(show=show)
        
        print("All visualizations completed successfully!")
        print(f"Visualizations saved to: {visualizations_dir}")
        
        # Return the path to make it easier to find the files
        return visualizations_dir

# Execute if run as a script
if __name__ == "__main__":
    dashboard = BasketDashboard()
    dashboard.create_all_visualizations(show=False) 