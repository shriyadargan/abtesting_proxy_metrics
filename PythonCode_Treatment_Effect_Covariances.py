"""
Project: Learning Treatment Effect Covariances
From Naive to Robust Methods (JIVE, LIML and TC)

Author: Shriya Dargan
Date: December 2025

Description: This script contains utility functions for synthetic data generation, 
calculation of treatment effects, naive OLS estimators and 
robust estimators (JIVE, LIML, TC), followed by vizualisation.

Short-term metric (S): Average watch time in week 1
Long-term metric (Y): Subscriber retention at 6 months

Goal: Find the relationship β where Treatment Effect on Y ≈ β x Treatment Effect on S
This way, in future experiments, we can predict long-term impact from short-term observations.
Users with high S tend to have high Y. This is called noise correlation (ρ) due to user-level heterogeneity.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

#======================================================
# Part 1: Synthetic Data Generation
#======================================================

class ABTestDataGenerator:
    """
    Genrates synthetic A/B test data mimicking real-world scenarios
    where treatment effects are weak and user-level noise is correlated.

    """
    def __init__(self,
                 num_experiments: int = 100,
                 users_per_experiment: int = 10000,
                 true_beta: float = 0.2,
                 noise_correlation: float = 0.6,
                 signal_strength: float = 0.5,
                 random_seed: int = 42):
        """
        Initializes the data generator with specified parameters.
        Parameters:

        num_experiments (int): Number of A/B test experiments to simulate.
        users_per_experiment (int): Number of users in each experiment. Sample size per experiment.
        true_beta (float): The true treatment effect size. True causal effect of S on Y (Beta in paper)
        noise_correlation (float): User-level correlation coefficient between S and Y.
        signal_strength (float): Magnitude of tratement effect.
        random_seed (int): Seed for reproducibility.
        """
        self.K = num_experiments
        self.n = users_per_experiment
        self.beta = true_beta
        self.rho = noise_correlation
        self.signal = signal_strength
        np.random.seed(random_seed)

    def generate_experiments(self) -> pd.DataFrame:
        """
        Generates synthetic data from K A/B test experiments with weak treatment effects.
        Returns:
        pd.DataFrame: DataFrame containing synthetic data for all experiments.
        """
        data_list = []
        for exp_id in range(self.K):
            # true treatmente effect on S for this experimet
            true_tau_S = np.random.uniform(-self.signal, self.signal)

            # true treatment effect on Y: Y = Beta * S (no direct effects)
            true_tau_Y = self.beta * true_tau_S

            # generrate user-level data
            for user_id in range(self.n):
                treatment = 1 if user_id < self.n // 2 else 0  # 50-50 split
                
                # Generate correlated noise
                noise_base =np.random.randn()
                noise_S = noise_base + np.random.randn()*np.sqrt(1 - self.rho**2)
                noise_Y = self.rho * noise_base + np.random.randn()*np.sqrt(1 - self.rho**2)

                # Observed Outcomes
                S = treatment * true_tau_S + noise_S
                Y = treatment * true_tau_Y + noise_Y * 0.5 # scaled noise for Y

                data_list.append({
                    'experiment_id': exp_id,
                    'treatment': treatment,
                    'S': S,
                    'Y': Y,
                    'true_tau_S': true_tau_S,
                    'user_id': user_id,
                    'true_tau_Y': true_tau_Y 
                    })
                
        return pd.DataFrame(data_list)
    
#======================================================
    def get_noise_covariance(self) -> np.ndarray:
        """
        Return the true noise covariance matrix Ω. 
        Constructs the noise covariance matrix based on the specified correlation.
         In practice, this would be estimated from historical data.
        Returns:
        np.ndarray: Covariance matrix for noise in S and Y.
        """
        variance_S = 1.0
        variance_Y = 0.25

        cov_SY = self.rho * np.sqrt(variance_S * variance_Y)

        return np.array([
            [variance_S, cov_SY],
            [cov_SY, variance_Y]
            ])

#======================================================
# Part 2: Treatment Effect Estimators
#======================================================

def calculate_treatment_effects(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate estimated treatment effects for each experiment.

    Returns:
    DataFrame with estimated treatment effects on S and Y for each experiment with columns:
    experiment_id, tau_S, tau_Y, true_tau_S, true_tau_Y
    """
    effects = data.groupby('experiment_id').apply(
        lambda exp: pd.Series({
            'tau_S': exp[exp['treatment']==1]['S'].mean() -exp[exp['treatment']==0]['S'].mean(),
            'tau_Y': exp[exp['treatment']==1]['Y'].mean() -exp[exp['treatment']==0]['Y'].mean(),
            'true_tau_S': exp['true_tau_S'].iloc[0],
            'true_tau_Y': exp['true_tau_Y'].iloc[0],
            'n': len(exp)
        })
    ).reset_index()
    
    return effects
    
#======================================================
# Part 3: Naive Estimator (Biased)
#======================================================

def naive_ols_estimator(effects: pd.DataFrame) -> Dict:
    """
    Naive OLS: Regress estimated tau_Y on estimated tau_S.
    This is BIASED when noise is correlated and effects are weak.
    
    Returns:
    Dict: Dictionary containing estimated beta and standard error.

    """
    # Calculate emprirical covariance matrix
    tau_S = effects['tau_S'].values
    tau_Y = effects['tau_Y'].values

    # Center the effects
    tau_S_centered = tau_S - tau_S.mean()
    tau_Y_centered = tau_Y - tau_Y.mean()

    # covariance matrix of ESTIMATED treatment effects
    cov_SS = np.mean(tau_S_centered ** 2)
    cov_SY = np.mean(tau_S_centered * tau_Y_centered)

    # Naive beta estimate
    beta_naive = cov_SY / cov_SS


    return {
    'beta': beta_naive,
    'cov_SS': cov_SS,
    'cov_SY': cov_SY,
    'method': 'Naive OLS'
    }

#======================================================
# Part 4: Robust Estimators - TOTAL COVARIANCE (TC)
#======================================================

def tc_estimator(effects: pd.DataFrame,
                 noise_cov: np.ndarray,
                 n: int) -> Dict:
    """
    Total Covariance (TC) Estimator: Subtract noise covariance from empirical covariance.
    Formula: β_TC = (Cov(tau_S, tau_Y) - Ω_SY) / (Cov(tau_S, tau_S) - Ω_SS)
    OR
    Formula: Λ_true = Σ_observed - (4/n) * Ω
    This is ROBUST to direct effects under INSIDE assumption.
    Returns:
    Dict: Dictionary containing estimated beta and standard error.
    """
    tau_S = effects['tau_S'].values
    tau_Y = effects['tau_Y'].values

    # Observed covariance (same as naive)
    tau_S_centered = tau_S - tau_S.mean()
    tau_Y_centered = tau_Y - tau_Y.mean()
   
    obs_cov_SS = np.mean(tau_S_centered ** 2)
    obs_cov_SY = np.mean(tau_S_centered * tau_Y_centered)

    # Noise contribution (4/n scaling for A/B test with equal split for treatment and control)
    noise_factor = 4/n
    noise_cov_SS = noise_cov[0,0]
    noise_cov_SY = noise_cov[0,1]

    # Subtract noise from observed covariance to get TRUE treatment effect covariance
    true_cov_SS = obs_cov_SS - noise_factor * noise_cov_SS
    true_cov_SY = obs_cov_SY - noise_factor * noise_cov_SY

    # TC beta estimate - corrected for noise
    beta_tc = true_cov_SY / true_cov_SS

    return {
       'beta': beta_tc,
       'cov_SS': true_cov_SS,
       'cov_SY': true_cov_SY,
       'obs_cov_SS': obs_cov_SS,
       'obs_cov_sy': obs_cov_SY,
       'noise_contrib_SS': noise_factor * noise_cov_SS,
       'noise_contrib_SY': noise_factor * noise_cov_SY,
        'method': 'Total Covariance (TC)'
    }

#======================================================
# Part 5: JACKKNIFE IV (JIVE) ESTIMATOR
#======================================================

def jive_estimator(data: pd.DataFrame) -> Dict:
    """
    Jackknife IV (JIVE) Estimator: Uses leave-one-out predictions as instruments.
    For each user, calculate treatment effect WITHOUT that user, 
    then correlate with that user's outcome.
    This is computationally expensive but requires no assumptions.

    """
    experiments = data['experiment_id'].unique()
    K = len(experiments)

    # store jackknife covariances for each experiment
    jive_cov_SS_list = []
    jive_cov_SY_list = []

    for exp_id in experiments:
        exp_data = data[data['experiment_id'] == exp_id].copy()
        n_exp = len(exp_data)

        # precompute full treatment effect
        treat_mean_S = exp_data[exp_data['treatment']==1]['S'].mean()
        control_mean_S = exp_data[exp_data['treatment']==0]['S'].mean()
        treat_mean_Y = exp_data[exp_data['treatment']==1]['Y'].mean()
        control_mean_Y = exp_data[exp_data['treatment']==0]['Y'].mean()

        tau_S_full = treat_mean_S - control_mean_S
        tau_Y_full = treat_mean_Y - control_mean_Y

        # Jackknife: for each user, calculate effect without them - 
        # Leave-one-out treatment effects

        jive_products_S = []
        jive_products_Y = []

        for idx, row in exp_data.iterrows():
            user_S = row['S']
            user_Y = row['Y']
            is_treatment = row['treatment']

            # calculate effect WITHOUT this user
            if is_treatment:
                n_treat = (exp_data['treatment'] == 1).sum()
                tau_S_minus_i = ((treat_mean_S * n_treat - user_S)/(n_treat - 1)) - control_mean_S
                tau_Y_minus_i = ((treat_mean_Y * n_treat - user_Y)/(n_treat - 1)) - control_mean_Y
            else:
                n_control = (exp_data['treatment'] == 0).sum()
                tau_S_minus_i = treat_mean_S - ((control_mean_S * n_control - user_S)/(n_control - 1))
                tau_Y_minus_i = treat_mean_Y - ((control_mean_Y * n_control - user_Y)/(n_control - 1))

            # Transform user outcome to treatment effect scale
            user_tau_S = 2 * user_S * ( 2 * is_treatment - 1)
            user_tau_Y = 2 * user_Y * (2 * is_treatment - 1)

            # accumulate products for covariance
            jive_products_S.append(tau_S_minus_i * user_tau_S)
            jive_products_Y.append(tau_S_minus_i * user_tau_Y)

        jive_cov_SS_list.append(np.mean(jive_products_S))
        jive_cov_SY_list.append(np.mean(jive_products_Y))

        # Average over users to get covariances for this experiment
        jive_cov_SS = np.mean(jive_cov_SS_list)
        jive_cov_SY = np.mean(jive_cov_SY_list)

        # JIVE beta estimate
        beta_jive = jive_cov_SY / jive_cov_SS

    return {
        'beta': beta_jive,
        'cov_SS': jive_cov_SS,
        'cov_SY': jive_cov_SY,
        'method': 'Jackknife IV (JIVE)'
           }

#======================================================
# Part 6: LIML ESTIMATOR 
#======================================================

def liml_estimator(effects: pd.DataFrame,
                   noise_cov: np.ndarray,
                   n: int) -> Dict:
    """
    Limited Information Maximum Likelihood (LIML) Estimator.

    Transform data to make noise isotropic, then use smallest eigenvector.
    Assumes: No direct effects (S fully mediates effect on Y).
    parameters:
    effects (pd.DataFrame): DataFrame containing estimated treatment effects.
    noise_cov (np.ndarray): Covariance matrix for noise in S and Y.
    n (int): Sample size per experiment. 
            It is the number of samples (users) in the dataset.

    Returns:
    Dict: Dictionary containing estimated beta and standard error.
    """
    # extracting treatment effects from "effects" DataFrame
    tau_S = effects['tau_S'].values.reshape(-1,1)
    tau_Y = effects['tau_Y'].values.reshape(-1,1)
    tau = np.hstack([tau_Y, tau_S])

    # centering the data around zero
    tau_centered = tau - tau.mean(axis = 0)

    # observed covariance
    K = len(effects)
    # @ performs matrix multiplication
    obs_cov = tau_centered.T @ tau_centered / K

    # Transform by Ω^(-1/2) to make noise isotropic
    # omega_sqrt_inv is the transformation matrix 
    # that makes the noise isotropic (uniform in all directions)
    omega_sqrt_inv = linalg.sqrtm(linalg.inv(noise_cov))
    transformed_cov = omega_sqrt_inv @ obs_cov @ omega_sqrt_inv.T
    # ensures that the noise is accounted for in the covariance structure

    # find smallest eigenvector
    # it is the smallest variance in the transformed space
    eigenvalues, eigenvectors = linalg.eig(transformed_cov)
    smallest_idx = np.argmin(np.real(eigenvalues))
    gamma = np.real(eigenvectors[:, smallest_idx])

    # Transform back
    gamma_original = omega_sqrt_inv @ gamma

    # Extract beta: [1, -beta]^T is the eigenvector
    beta_liml = -gamma_original[1]/ gamma_original[0]

    return {
        'beta': beta_liml, # LIML estimate of the causal effect
        'cov_SS': obs_cov[1,1], # variance of tau_S
        'cov_SY': obs_cov[0,1], # covariance between tau_S and tau_Y
        'eigenvalue': np.real(eigenvalues[smallest_idx]), #smallest eigenvalue - transformed covariance matrix
        'method': 'LIML'
    }

#======================================================
# PART 7: GROUND TRUTH (from True Effects)
# ======================================================

def ground_truth_beta(effects: pd.DataFrame) -> Dict:
    """
    Calculate beta from TRUE treatment effects not estimated effects (not available in practice).
    This is our benchmark for comparison.
    """

    true_tau_S = effects['true_tau_S'].values
    true_tau_Y = effects['true_tau_Y'].values

    # Center the true effects
    true_tau_S_centered = true_tau_S - true_tau_S.mean()
    true_tau_Y_centered = true_tau_Y - true_tau_Y.mean()

    true_cov_SS = np.mean(true_tau_S_centered ** 2)
    true_cov_SY = np.mean(true_tau_S_centered * true_tau_Y_centered)

    beta_true = true_cov_SY / true_cov_SS

    return {
        'beta': beta_true,
        'cov_SS': true_cov_SS,
        'cov_SY': true_cov_SY,  
        'method': 'Ground Truth'
    }

# ======================================================
# PART 8: VISUALIZATION
# ======================================================

def plot_treatment_effects(effects: pd.DataFrame, 
                           results: Dict,
                           title: str = "Treatment Effect Scatterplot"):
    """
    Visualize treatment effects and regression lines from different methods.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    methods = [
        ('true_tau_S', 'true_tau_Y', 'Ground Truth', 'green'),
        ('tau_S', 'tau_Y', 'Observed (Naive)', 'red'),
        ('tau_S', 'tau_Y', 'TC-Corrected', 'blue')
    ]
    
    for idx, (x_col, y_col, label, color) in enumerate(methods):
        ax = axes[idx]
        
        # Scatter plot
        ax.scatter(effects[x_col], effects[y_col], alpha=0.5, s=30)
        
        # Regression line
        x_range = np.array([effects[x_col].min(), effects[x_col].max()])
        
        if label == 'Ground Truth':
            beta = results['ground_truth']['beta']
        elif label == 'Observed (Naive)':
            beta = results['naive']['beta']
        else:
            beta = results['tc']['beta']
        
        ax.plot(x_range, beta * x_range, color=color, linewidth=2, 
                label=f'β = {beta:.3f}')
        
        ax.set_xlabel('Treatment Effect on S', fontsize=11)
        ax.set_ylabel('Treatment Effect on Y', fontsize=11)
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('treatment_effects_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_bias_vs_sample_size(generator: ABTestDataGenerator,
                             sample_sizes: list):
    """
    Show how bias changes with experiment sample size.
    """
    biases_naive = []
    biases_tc = []
    
    true_beta = generator.beta
    noise_cov = generator.get_noise_covariance()
    
    for n in sample_sizes:
        # Generate data with this sample size
        gen_temp = ABTestDataGenerator(
            num_experiments=generator.K,
            users_per_experiment=n,
            true_beta=true_beta,
            noise_correlation=generator.rho,
            signal_strength=generator.signal
        )
        
        data = gen_temp.generate_experiments()
        effects = calculate_treatment_effects(data)
        
        naive_result = naive_ols_estimator(effects)
        tc_result = tc_estimator(effects, noise_cov, n)
        
        biases_naive.append(abs(naive_result['beta'] - true_beta))
        biases_tc.append(abs(tc_result['beta'] - true_beta))
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, biases_naive, 'o-', color='red', 
             linewidth=2, markersize=8, label='Naive OLS')
    plt.plot(sample_sizes, biases_tc, 's-', color='blue', 
             linewidth=2, markersize=8, label='TC-Corrected')
    plt.axhline(0, color='green', linestyle='--', alpha=0.5, label='Zero Bias')
    
    plt.xlabel('Users per Experiment', fontsize=12)
    plt.ylabel('Absolute Bias in β', fontsize=12)
    plt.title('Estimation Bias vs Sample Size', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('bias_vs_sample_size.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_results_table(results: Dict) -> pd.DataFrame:
    """
    Create a summary table comparing all methods.
    """
    table_data = []
    
    for method_name, result in results.items():
        table_data.append({
            'Method': result['method'],
            'Estimated β': f"{result['beta']:.4f}",
            'Bias': f"{abs(result['beta'] - results['ground_truth']['beta']):.4f}",
            'Bias %': f"{abs(result['beta'] - results['ground_truth']['beta']) / results['ground_truth']['beta'] * 100:.1f}%"
        })
    
    return pd.DataFrame(table_data)

#==============================================================================
# PART 9: MAIN ANALYSIS PIPELINE
#==============================================================================

def run_complete_analysis():
    """
    Execute the complete analysis pipeline.
    """
    print("="*80)
    print("PORTFOLIO PROJECT: Learning Treatment Effect Covariances")
    print("From Naive to Robust Methods")
    print("="*80)
    
    # Step 1: Generate data
    print("\n[1/6] Generating synthetic A/B test data...")
    generator = ABTestDataGenerator(
        num_experiments=100,
        users_per_experiment=10000,
        true_beta=0.25,
        noise_correlation=0.7,
        signal_strength=0.5
    )
    
    data = generator.generate_experiments()
    print(f"   Generated {len(data):,} user observations across {generator.K} experiments")
    
    # Step 2: Calculate treatment effects
    print("\n[2/6] Calculating treatment effects per experiment...")
    effects = calculate_treatment_effects(data)
    print(f"   Calculated effects for {len(effects)} experiments")
    
    # Step 3: Get noise covariance (in practice, estimated from historical data)
    noise_cov = generator.get_noise_covariance()
    print(f"\n[3/6] Noise covariance matrix:")
    print(noise_cov)
    
    # Step 4: Apply all methods
    print("\n[4/6] Applying estimation methods...")
    results = {}
    
    results['ground_truth'] = ground_truth_beta(effects)
    print(f"   ✓ Ground Truth: β = {results['ground_truth']['beta']:.4f}")
    
    results['naive'] = naive_ols_estimator(effects)
    print(f"   ✓ Naive OLS: β = {results['naive']['beta']:.4f}")
    
    results['tc'] = tc_estimator(effects, noise_cov, generator.n)
    print(f"   ✓ TC Estimator: β = {results['tc']['beta']:.4f}")
    
    print("   ⏳ JIVE (this may take a minute)...")
    results['jive'] = jive_estimator(data)
    print(f"   ✓ JIVE: β = {results['jive']['beta']:.4f}")
    
    results['liml'] = liml_estimator(effects, noise_cov, generator.n)
    print(f"   ✓ LIML: β = {results['liml']['beta']:.4f}")
    
    # Step 5: Create comparison table
    print("\n[5/6] Creating results summary...")
    results_table = create_results_table(results)
    print("\n" + results_table.to_string(index=False))
    
    # Step 6: Generate visualizations
    print("\n[6/6] Generating visualizations...")
    plot_treatment_effects(effects, results)
    
    sample_sizes = [1000, 2500, 5000, 10000, 25000, 50000]
    plot_bias_vs_sample_size(generator, sample_sizes)
    
    print("\n" + "="*80)
    print("Analysis complete! Check the generated PNG files.")
    print("="*80)
    
    return data, effects, results

# Run the analysis
if __name__ == "__main__":
    data, effects, results = run_complete_analysis()
                                                    