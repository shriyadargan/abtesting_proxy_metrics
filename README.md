# abtesting_proxy_metrics
When running many A/B tests with weak treatment effects, naive regression-based methods for estimating relationships between metrics produce biased results due to correlated user-level noise. This project implements three sophisticated methods (JIVE, LIML, TC) that provide unbiased estimates using a synthetic dataset.

A complete portfolio project demonstrating advanced A/B testing methodologies for handling weak treatment effects and correlated measurement noise, inspired by Netflix's KDD 2024 paper.

## Project Overview

**Problem Statement**: When running many A/B tests with weak treatment effects (common in tech), naive methods for estimating relationships between metrics produce biased results due to correlated user-level noise.

**Solution**: Implement three sophisticated methods (JIVE, LIML, TC) that provide unbiased estimates even when:
- Treatment effects are small
- User-level metrics are correlated
- Each experiment has limited statistical power

## Project Structure

```
ab-test-covariance-analysis/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── notebooks/
│   ├── 01_introduction.ipynb          # Problem setup & motivation
│   ├── 02_data_generation.ipynb      # Creating synthetic data
│   ├── 03_naive_approach.ipynb       # Demonstrating the problem
│   ├── 04_tc_estimator.ipynb         # Total Covariance method
│   ├── 05_jive_estimator.ipynb       # Jackknife IV method
│   ├── 06_liml_estimator.ipynb       # LIML method
│   └── 07_comparison.ipynb           # Full comparison & results
│
├── src/
│   ├── __init__.py
│   ├── data_generator.py              # Synthetic data generation
│   ├── estimators.py                  # All estimation methods
│   ├── visualizations.py              # Plotting functions
│   └── utils.py                       # Helper functions
│
├── data/
│   ├── synthetic_experiments.csv      # Generated dataset
│   └── experiment_effects.csv         # Computed treatment effects
│
├── figures/
│   ├── treatment_effects_comparison.png
│   ├── bias_vs_sample_size.png
│   └── method_comparison_table.png
│
└── presentation/
    └── project_slides.pdf             # Executive summary
```

##  Dataset Description

The synthetic dataset mimics real-world A/B testing scenarios:

**Scale**:
- 100 historical experiments
- 10,000 users per experiment
- 1 million total observations

**Metrics**:
- `S`: Short-term metric (e.g., watch time in week 1)
- `Y`: Long-term metric (e.g., 6-month retention)
- Correlated at user level (ρ = 0.6-0.7)

**Characteristics**:
- Weak treatment effects (signal-to-noise ratio < 0.1)
- True relationship: Y = β × S where β = 0.25
- Realistic noise structure mimicking tech company data

## 🛠️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/ab-test-covariance-analysis.git
cd ab-test-covariance-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```
## Key Results

| Method | Estimated β | Bias | Computation Time |
|--------|------------|------|------------------|
| Ground Truth | 0.250 | 0.000 | N/A |
| Naive OLS | 0.389 | **+55.6%** | 0.1s |
| TC-Corrected | 0.253 | +1.2% | 0.1s |
| JIVE | 0.248 | -0.8% | 45.3s |
| LIML | 0.252 | +0.8% | 0.2s |

**Key Insight**: With 10K users per experiment and ρ=0.7 noise correlation, naive OLS overestimates β by 56%! TC correction reduces bias to just 1% (From the paper).

Business Implication: The sophisticated methods (TC, JIVE, LIML) become critical when:

 - You can't afford large experiments (cost, time, or risk constraints)
 - Treatment effects are subtle (common for mature products)
 - Metrics have high **user-level correlation** (engagement metrics typically do e.g.  users with a higher click-through rate ( a short term metric S) also tend to have a higher retention (longer-term metric Y)). There are many confounders between S and Y — many of which one can never reliably observe and control for in an experimentation. The reason for this is correlated measurement error — if S and Y are positively correlated in the population, then treatment arms that happen to have more users with high S will also have more users with high Y.

## Methods Explained

### 1. Naive OLS (Baseline)
- **Approach**: Simple regression of estimated effects
- **Problem**: Biased when noise is correlated
- **When to use**: Not in production, but for educational only

### 2. Total Covariance (TC) - **Moderate Computation**
- **Approach**: Subtract noise contribution from observed covariance
- **Formula**: `Λ_true = Σ_observed - (4/n) × Ω`
- **Pros**: Fast, robust to direct effects
- **Cons**: Requires estimating Ω (correlated measurement error_ assumed to be the same across experiments (homogeneous covariances)

### 3. Jackknife IV (JIVE)
- **Approach**: Leave-one-out correlation
- **Pros**: No assumptions about noise
- **Cons**: Computationally expensive

### 4. LIML
- **Approach**: Transform to isotropic noise, find smallest eigenvector
- **Pros**: Most efficient when assumptions hold
- **Cons**: Assumes full mediation (no direct effects between the treatment and Y). LIML is highly sensitive to assumption that S fully mediates all treatment effects on Y.Thus, TC and JIVE are recommended for industry applications. 

## 📝 Notebook Walkthrough

### Part 1: Introduction
- Problem motivation with real-world examples
- Why naive methods fail
- Overview of sophisticated approaches

### Part 2: Data Generation
- Simulating realistic A/B test environments
- Understanding correlated noise
- Visualizing the challenge

### Part 3: Naive Approach
- Implementing standard OLS
- Demonstrating bias empirically
- Quantifying the problem

### Part 4: TC Estimator
- Theory: Total variance formula
- Implementation step-by-step
- Results and validation

### Part 5: JIVE Estimator
- Jackknife methodology
- Computational considerations
- Performance comparison

### Part 6: LIML Estimator
- Eigenvalue decomposition approach
- Transforming to isotropic noise
- Visual intuition

### Part 7: Final Comparison
- All methods side-by-side
- Sensitivity analysis
- Practical recommendations

## Key Visualizations

1. **Treatment Effect Scatterplot**: Shows true vs. observed effects
2. **Bias vs Sample Size**: How bias decreases with larger experiments
3. **Method Comparison Dashboard**: Interactive comparison of all methods
4. **Noise Visualization**: Understanding correlated measurement error

##  Business Applications

This methodology is valuable for:

**Tech Companies**:
- Optimizing short-term engagement proxies for long-term retention
- Building composite metrics from multiple signals
- Making decisions with limited long-term data

**E-commerce**:
- Connecting click behavior to purchase conversion
- Balancing multiple business metrics

**Digital Products**:
- Understanding feature impact on user retention
- Prioritizing development based on proxy metrics


## References

1. Bibaut et al. (2024). "Learning the Covariance of Treatment Effects Across Many Weak Experiments." KDD 2024. Blog post: https://netflixtechblog.com/improve-your-next-experiment-by-learning-better-proxy-metrics-from-past-experiments-64c786c2a3ac 
2. Angrist et al. (1999). "Jackknife Instrumental Variables Estimation."
3. Stock & Yogo (2005). "Testing for Weak Instruments in IV Regression."



5. **Full Stack**: "The project includes data generation, statistical estimation, visualization, and documentation—showing end-to-end data science skills."
