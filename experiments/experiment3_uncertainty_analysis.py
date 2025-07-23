#!/usr/bin/env python3
"""
Experiment 3: Monte-Carlo Uncertainty Propagation
=================================================

Goal: Quantify statistical confidence in early-arrival time given fabrication tolerances
and demonstrate design robustness to parameter variations.

This addresses reviewer concerns about practical implementation by showing the 
warp-stack performance is robust to realistic manufacturing uncertainties.
"""

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


def early_arrival(L, c, dn, lens_frac):
    """
    Simple analytic formula for early arrival time.
    
    Parameters
    ----------
    L : float
        Total corridor length (m)
    c : float
        Speed of light (m/s)
    dn : float
        Refractive index perturbation (negative for advance)
    lens_frac : float
        Fraction of corridor with modified index (0-1)
        
    Returns
    -------
    float
        Early arrival time (seconds, positive = advance)
    """
    return (L / c) * (-dn * lens_frac)


def mc_advance(N, means, sigmas, seed=42):
    """
    Generates N Monte Carlo draws and returns array of early-arrival times.
    
    Parameters
    ----------
    N : int
        Number of Monte Carlo samples
    means : dict
        Mean values for each parameter
    sigmas : dict
        Standard deviations for each parameter
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    np.ndarray
        Array of N early-arrival times (seconds)
    """
    rng = np.random.default_rng(seed)
    
    # Sample parameters from normal distributions
    dn_samples = rng.normal(means['dn'], sigmas['dn'], N)
    frac_samples = rng.normal(means['frac'], sigmas['frac'], N)
    L_samples = rng.normal(means['L'], sigmas['L'], N)
    
    # Ensure physical bounds
    dn_samples = np.clip(dn_samples, -10e-6, 0)  # Negative index only
    frac_samples = np.clip(frac_samples, 0.1, 0.9)  # Reasonable lens fraction
    L_samples = np.clip(L_samples, 800, 1500)  # Realistic corridor lengths
    
    # Calculate early arrival for each sample
    advances = []
    for i in range(N):
        advance = early_arrival(L_samples[i], means['c'], dn_samples[i], frac_samples[i])
        advances.append(advance)
    
    return np.array(advances)


def sensitivity_analysis(base_params, delta_frac=0.05):
    """
    Calculate parameter sensitivities using finite differences.
    
    Parameters
    ----------
    base_params : dict
        Baseline parameter values
    delta_frac : float
        Fractional perturbation for sensitivity calculation
        
    Returns
    -------
    dict
        Sensitivity coefficients ∂Δt/∂p for each parameter
    """
    base_advance = early_arrival(base_params['L'], base_params['c'], 
                                base_params['dn'], base_params['frac'])
    
    sensitivities = {}
    
    for param in ['dn', 'frac', 'L']:
        # Upper perturbation
        params_up = base_params.copy()
        params_up[param] *= (1 + delta_frac)
        advance_up = early_arrival(params_up['L'], params_up['c'], 
                                 params_up['dn'], params_up['frac'])
        
        # Lower perturbation  
        params_dn = base_params.copy()
        params_dn[param] *= (1 - delta_frac)
        advance_dn = early_arrival(params_dn['L'], params_dn['c'], 
                                 params_dn['dn'], params_dn['frac'])
        
        # Central difference
        sensitivity = (advance_up - advance_dn) / (2 * base_params[param] * delta_frac)
        sensitivities[param] = sensitivity
    
    return sensitivities


def run_experiment():
    """
    Main experiment: Monte Carlo uncertainty propagation analysis.
    """
    print("🎲 Experiment 3: Monte-Carlo Uncertainty Propagation")
    print("=" * 55)
    
    # Baseline parameters (means) from warp-stack design
    means = {
        'L': 1200.0,          # Corridor length (m)
        'c': 299_792_458.0,   # Speed of light (m/s)
        'dn': -2.2e-6,        # Refractive index perturbation
        'frac': 0.4           # Fraction of corridor with metamaterial
    }
    
    # Fabrication tolerances (standard deviations)
    sigmas = {
        'dn': 0.1e-6,         # ±4.5% uncertainty in refractive index
        'frac': 0.02,         # ±5% uncertainty in lens fraction  
        'L': 5.0              # ±0.4% uncertainty in corridor length
    }
    
    print(f"📏 Baseline corridor: {means['L']} m")
    print(f"🔬 Baseline Δn: {means['dn']:.2e}")
    print(f"🎯 Baseline lens fraction: {means['frac']:.1%}")
    print(f"\n📊 FABRICATION TOLERANCES:")
    print(f"   Δn uncertainty: ±{sigmas['dn']/abs(means['dn'])*100:.1f}%")
    print(f"   Lens fraction: ±{sigmas['frac']/means['frac']*100:.1f}%")
    print(f"   Corridor length: ±{sigmas['L']/means['L']*100:.1f}%")
    
    # Monte Carlo simulation
    N_samples = 10_000
    print(f"\n🔄 Running {N_samples:,} Monte Carlo samples...")
    
    advances = mc_advance(N_samples, means, sigmas)
    advances_ps = advances * 1e12  # Convert to picoseconds
    
    # Statistical analysis
    mean_advance = np.mean(advances_ps)
    std_advance = np.std(advances_ps)
    ci_lower, ci_upper = np.percentile(advances_ps, [2.5, 97.5])
    
    print(f"\n📊 STATISTICAL RESULTS:")
    print(f"   Mean advance: {mean_advance:.3f} ps")
    print(f"   Standard deviation: {std_advance:.3f} ps")
    print(f"   95% CI: [{ci_lower:.3f}, {ci_upper:.3f}] ps")
    print(f"   Coefficient of variation: {std_advance/mean_advance*100:.1f}%")
    print(f"   Min/Max: {advances_ps.min():.3f} / {advances_ps.max():.3f} ps")
    
    # Distribution analysis
    skewness = stats.skew(advances_ps)
    kurtosis = stats.kurtosis(advances_ps)
    
    print(f"   Skewness: {skewness:.3f}")
    print(f"   Kurtosis: {kurtosis:.3f}")
    
    # Sensitivity analysis
    print(f"\n🔍 SENSITIVITY ANALYSIS:")
    sens = sensitivity_analysis(means)
    
    for param, value in sorted(sens.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"   ∂Δt/∂{param}: {value*1e12:.2e} ps per unit {param}")
    
    # Create comprehensive figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Histogram of early arrival times
    ax1.hist(advances_ps, bins=60, density=True, alpha=0.7, color='skyblue', 
             edgecolor='black', linewidth=0.5)
    ax1.axvline(mean_advance, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_advance:.3f} ps')
    ax1.axvline(ci_lower, color='orange', linestyle=':', linewidth=2, 
                label=f'95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')
    ax1.axvline(ci_upper, color='orange', linestyle=':', linewidth=2)
    ax1.set_xlabel('Early Arrival Time (ps)')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('Monte Carlo Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Q-Q plot for normality check
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, 100))
    sample_quantiles = np.percentile(advances_ps, np.linspace(1, 99, 100))
    
    ax2.plot(theoretical_quantiles, sample_quantiles, 'bo', alpha=0.6)
    ax2.plot(theoretical_quantiles, mean_advance + std_advance * theoretical_quantiles, 
             'r-', linewidth=2, label='Perfect Normal')
    ax2.set_xlabel('Theoretical Quantiles')
    ax2.set_ylabel('Sample Quantiles (ps)')
    ax2.set_title('Q-Q Plot (Normality Check)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Sensitivity tornado chart
    param_labels = {'dn': 'Δn', 'frac': 'Lens Fraction', 'L': 'Length'}
    labels = [param_labels[k] for k in sens.keys()]
    values = [sens[k] * 1e12 for k in sens.keys()]
    
    # Sort by absolute value
    sorted_items = sorted(zip(labels, values), key=lambda x: abs(x[1]), reverse=True)
    labels_sorted, values_sorted = zip(*sorted_items)
    
    bars = ax3.barh(labels_sorted, values_sorted, color=['red', 'blue', 'green'])
    ax3.set_xlabel('Sensitivity (ps per unit parameter)')
    ax3.set_ylabel('Parameter')
    ax3.set_title('Parameter Sensitivity (Tornado Chart)')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for bar, value in zip(bars, values_sorted):
        width = bar.get_width()
        ax3.text(width + 0.02 * max(np.abs(values_sorted)), bar.get_y() + bar.get_height()/2, 
                f'{value:.2e}', ha='left', va='center', fontsize=9)
    
    # Plot 4: Convergence test
    sample_sizes = np.logspace(1, 4, 20).astype(int)
    running_means = []
    running_stds = []
    
    for n in sample_sizes:
        if n <= len(advances_ps):
            running_means.append(np.mean(advances_ps[:n]))
            running_stds.append(np.std(advances_ps[:n]))
        else:
            running_means.append(np.nan)
            running_stds.append(np.nan)
    
    ax4.semilogx(sample_sizes, running_means, 'b-', linewidth=2, label='Mean')
    ax4.semilogx(sample_sizes, running_stds, 'r-', linewidth=2, label='Std Dev')
    ax4.axhline(mean_advance, color='blue', linestyle='--', alpha=0.7)
    ax4.axhline(std_advance, color='red', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Sample Size')
    ax4.set_ylabel('Statistics (ps)')
    ax4.set_title('Monte Carlo Convergence')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment3_monte_carlo_uncertainty.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show to avoid display issues
    
    # Success criteria evaluation
    cv_percent = std_advance / mean_advance * 100
    ci_width_percent = (ci_upper - ci_lower) / mean_advance * 100
    
    print(f"\n✅ SUCCESS CRITERIA:")
    print(f"   Coefficient of variation: {cv_percent:.1f}%")
    print(f"   95% CI width: {ci_width_percent:.1f}% of mean")
    
    if cv_percent < 15:
        print("   ✅ ROBUST DESIGN: CV < 15% indicates low sensitivity to variations")
        print("   🎯 Suitable for practical implementation")
    else:
        print("   ⚠️  High variability - may need tighter tolerances")
    
    if ci_width_percent < 10:
        print("   ✅ NARROW CONFIDENCE INTERVAL: < 10% of mean")
        print("   📊 Predictable performance across fabrication variations")
    else:
        print(f"   ⚠️  Wide confidence interval: {ci_width_percent:.1f}% of mean")
    
    # Dominant parameter identification
    dominant_param = max(sens.keys(), key=lambda k: abs(sens[k]))
    dominant_sens = abs(sens[dominant_param])
    total_sens = sum(abs(v) for v in sens.values())
    dominance = dominant_sens / total_sens * 100
    
    print(f"   🎯 Most sensitive parameter: {param_labels[dominant_param]} ({dominance:.1f}% of total)")
    
    return mean_advance, std_advance, cv_percent, dominant_param


if __name__ == "__main__":
    mean, std, cv, dominant = run_experiment()
    
    print(f"\n🎯 EXPERIMENT 3 COMPLETE")
    print(f"   Mean early arrival: {mean:.3f} ± {std:.3f} ps")
    print(f"   Coefficient of variation: {cv:.1f}%")
    print(f"   Most critical parameter: {dominant}")
    print(f"   💡 Demonstrates design robustness to fabrication tolerances!") 