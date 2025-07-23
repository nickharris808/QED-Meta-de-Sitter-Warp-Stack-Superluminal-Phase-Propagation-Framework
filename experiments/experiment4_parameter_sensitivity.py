#!/usr/bin/env python3
"""
Experiment 4: Parameter-Sensitivity (Tornado Chart)
===================================================

Goal: Identify which physical parameter dominates the arrival-time uncertainty
and provide guidance for experimental optimization priorities.

This analysis helps focus experimental effort on the most critical control 
parameters for achieving reproducible superluminal phase propagation.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import itertools


def early_arrival(L, c, dn, lens_frac, qed_field=0, warp_amp=0):
    """
    Enhanced analytic formula including all warp-stack contributions.
    
    Parameters
    ----------
    L : float
        Total corridor length (m)
    c : float
        Speed of light (m/s)
    dn : float
        Primary refractive index perturbation
    lens_frac : float
        Fraction of corridor with metamaterial
    qed_field : float
        QED field strength contribution
    warp_amp : float
        de-Sitter warp amplitude
        
    Returns
    -------
    float
        Early arrival time (seconds)
    """
    # QED vacuum birefringence contribution (CORRECTED)
    if qed_field > 0:
        alpha = 1/137
        # Use proper Schwinger critical field
        from scipy import constants
        E_crit = constants.m_e**2 * constants.c**3 / (constants.e * constants.hbar)  # V/m
        field_ratio = qed_field / E_crit
        dn_qed = (2 * alpha**2 / (45 * np.pi)) * field_ratio**2
    else:
        dn_qed = 0
    
    # Total refractive index change
    dn_total = dn + dn_qed + warp_amp
    
    return (L / c) * (-dn_total * lens_frac)


def sensitivity(base_params, param_name, delta_frac=0.05):
    """
    Calculate sensitivity ∂Δt/∂p for a single parameter.
    
    Parameters
    ----------
    base_params : dict
        Baseline parameter values
    param_name : str
        Name of parameter to vary
    delta_frac : float
        Fractional perturbation size
        
    Returns
    -------
    float
        Sensitivity coefficient ∂Δt/∂p
    """
    base_value = base_params[param_name]
    
    # Upper perturbation
    params_up = base_params.copy()
    params_up[param_name] = base_value * (1 + delta_frac)
    advance_up = early_arrival(**params_up)
    
    # Lower perturbation
    params_dn = base_params.copy()
    params_dn[param_name] = base_value * (1 - delta_frac)
    advance_dn = early_arrival(**params_dn)
    
    # Central difference
    return (advance_up - advance_dn) / (2 * base_value * delta_frac)


def sensitivity_analysis(base_params, param_list, delta_frac=0.05):
    """
    Calculate sensitivities for all parameters.
    
    Parameters
    ----------
    base_params : dict
        Baseline parameter values
    param_list : list
        List of parameter names to analyze
    delta_frac : float
        Fractional perturbation size
        
    Returns
    -------
    dict
        Dictionary of sensitivity coefficients
    """
    sensitivities = {}
    
    for param in param_list:
        sens = sensitivity(base_params, param, delta_frac)
        sensitivities[param] = sens
    
    return sensitivities


def parameter_sweep_2d(base_params, param1, param2, ranges, n_points=50):
    """
    2D parameter sweep for contour analysis.
    
    Parameters
    ----------
    base_params : dict
        Baseline parameters
    param1, param2 : str
        Names of parameters to sweep
    ranges : dict
        Ranges for each parameter {param: (min, max)}
    n_points : int
        Number of points per dimension
        
    Returns
    -------
    tuple
        (X, Y, Z) meshgrids for contour plotting
    """
    p1_range = np.linspace(ranges[param1][0], ranges[param1][1], n_points)
    p2_range = np.linspace(ranges[param2][0], ranges[param2][1], n_points)
    
    X, Y = np.meshgrid(p1_range, p2_range)
    Z = np.zeros_like(X)
    
    for i in range(n_points):
        for j in range(n_points):
            params = base_params.copy()
            params[param1] = X[i, j]
            params[param2] = Y[i, j]
            Z[i, j] = early_arrival(**params) * 1e12  # Convert to ps
    
    return X, Y, Z


def run_experiment():
    """
    Main experiment: Comprehensive parameter sensitivity analysis.
    """
    print("🌪️  Experiment 4: Parameter-Sensitivity (Tornado Chart)")
    print("=" * 55)
    
    # Enhanced baseline parameters
    base_params = {
        'L': 1200.0,              # Corridor length (m)
        'c': 299_792_458.0,       # Speed of light (m/s)
        'dn': -2.2e-6,            # Primary metamaterial Δn
        'lens_frac': 0.4,         # Metamaterial fraction
        'qed_field': 2e13,        # QED field strength (V/m)
        'warp_amp': -1e-7         # de-Sitter amplitude
    }
    
    # Parameters to analyze
    param_list = ['L', 'dn', 'lens_frac', 'qed_field', 'warp_amp']
    param_labels = {
        'L': 'Corridor Length',
        'dn': 'Metamaterial Δn', 
        'lens_frac': 'Lens Fraction',
        'qed_field': 'QED Field',
        'warp_amp': 'Warp Amplitude'
    }
    
    print(f"📏 Baseline corridor: {base_params['L']} m")
    print(f"🔬 Baseline Δn: {base_params['dn']:.2e}")
    print(f"🎯 Baseline lens fraction: {base_params['lens_frac']:.1%}")
    print(f"⚡ QED field: {base_params['qed_field']:.1e} V/m")
    print(f"🌌 Warp amplitude: {base_params['warp_amp']:.1e}")
    
    # Calculate baseline early arrival
    baseline_advance = early_arrival(**base_params) * 1e12
    print(f"\n📊 Baseline early arrival: {baseline_advance:.3f} ps")
    
    # Sensitivity analysis
    print(f"\n🔍 SENSITIVITY ANALYSIS:")
    sensitivities = sensitivity_analysis(base_params, param_list)
    
    # Convert to ps and sort by absolute value
    sens_ps = {k: v * 1e12 for k, v in sensitivities.items()}
    sorted_sens = sorted(sens_ps.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print(f"   Parameter sensitivities (ps per unit change):")
    for param, sens_val in sorted_sens:
        print(f"   {param_labels[param]:18s}: {sens_val:+.2e} ps/unit")
    
    # Relative importance analysis
    total_abs_sens = sum(abs(v) for v in sens_ps.values())
    rel_importance = {k: abs(v)/total_abs_sens*100 for k, v in sens_ps.items()}
    
    print(f"\n📊 RELATIVE IMPORTANCE:")
    for param, importance in sorted(rel_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"   {param_labels[param]:18s}: {importance:5.1f}%")
    
    # Uncertainty propagation analysis
    # Typical fabrication uncertainties
    uncertainties = {
        'L': 5.0,              # ±5 m length uncertainty
        'dn': 0.1e-6,          # ±0.1×10⁻⁶ index uncertainty  
        'lens_frac': 0.02,     # ±2% fraction uncertainty
        'qed_field': 1e12,     # ±1×10¹² V/m field uncertainty
        'warp_amp': 0.1e-7     # ±10% warp uncertainty
    }
    
    print(f"\n🎯 UNCERTAINTY CONTRIBUTIONS:")
    variance_contributions = {}
    total_variance = 0
    
    for param in param_list:
        # Variance contribution = (sensitivity × uncertainty)²
        var_contrib = (sensitivities[param] * uncertainties[param] * 1e12)**2
        variance_contributions[param] = var_contrib
        total_variance += var_contrib
        
    total_std = np.sqrt(total_variance)
    
    for param in sorted(variance_contributions.keys(), 
                       key=lambda k: variance_contributions[k], reverse=True):
        contribution_pct = variance_contributions[param] / total_variance * 100
        contribution_std = np.sqrt(variance_contributions[param])
        print(f"   {param_labels[param]:18s}: {contribution_std:6.3f} ps ({contribution_pct:4.1f}%)")
    
    print(f"   {'Total uncertainty':18s}: {total_std:6.3f} ps")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Tornado chart (sensitivities)
    ax1 = plt.subplot(2, 3, 1)
    labels = [param_labels[k] for k, v in sorted_sens]
    values = [v for k, v in sorted_sens]
    colors = ['red' if v < 0 else 'blue' for v in values]
    
    bars = ax1.barh(range(len(labels)), values, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels)
    ax1.set_xlabel('Sensitivity (ps per unit parameter)')
    ax1.set_title('Parameter Sensitivity Tornado Chart')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, values)):
        width = bar.get_width()
        ax1.text(width + np.sign(width) * 0.02 * max(np.abs(values)), 
                bar.get_y() + bar.get_height()/2, 
                f'{value:.1e}', ha='left' if width > 0 else 'right', va='center', fontsize=8)
    
    # Plot 2: Relative importance pie chart
    ax2 = plt.subplot(2, 3, 2)
    importance_values = [rel_importance[k] for k, v in sorted_sens]
    importance_labels = [param_labels[k] for k, v in sorted_sens]
    
    wedges, texts, autotexts = ax2.pie(importance_values, labels=importance_labels, 
                                      autopct='%1.1f%%', startangle=90)
    ax2.set_title('Relative Parameter Importance')
    
    # Plot 3: Uncertainty contributions
    ax3 = plt.subplot(2, 3, 3)
    unc_labels = [param_labels[k] for k in sorted(variance_contributions.keys(), 
                                                 key=lambda k: variance_contributions[k], reverse=True)]
    unc_values = [np.sqrt(variance_contributions[k]) for k in sorted(variance_contributions.keys(), 
                                                                   key=lambda k: variance_contributions[k], reverse=True)]
    
    bars = ax3.bar(range(len(unc_labels)), unc_values, alpha=0.7, color='green')
    ax3.set_xticks(range(len(unc_labels)))
    ax3.set_xticklabels(unc_labels, rotation=45, ha='right')
    ax3.set_ylabel('Uncertainty Contribution (ps)')
    ax3.set_title('Uncertainty Budget')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: 2D parameter sweep (most important parameters)
    ax4 = plt.subplot(2, 3, 4)
    top_params = [k for k, v in sorted_sens[:2]]  # Two most important parameters
    
    if len(top_params) >= 2:
        param1, param2 = top_params[0], top_params[1]
        
        # Define sweep ranges (±20% around baseline)
        ranges = {}
        for p in [param1, param2]:
            base_val = base_params[p]
            ranges[p] = (base_val * 0.8, base_val * 1.2)
        
        X, Y, Z = parameter_sweep_2d(base_params, param1, param2, ranges, n_points=30)
        
        contour = ax4.contour(X, Y, Z, levels=15, colors='black', alpha=0.4, linewidths=0.5)
        contour_f = ax4.contourf(X, Y, Z, levels=15, cmap='RdYlBu_r', alpha=0.8)
        ax4.plot(base_params[param1], base_params[param2], 'ko', markersize=8, 
                label=f'Baseline ({baseline_advance:.3f} ps)')
        
        ax4.set_xlabel(param_labels[param1])
        ax4.set_ylabel(param_labels[param2])
        ax4.set_title(f'Early Arrival vs {param_labels[param1]} & {param_labels[param2]}')
        ax4.legend()
        
        # Add colorbar
        cbar = plt.colorbar(contour_f, ax=ax4)
        cbar.set_label('Early Arrival (ps)')
    
    # Plot 5: Parameter correlation matrix (simplified)
    ax5 = plt.subplot(2, 3, 5)
    
    # Calculate correlation matrix based on sensitivities
    sens_array = np.array([sensitivities[p] for p in param_list])
    corr_matrix = np.outer(sens_array, sens_array) / (np.outer(np.abs(sens_array), np.abs(sens_array)) + 1e-12)
    
    im = ax5.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax5.set_xticks(range(len(param_list)))
    ax5.set_yticks(range(len(param_list)))
    ax5.set_xticklabels([param_labels[p] for p in param_list], rotation=45, ha='right')
    ax5.set_yticklabels([param_labels[p] for p in param_list])
    ax5.set_title('Parameter Sensitivity Correlation')
    
    # Add text annotations
    for i in range(len(param_list)):
        for j in range(len(param_list)):
            text = ax5.text(j, i, f'{corr_matrix[i, j]:.2f}', 
                           ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=ax5)
    
    # Plot 6: Optimization landscape (1D cuts)
    ax6 = plt.subplot(2, 3, 6)
    
    # Show how early arrival varies with most important parameter
    most_important = sorted_sens[0][0]
    base_val = base_params[most_important]
    param_range = np.linspace(base_val * 0.5, base_val * 1.5, 100)
    
    advances = []
    for val in param_range:
        params = base_params.copy()
        params[most_important] = val
        advances.append(early_arrival(**params) * 1e12)
    
    ax6.plot(param_range, advances, 'b-', linewidth=2)
    ax6.axvline(base_val, color='red', linestyle='--', 
               label=f'Baseline ({baseline_advance:.3f} ps)')
    ax6.axhline(baseline_advance, color='red', linestyle='--', alpha=0.5)
    
    ax6.set_xlabel(param_labels[most_important])
    ax6.set_ylabel('Early Arrival (ps)')
    ax6.set_title(f'Optimization Landscape: {param_labels[most_important]}')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment4_parameter_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show to avoid display issues
    
    # Success criteria and recommendations
    print(f"\n✅ SUCCESS CRITERIA & RECOMMENDATIONS:")
    
    # Identify dominant parameter
    dominant_param = sorted_sens[0][0]
    dominant_importance = rel_importance[dominant_param]
    
    print(f"   🎯 DOMINANT PARAMETER: {param_labels[dominant_param]} ({dominant_importance:.1f}% of total)")
    
    if dominant_importance > 50:
        print(f"   ✅ CLEAR OPTIMIZATION TARGET: Focus on {param_labels[dominant_param]} control")
        print(f"   📊 Single-parameter dominance simplifies experimental design")
    else:
        print(f"   ⚠️  Multi-parameter system: Top 2-3 parameters need simultaneous control")
    
    # Control precision recommendations
    top_contributors = sorted(variance_contributions.items(), 
                            key=lambda x: x[1], reverse=True)[:3]
    
    print(f"\n🔧 CONTROL PRECISION RECOMMENDATIONS:")
    for param, var_contrib in top_contributors:
        current_unc = uncertainties[param]
        contrib_std = np.sqrt(var_contrib)
        
        # Calculate required precision for 1% total uncertainty
        target_total_std = baseline_advance * 0.01  # 1% target
        if contrib_std > target_total_std:
            required_unc = current_unc * (target_total_std / contrib_std)
            improvement_factor = current_unc / required_unc
            
            print(f"   {param_labels[param]:18s}: Improve by {improvement_factor:.1f}× " +
                  f"(current: ±{current_unc:.2e}, target: ±{required_unc:.2e})")
        else:
            print(f"   {param_labels[param]:18s}: Current precision sufficient")
    
    return sorted_sens, total_std, dominant_param


if __name__ == "__main__":
    sensitivities, total_uncertainty, dominant = run_experiment()
    
    print(f"\n🎯 EXPERIMENT 4 COMPLETE")
    print(f"   Most sensitive parameter: {dominant}")
    print(f"   Total predicted uncertainty: {total_uncertainty:.3f} ps")
    print(f"   Control recommendation: Focus on {dominant} precision")
    print(f"   💡 Provides clear experimental optimization roadmap!") 