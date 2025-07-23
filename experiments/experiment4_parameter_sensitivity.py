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
    
    # Convert to normalized sensitivities (% change in output per % change in input)
    normalized_sensitivities = {}
    baseline_advance_sec = baseline_advance * 1e-12  # Convert ps back to seconds
    
    for param in param_list:
        # Calculate: (% change in arrival time) / (% change in parameter)
        baseline_val = base_params[param]
        if baseline_val != 0 and baseline_advance_sec != 0:
            # Sensitivity is d(arrival_sec)/d(param)
            # We want: (% change in arrival) per (% change in param)
            # = (d(arrival)/arrival) / (d(param)/param) = (d(arrival)/d(param)) * (param/arrival)
            normalized_sens = sensitivities[param] * baseline_val / baseline_advance_sec * 100
            normalized_sensitivities[param] = normalized_sens
        else:
            normalized_sensitivities[param] = 0
    
    # Convert raw sensitivities to ps and sort by absolute value
    sens_ps = {k: v * 1e12 for k, v in sensitivities.items()}
    sorted_sens = sorted(sens_ps.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print(f"   Parameter sensitivities (ps per unit change):")
    for param, sens_val in sorted_sens:
        print(f"   {param_labels[param]:18s}: {sens_val:+.2e} ps/unit")
    
    print(f"\n   Normalized sensitivities (% change per % parameter change):")
    sorted_norm_sens = sorted(normalized_sensitivities.items(), key=lambda x: abs(x[1]), reverse=True)
    for param, norm_sens in sorted_norm_sens:
        print(f"   {param_labels[param]:18s}: {norm_sens:+6.1f}%")
    
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
    fig = plt.figure(figsize=(14, 10))  # Larger figure for better visibility
    
    # MAIN PLOT: Tornado chart (normalized sensitivities)
    ax1 = plt.subplot(1, 1, 1)
    # Sort normalized sensitivities by absolute value
    sorted_norm_sens = sorted(normalized_sensitivities.items(), key=lambda x: abs(x[1]), reverse=True)
    labels = [param_labels[k] for k, v in sorted_norm_sens]
    values = [v for k, v in sorted_norm_sens]
    colors = ['red' if v < 0 else 'blue' for v in values]
    
    # Create horizontal bar chart
    bars = ax1.barh(range(len(labels)), values, color=colors, alpha=0.7)
    
    # Set y-axis
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=12)
    
    # Set x-axis
    ax1.set_xlabel('Normalized Sensitivity (% change per % parameter change)', fontsize=12, fontweight='bold')
    
    # Set title
    ax1.set_title('Parameter Sensitivity Tornado Chart\n(QED-Meta-de Sitter Warp Stack)', fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Ensure all labels are visible
    ax1.margins(y=0.1)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        width = bar.get_width()
        # Position text label
        if width >= 0:
            x_pos = width + max(abs(v) for v in values) * 0.02
            ha = 'left'
        else:
            x_pos = width - max(abs(v) for v in values) * 0.02
            ha = 'right'
        
        ax1.text(x_pos, bar.get_y() + bar.get_height()/2, 
                f'{value:.1f}%', ha=ha, va='center', fontsize=11, fontweight='bold')
    
    # Adjust layout for better spacing
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(left=0.25, right=0.85, top=0.9, bottom=0.15)
    
    plt.savefig('experiment4_parameter_sensitivity.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
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