#!/usr/bin/env python3
"""
Fixed Tornado Chart - Parameter Sensitivity Analysis
Shows NORMALIZED sensitivities (% change per % parameter change)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Baseline parameters
base_params = {
    'L': 1200.0,           # corridor length (m)
    'dn': -2.2e-6,         # metamaterial index perturbation
    'lens_frac': 0.4,      # lens fraction (40%)
    'qed_field': 2e13,     # QED field strength (V/m)
    'warp_amp': -1e-7      # warp amplitude
}

def early_arrival(L, dn, lens_frac, qed_field=0, warp_amp=0):
    """Calculate early arrival time (seconds)"""
    c = 299792458  # m/s
    
    # Primary metamaterial contribution
    dt_meta = L * lens_frac * dn / c
    
    # QED vacuum birefringence (small contribution)
    alpha = 1/137
    E_crit = 1.32e18  # Schwinger critical field (V/m)
    qed_contrib = 0
    if qed_field > 0:
        qed_contrib = -L * (2*alpha/45) * (qed_field/E_crit)**2 / c
    
    # Warp contribution (small)
    warp_contrib = 0
    if warp_amp != 0:
        warp_contrib = -L * warp_amp / c
    
    return dt_meta + qed_contrib + warp_contrib

# Calculate baseline early arrival
baseline_advance = early_arrival(**base_params)
print(f"Baseline early arrival: {baseline_advance*1e12:.3f} ps")

# Calculate sensitivities (derivatives)
delta = 1e-8  # Small perturbation
sensitivities = {}

for param in ['L', 'dn', 'lens_frac', 'qed_field', 'warp_amp']:
    # Perturb parameter
    params_plus = base_params.copy()
    params_plus[param] *= (1 + delta)
    
    params_minus = base_params.copy()
    params_minus[param] *= (1 - delta)
    
    # Calculate sensitivity
    advance_plus = early_arrival(**params_plus)
    advance_minus = early_arrival(**params_minus)
    
    # d(arrival)/d(param)
    sens = (advance_plus - advance_minus) / (2 * delta * base_params[param])
    sensitivities[param] = sens

# Calculate NORMALIZED sensitivities (% change per % change)
normalized_sensitivities = {}
for param in ['L', 'dn', 'lens_frac', 'qed_field', 'warp_amp']:
    # (% change in arrival) per (% change in parameter)
    baseline_val = base_params[param]
    if baseline_val != 0 and baseline_advance != 0:
        normalized_sens = sensitivities[param] * baseline_val / baseline_advance * 100
        normalized_sensitivities[param] = normalized_sens
    else:
        normalized_sensitivities[param] = 0

# Print values for verification
print("\nNormalized Sensitivities (% change per % parameter change):")
param_labels = {
    'L': 'Corridor Length',
    'dn': 'Metamaterial Δn', 
    'lens_frac': 'Lens Fraction',
    'qed_field': 'QED Field',
    'warp_amp': 'Warp Amplitude'
}

for param, norm_sens in sorted(normalized_sensitivities.items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"   {param_labels[param]:18s}: {norm_sens:+6.1f}%")

# Create tornado chart with better formatting
fig, ax = plt.subplots(figsize=(10, 6))  # Smaller, more reasonable size

# Sort by absolute value
sorted_items = sorted(normalized_sensitivities.items(), key=lambda x: abs(x[1]), reverse=True)
labels = [param_labels[k] for k, v in sorted_items]
values = [v for k, v in sorted_items]

print(f"\nChart will show {len(labels)} parameters:")
for label, value in zip(labels, values):
    print(f"   {label}: {value:+.1f}%")

# Create colors (red for negative, blue for positive)
colors = ['red' if v < 0 else 'blue' for v in values]

# Create horizontal bar chart
y_pos = np.arange(len(labels))
bars = ax.barh(y_pos, values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

# Set labels and title
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel('Normalized Sensitivity (% change per % parameter change)', fontsize=11, fontweight='bold')
ax.set_title('Parameter Sensitivity Tornado Chart\n(QED-Meta-de Sitter Warp Stack)', fontsize=12, fontweight='bold', pad=15)

# Add grid and center line
ax.grid(True, alpha=0.3, axis='x')
ax.axvline(x=0, color='black', linewidth=1.0)

# Set reasonable axis limits
max_abs_value = max(abs(v) for v in values) if values else 1
ax.set_xlim(-max_abs_value * 1.3, max_abs_value * 1.3)

# Add value labels on bars with better positioning
for i, (bar, value) in enumerate(zip(bars, values)):
    width = bar.get_width()
    
    # Position text outside the bar to avoid overlap
    if width >= 0:
        x_pos = width + max_abs_value * 0.05
        ha = 'left'
    else:
        x_pos = width - max_abs_value * 0.05
        ha = 'right'
    
    ax.text(x_pos, bar.get_y() + bar.get_height()/2, 
            f'{value:.1f}%', ha=ha, va='center', fontsize=10, fontweight='bold')

# Adjust layout with better margins
plt.tight_layout()
plt.subplots_adjust(left=0.25, right=0.9, top=0.85, bottom=0.15)

# Save chart
plt.savefig('experiment4_parameter_sensitivity.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print(f"\n✅ Chart saved as: experiment4_parameter_sensitivity.png")
plt.close()

print(f"\n🎯 CHART FIXES APPLIED:")
print(f"   - Proper negative values shown (red bars)")
print(f"   - All 5 parameters visible")
print(f"   - Better text positioning (no overlap)")
print(f"   - More reasonable chart size (10×6 inches)")
print(f"   - Clear color coding: Blue=positive, Red=negative") 