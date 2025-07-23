#!/usr/bin/env python3
"""
Quick verification that the tornado chart shows normalized sensitivities
"""
import sys
sys.path.append('../experiments')
from experiment4_parameter_sensitivity import *

# Run just the sensitivity calculation
base_params = {
    'L': 1200.0,
    'c': 299_792_458.0,
    'dn': -2.2e-6,
    'lens_frac': 0.4,
    'qed_field': 2e13,
    'warp_amp': -1e-7
}

param_list = ['L', 'dn', 'lens_frac', 'qed_field', 'warp_amp']
param_labels = {
    'L': 'Corridor Length',
    'dn': 'Metamaterial Δn', 
    'lens_frac': 'Lens Fraction',
    'qed_field': 'QED Field',
    'warp_amp': 'Warp Amplitude'
}

baseline_advance = early_arrival(**base_params) * 1e12
sensitivities = sensitivity_analysis(base_params, param_list)

# Calculate normalized sensitivities
normalized_sensitivities = {}
baseline_advance_sec = baseline_advance * 1e-12

for param in param_list:
    baseline_val = base_params[param]
    if baseline_val != 0 and baseline_advance_sec != 0:
        normalized_sens = sensitivities[param] * baseline_val / baseline_advance_sec * 100
        normalized_sensitivities[param] = normalized_sens
    else:
        normalized_sensitivities[param] = 0

print("🔍 VERIFICATION: Current Figure Should Show These Values:")
print("=" * 60)
sorted_norm_sens = sorted(normalized_sensitivities.items(), key=lambda x: abs(x[1]), reverse=True)
for param, norm_sens in sorted_norm_sens:
    print(f"   {param_labels[param]:18s}: {norm_sens:+6.1f}%")

print(f"\n✅ If your chart shows values like -1.60e+06, it's displaying OLD DATA")
print(f"✅ If your chart shows values like +100.0%, +95.7%, it's CORRECT")
print(f"✅ Current timestamp: {base_params}") 