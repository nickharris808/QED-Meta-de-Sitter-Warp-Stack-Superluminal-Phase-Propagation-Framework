# QED-Meta-de Sitter Warp Stack: Project Structure

**Organized Structure Version 2.0** | **Updated**: July 23, 2024

---

## 📁 **Complete Directory Organization**

```
qed-meta-sitter-warp-stack/
├── 📄 README.md                               # Complete user guide & quick start
├── 📄 PROJECT_STATUS.md                       # Current validation status
├── 📄 PROJECT_STRUCTURE.md                    # This file - detailed organization
├── 📄 PROJECT_MANIFEST.md                     # Historical development record
├── 📄 requirements.txt                        # Python dependencies
├── 📄 .gitignore                             # Version control settings
│
├── 📁 manuscript/                             # PUBLICATION MATERIALS ✅
│   ├── manuscript_final.pdf                  # Ready for submission (22 pages, 4.8 MB)
│   ├── manuscript_final.tex                  # LaTeX source with all corrections
│   ├── references.bib                        # 15 authoritative references (AMA format)
│   ├── manuscript_final.aux                  # LaTeX auxiliary files
│   ├── manuscript_final.bbl                  # Bibliography compilation
│   ├── manuscript_final.log                  # Compilation log
│   ├── simulation_report.html                # Supplementary analysis
│   │
│   ├── figures/                              # All publication figures
│   │   ├── experiment1_group_velocity_fdtd.png
│   │   ├── experiment2_dispersion_bandwidth.png     # ✅ CORRECTED (4.02 THz)
│   │   ├── experiment3_monte_carlo_uncertainty.png
│   │   ├── experiment4_parameter_sensitivity.png    # ✅ CORRECTED (realistic)
│   │   ├── experiment5_symbolic_anec.png
│   │   ├── energy_conditions.png
│   │   ├── fdtd_trace.png
│   │   ├── geodesic_lead.png
│   │   ├── n_index_profile.png
│   │   ├── phase_jitter_hist.png
│   │   ├── grid_convergence_study.png
│   │   ├── geodesic_fdtd_correlation.png
│   │   └── parameter_sweep_heatmap.png
│   │
│   ├── data/                                 # Simulation datasets
│   └── archive_manuscript_files/             # Historical/temporary files
│
├── 📁 primary_simulation/                     # MAIN SCIENTIFIC VALIDATION ✅
│   ├── superluminal_warp_stack.py            # Primary physics engine (renamed from afs.py)
│   ├── physics_parameters.py                 # Optimized parameter set (renamed from validation_studies.py)
│   └── outputs/                              # Primary scientific results
│       ├── figures/                          # Real physics visualizations
│       │   ├── geodesic_lead.png
│       │   ├── fdtd_trace.png
│       │   ├── energy_conditions.png
│       │   ├── n_index_profile.png
│       │   └── phase_jitter_hist.png
│       ├── data/                             # Comprehensive datasets
│       │   ├── simulation_summary.csv
│       │   ├── energy_integral.txt
│       │   ├── phase_control_statistics.csv
│       │   └── grid_convergence_data.csv
│       └── simulation_report.html            # Detailed analysis report
│
├── 📁 experiments/                           # VALIDATION EXPERIMENTS ✅ (ALL 5/5 WORKING)
│   ├── run_all_experiments.py               # Execute complete validation suite
│   │
│   ├── experiment1_causality_analysis.py     # Group vs phase velocity (renamed)
│   ├── experiment2_dispersion_study.py       # ✅ FIXED: 4.02 THz plasma freq (renamed)
│   ├── experiment3_uncertainty_analysis.py   # Monte Carlo robustness (renamed)
│   ├── experiment4_parameter_sensitivity.py  # ✅ FIXED: Realistic values
│   ├── experiment5_energy_conditions.py      # Symbolic ANEC proof (renamed)
│   │
│   ├── experiment_fixes_summary.md           # Documentation of corrections
│   ├── warpstack_validation_summary.txt      # Comprehensive validation report
│   │
│   └── [Generated figures and outputs]       # PNG files and validation data
│       ├── experiment1_group_velocity_fdtd.png
│       ├── experiment2_dispersion_bandwidth.png
│       ├── experiment3_monte_carlo_uncertainty.png
│       ├── experiment4_parameter_sensitivity.png
│       └── experiment5_symbolic_anec.png
│
├── 📁 analysis/                              # RESEARCH ANALYSIS ✅
│   ├── physics_verification.md              # Detailed physics validation
│   ├── research_paper_analysis.md           # Literature context & positioning
│   ├── results_analysis.md                  # Key findings summary
│   └── README.md                            # Analysis directory guide
│
├── 📁 documentation/                         # DEVELOPMENT DOCUMENTATION ✅
│   ├── experiment_corrections/              # Fix documentation
│   │   ├── experiment_fixes_summary.md
│   │   ├── qed_scaling_corrections.md
│   │   └── parameter_optimization_notes.md
│   │
│   ├── figure_generation/                   # Figure creation details
│   │   ├── figure_correction_methods.md
│   │   ├── matplotlib_generation_scripts.py
│   │   └── figure_quality_standards.md
│   │
│   ├── validation_reports/                  # Comprehensive validation
│   │   ├── comprehensive_validation.md
│   │   ├── physics_verification_complete.md
│   │   └── numerical_stability_analysis.md
│   │
│   └── [Historical investigation files]
│       ├── CORRECTED_EXPERIMENT_ANALYSIS.md
│       ├── CITATIONS_CORRECTED_SUMMARY.md
│       ├── DEEP_EXPERIMENT_INVESTIGATION_REPORT.md
│       └── FIGURE_CORRECTION_COMPLETE.md
│
├── 📁 archive/                               # HISTORICAL VERSIONS ✅
│   ├── previous_manuscripts/                # Earlier manuscript versions
│   │   ├── manuscript.tex
│   │   ├── manuscript_complete.pdf
│   │   └── [previous versions]
│   │
│   └── obsolete_experiments/                # Old parameter sets
│       ├── [Original problematic scripts]
│       └── [Deprecated figure files]
│
├── 📁 results/                               # ADDITIONAL OUTPUTS
│   ├── outputs/                             # Secondary simulation results
│   └── validation_outputs/                  # Experiment validation data
│
└── 📁 venv/                                 # PYTHON VIRTUAL ENVIRONMENT
    ├── bin/                                 # Executables
    ├── lib/                                 # Python packages
    └── [standard venv structure]
```

---

## 🎯 **Key File Locations**

### **For Immediate Use**
| **Purpose** | **File Location** | **Description** |
|-------------|-------------------|-----------------|
| **Quick Start** | `README.md` | Complete user guide |
| **Primary Results** | `primary_simulation/superluminal_warp_stack.py` | Main physics validation |
| **Manuscript** | `manuscript/manuscript_final.pdf` | Publication-ready paper |
| **All Experiments** | `experiments/run_all_experiments.py` | Complete validation suite |
| **Status Check** | `PROJECT_STATUS.md` | Current validation status |

### **For Development**
| **Purpose** | **File Location** | **Description** |
|-------------|-------------------|-----------------|
| **Physics Parameters** | `primary_simulation/physics_parameters.py` | Optimized parameter set |
| **Experiment Fixes** | `experiments/experiment_fixes_summary.md` | Correction documentation |
| **Results Analysis** | `analysis/` | Complete physics verification |
| **Figure Generation** | `documentation/figure_generation/` | Figure creation methods |

### **For Peer Review**
| **Purpose** | **File Location** | **Description** |
|-------------|-------------------|-----------------|
| **Main Paper** | `manuscript/manuscript_final.pdf` | 22 pages, 4.8 MB |
| **Supplementary Code** | `primary_simulation/` | Complete simulation suite |
| **Validation Suite** | `experiments/` | All 5 working experiments |
| **Documentation** | `documentation/validation_reports/` | Comprehensive validation |

---

## 🔧 **Major Organizational Improvements**

### **Directory Renaming** ✅
- `src/` → `primary_simulation/` (clearer purpose)
- `validation_experiments/` → `experiments/` (concise)
- `paper/` → `manuscript/` (academic standard)

### **File Renaming** ✅
- `afs.py` → `superluminal_warp_stack.py` (descriptive)
- `validation_studies.py` → `physics_parameters.py` (clear function)
- `experiment1_group_velocity_fdtd.py` → `experiment1_causality_analysis.py`
- `experiment2_dispersion_bandwidth.py` → `experiment2_dispersion_study.py`
- `experiment3_monte_carlo_uncertainty.py` → `experiment3_uncertainty_analysis.py`
- `experiment5_symbolic_anec.py` → `experiment5_energy_conditions.py`

### **Structure Organization** ✅
- Separated manuscript figures into `manuscript/figures/`
- Organized documentation into themed subdirectories
- Archived temporary and historical files
- Clear separation between primary and validation results

### **Documentation Hierarchy** ✅
- `README.md`: User-facing quick start and overview
- `PROJECT_STATUS.md`: Current validation and submission status
- `PROJECT_STRUCTURE.md`: This file - detailed organization
- `PROJECT_MANIFEST.md`: Historical development record

---

## 🚀 **Usage by Directory**

### **For Running Simulations**
```bash
# Primary physics validation
cd primary_simulation
python superluminal_warp_stack.py

# All validation experiments  
cd experiments
python run_all_experiments.py
```

### **For Manuscript Work**
```bash
# Compile manuscript
cd manuscript
pdflatex manuscript_final.tex

# Access figures
ls manuscript/figures/
```

### **For Analysis**
```bash
# Review physics validation
cd analysis
cat physics_verification.md

# Check experiment fixes
cd experiments
cat experiment_fixes_summary.md
```

---

## 📊 **File Count Summary**

| **Directory** | **Key Files** | **Purpose** |
|---------------|---------------|-------------|
| `manuscript/` | 1 PDF, 1 TEX, 15+ figures | Publication materials |
| `primary_simulation/` | 2 Python files, outputs/ | Main scientific validation |
| `experiments/` | 6 Python files, 5+ PNG | Validation experiments |
| `analysis/` | 3 MD files | Research documentation |
| `documentation/` | 10+ MD files | Development records |
| `archive/` | Historical files | Previous versions |

**Total**: ~50+ organized files across 6 main directories

---

## ✅ **Organization Benefits**

### **Clarity** 
- Descriptive directory and file names
- Clear separation of primary vs validation work
- Logical grouping of related materials

### **Accessibility**
- README.md provides immediate orientation
- Quick access to key results and code
- Clear documentation hierarchy

### **Maintainability**
- Archived obsolete materials
- Organized development documentation
- Version control friendly structure

### **Peer Review Ready**
- Complete manuscript with corrected figures
- Reproducible code with clear entry points
- Comprehensive validation documentation

---

**Structure Status**: ✅ **FULLY ORGANIZED**  
**Last Updated**: July 23, 2024  
**Ready For**: Peer review submission and long-term maintenance 