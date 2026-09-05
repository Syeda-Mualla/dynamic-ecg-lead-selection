# \# Disease-Specific Dynamic ECG Lead Selection

# 

# \## Overview

# 

 This research project presents a machine learning framework for \*\*disease-specific dynamic ECG lead selection\*\* for multi-label cardiac classification. The framework investigates whether a reduced number of ECG leads can provide reliable diagnostic information while improving robustness to missing or low-quality leads.

# 

# The project uses the \*\*PTB-XL\*\* dataset for development and evaluation and the \*\*Chapman–Shaoxing ECG\*\* dataset for external validation.

# 

# \## Objectives

# 

# \- Develop disease-specific ECG lead selection strategies.

# \- Evaluate reduced-lead ECG configurations for cardiac classification.

# \- Analyze model robustness to missing and low-quality ECG leads.

# \- Incorporate uncertainty-based referral for unreliable predictions.

# \- Evaluate generalization through external validation.

# 

# \## Methodology

# 

# The research framework includes:

# 

# 1\. ECG preprocessing and signal-quality assessment.

# 2\. Disease-specific lead scoring and selection.

# 3\. Hard top-4 ECG lead selection.

# 4\. Multi-label cardiac disease classification.

# 5\. Missing-lead robustness evaluation.

# 6\. Uncertainty-based referral.

# 7\. Comparison with the full 12-lead ECG configuration.

# 8\. External validation using the Chapman–Shaoxing dataset.

# 

# \## Datasets

# 

# \### PTB-XL

# Used for model development and primary evaluation.

# 

# \### Chapman–Shaoxing ECG

# Used as an external dataset to evaluate model generalization.

# 

# > Dataset files are not included in this repository due to dataset licensing and size considerations.

# 

# \## Technologies

# 

# \- Python

# \- NumPy

# \- Pandas

# \- Scikit-learn

# \- SciPy

# \- Matplotlib

# \- Machine Learning

# 

# \## Project Structure

# 

# ```text

# ├── notebooks/        # Research notebooks and experiments

# ├── src/              # Source code

# ├── configs/           # Configuration files

# ├── results/           # Experimental results and figures

# ├── requirements.txt   # Python dependencies

# └── README.md          # Project documentation

