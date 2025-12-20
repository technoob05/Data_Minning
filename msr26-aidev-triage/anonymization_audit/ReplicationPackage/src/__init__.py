"""
Replication Package: Early-Stage Prediction of Review Effort in AI-Generated PRs
MSR 2026 Mining Challenge

This package contains utilities for analyzing AI-generated pull requests
and predicting review effort based on creation-time signals.

Main Components:
- config.py: Configuration constants and paths
- features.py: Feature engineering utilities
- load.py: Data loading and preprocessing
- metrics.py: Evaluation metrics
- models.py: Model training and evaluation
- utils.py: Shared utility functions
- viz.py: Visualization helpers

For usage instructions, see README.md
"""

__version__ = "1.0.0"
__author__ = "Anonymous for Review"
__paper__ = "Early-Stage Prediction of Review Effort in AI-Generated Pull Requests"

# Import key functions for convenient access
try:
    from .config import *
    from .features import extract_features
    from .models import train_model, evaluate_model
    from .metrics import compute_metrics
except ImportError:
    # Allow package to be imported even if dependencies are not installed
    pass
