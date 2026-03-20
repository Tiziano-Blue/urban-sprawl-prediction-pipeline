# Urban Expansion Transition Pipeline

A clean, reproducible XGBoost workflow for **transition-based urban expansion prediction** using raster data.

## Overview

This project models **urban expansion as land conversion**, rather than static land classification.  
2010?

The workflow is organized into three conceptual stages:

1. **Training** on 2000 → 2010 transitions  
2. **Validation** on 2010 → 2024 transitions  
3. **Prediction** from 2024 conditions to estimate 2030 expansion risk  

Although the project keeps these three stages conceptually separate, they are executed sequentially in one main script.
