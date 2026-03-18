# BMI Neural Decoding Competition — Imperial College London

**1st place** · Lowest RMSE: **6.0043 cm**

---

## Overview

This repository contains our submission for the Brain-Machine Interface (BMI) neural decoding competition, part of the BMI module at Imperial College London.

The task was to decode a monkey's hand X,Y position in real time from 98 neural spike train recordings across 8 reaching directions. Decoders must be strictly causal (no access to future data), scored every 20ms, and implemented in standard MATLAB with no toolboxes.

Our final submission achieved a root mean square error (RMSE) of **6.0043 cm**, placing 1st out of all competing teams.

---

## Approach

Our decoder uses population vector decoding with exponential spike rate smoothing, angle classification, and per-angle linear regression models trained on binned firing rates. A Kalman-filter-inspired velocity integration step refines position estimates over time.

---

## Repository Structure

```
final_submission/       # Competition submission files (winning entry)
│   positionEstimator8.m
│   positionEstimatorTraining8.m
archive/                # All previous versions and supporting files
monkeydata_training.mat # Training data (not included in submission)
README.md               # This file
```

---

## Results

| Metric | Value     |
|--------|-----------|
| RMSE   | 6.0043 cm |
| Rank   | 1st place |

---

*Captions and documentation generated with [Claude Code](https://claude.ai/claude-code).*
