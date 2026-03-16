# BMI Neural Decoding Competition – Quick Reference

## Task
Decode monkey hand X,Y position from spike trains. Causal decoder only (no future data).
Metric: **RMSe in cm** (lower = better). No Z position needed.

## Data: `monkeydata_training.mat`
- Variable: `trial` — 100×8 array of structs
- 98 neural units, 8 reaching angles, 100 training trials per angle (82×8 test, hidden)
- Trial window: 300ms before movement onset → 100ms after movement end
- `trial(n,k).spikes(i,:)` — unit i, trial n, angle k; binary 1ms bins
- `trial(n,k).handPos` — 3×T matrix, hand position in mm at 1ms steps (x=row1, y=row2, z=row3)
- Reaching angles (k=1..8): 30/180π, 70/180π, 110/180π, 150/180π, 190/180π, 230/180π, 310/180π, 350/180π

## Scoring / Testing
- Scored every 20ms: first 320ms fed in, then 340, 360, 380, 400… (causal enforcement)
- Max runtime: **5 minutes**
- Submit: `positionEstimatorTraining.m` + `positionEstimator.m` (two separate files in a zip)

## Constraints
- MATLAB only (no toolboxes to win — standard MATLAB functions only)
- All code must be your own
- Cite all resources
- Team member names on top of every file

## Deadlines
- Algorithm submission: 5pm GMT, 17 March 2025
- Report: due day of BMI exam (upload to Turnitin)

## Report
- 4 pages max, 2-column A4, LaTeX (Overleaf template on Blackboard)
- Sections: Methods, Results, Discussion, Contributions
- 10% penalty if over length or team doesn't participate

## File Naming Convention (this repo)
- `positionEstimatorTrainingN.m` + `positionEstimatorN.m` where N = version number
- Current best: version 7 (NRMSE below 7.3 for PE7)
