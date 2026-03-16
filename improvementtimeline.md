# Decoder Improvement Timeline: v1 → v10

**BMI Neural Decoding Competition — Spring 2025**

---

## v1 — Classical PCA + LDA + Ridge Baseline

### Architecture Overview

Version 1 established the foundational pipeline used as a reference point for all subsequent work. It followed a classical dimensionality-reduction-then-regression approach.

**Preprocessing.** Raw spike trains were convolved with a causal Gaussian kernel ($\sigma = 25$ ms) and then resampled at 20 ms intervals to produce smooth firing rate estimates per neuron. The square root transform was applied to each binned count before the Gaussian smooth, serving as a variance-stabilising step for Poisson spike counts.

Low-firing neurons (mean rate $< 0.02$ spikes/ms) were removed from the feature set to reduce noise dimensionality. This left a variable subset of the 98 units.

**PCA.** Principal Component Analysis was applied to the neuron-by-time feature matrix. Components explaining 95% of cumulative variance were retained, with a minimum of 10 principal components enforced. This reduced the feature space to roughly 10–15 dimensions, compressing correlated neural variability.

**LDA.** Linear Discriminant Analysis was performed in the PCA subspace using the generalised eigenvalue formulation:

$$S_w^{-1} S_b \mathbf{w} = \lambda \mathbf{w}$$

where $S_w$ is the within-class scatter and $S_b$ the between-class scatter. The top 7 discriminant axes were retained (one fewer than the number of classes $K = 8$). This produced a 7-dimensional classification space where the 8 reaching directions were maximally separated. kNN with $k = 5$ in the LDA space, using inverse-distance weighting, gave the direction probability vector $P(k)$.

**Ridge regression.** Per-direction ridge regression was trained with lagged PC features as input:

$$\mathbf{r}(t) = [\mathbf{z}(t),\ \mathbf{z}(t-20\text{ms}),\ \mathbf{z}(t-40\text{ms}),\ \Delta\mathbf{z}(t),\ 1]$$

where $\mathbf{z}(t)$ is the PCA score vector at time $t$ and $\Delta\mathbf{z}$ is the finite difference. Two regression targets were used: absolute position and velocity (first difference of position / 20 ms).

**Ensemble.** The final prediction blended the direct position regression (50%) with a kinematic prediction formed by integrating the velocity regression from the previous decoded position (50%).

**Adaptive EMA.** An exponential moving average was applied to the decoded position, with the smoothing weight $\alpha$ ramping from 0.4 (early, high smoothing) to 0.85 (late, lower smoothing) over the first 15 evaluation steps.

### Thought Process

The v1 design followed the standard neural decoding pipeline from the literature. PCA addresses the high dimensionality and removes correlated noise between neurons. LDA then finds the projection that maximally discriminates between the 8 reaching directions in the reduced space. Ridge regression with lagged features captures the temporal dynamics of the neural population response. The velocity-position ensemble was motivated by the observation that neural signals carry information about both velocity and position: integrating a velocity estimate from the previous step enforces smooth trajectories and prevents discontinuous jumps.

The primary limitation was that PCA compression discards a significant fraction of the feature space, and the single LDA model applied to all time points cannot capture how discriminability changes across the trial.

### Performance

NRMSE $\approx 0.0948$ (from commit log, exact RMSE in mm not recorded)

---

### v2 — Displacement Anchoring + Softmax Temperature

**Change:** The velocity regression and velocity-position ensemble were dropped. In their place, displacement-from-start predictions were introduced: a soft-weighted average displacement per direction and a KNN displacement prediction (hybrid with $\alpha = 0.3$ KNN, $0.7$ direction average). Softmax temperature scaling was added to the direction posterior.

**Rationale:** Regression directly to absolute position is sensitive to the global coordinate frame and drifts as the trial progresses. Displacement from the known start position is a more stable target; the anchor to `startHandPos` eliminates accumulated integration error. Softmax temperature was added after observing that the direction probability was often overconfident early in the trial, leading to cascading errors when the top-1 prediction was wrong.

*RMSE not recorded for this version.*

---

### v3 — Time-Dependent Classifier Bank

**Change:** A time-dependent classifier bank was introduced: one LDA + kNN model was trained per time bin rather than a single global model. The classifier feature for bin $b$ was the mean PC score over bins $1$ through $b$ (accumulating richer information as the trial progresses). Temperature was set to a ramp from 2.5 (early) to 1.5 (late), and hybrid alpha was made confidence-adaptive.

**Rationale:** A single global LDA model ignores the temporal structure of neural responses. Direction selectivity in motor cortex evolves over the course of a reach; a per-bin model captures this. Using cumulative mean PCs as features ensures the classifier has access to all available data at each step.

*RMSE not recorded for this version.*

---

### v4 — Per-Direction Per-Time Displacement Regression Bank

**Change:** A PCR-style regression bank was added: one per direction, per time bin. The input feature was `[meanPC; 1]` (mean PC feature with bias), and the target was absolute displacement at that time bin. The regression output was blended with the average trajectory fallback. The time-dependent classifier from v3 was retained.

**Rationale:** Rather than relying solely on average displacement templates, per-direction regression allows the decoder to personalise the displacement prediction to the observed neural activity, improving accuracy for trials that deviate from the mean trajectory shape.

*RMSE not recorded for this version.*

---

### v5 — Two-Stage Preprocessing and Velocity Integration

**Change:** A two-stage preprocessing pipeline was implemented: spike trains were binned and sqrt-transformed first, then Gaussian-smoothed on the binned data. Per-neuron z-score normalisation was applied before PCA. SVD replaced `eig` for PCA computation. The regression target was switched to per-step velocity (20 ms displacement), which was then integrated to produce absolute position. A step-size constraint capped positional increments at 5 mm.

**Rationale:** Smoothing after binning reduces temporal aliasing. Z-score normalisation ensures neurons with higher baseline firing rates do not dominate the PCA. Velocity targets are often better conditioned than absolute position targets because they are more stationary. The step-size constraint prevents runaway integration from brief misclassification events.

*RMSE not recorded for this version.*

---

### v6 — Step Trajectory and EMA Constraint

**Change:** Preprocessing was simplified back to the v3-style pipeline (the v5 two-stage approach was reverted after testing). Both step (20 ms delta position) and displacement trajectory templates were stored. A step-size EMA constraint and a maximum step of 6 mm were applied. Direction probability was temporally smoothed across successive evaluation steps.

**Rationale:** The reversion from v5 suggests the more complex preprocessing did not yield a consistent improvement. Combining step and displacement trajectories provided additional fallback signals. Temporal smoothing of direction probabilities reduced flickering between directions on adjacent time steps, improving trajectory continuity.

*RMSE not recorded for this version.*

---

## v7 — Ground-Up Redesign: Per-Window Regularised LDA + Kernel PCA Regression

### Motivation

Versions 1–6 shared a common preprocessing and classification backbone: Gaussian smooth → PCA → LDA. The critical insight motivating v7 was that this pipeline has a fundamental throughput bottleneck. PCA reduces the 98-dimensional neural feature space to roughly 10–15 components before any classification or regression is attempted. Even if those 15 components explain 95% of the variance across all conditions, they may not preserve the discriminative structure between reaching directions, and they certainly discard fine-grained temporal information within each 80 ms epoch.

Additionally, the per-time-bin classifier models from v3–v6 used cumulative mean PC features, which are a low-dimensional representation of cumulative activity. A direction-specific observation that has just entered the feature window is immediately diluted by averaging over all prior bins.

### Architecture

**Spike preprocessing.** Replaced Gaussian smooth + 20 ms bins with non-overlapping 80 ms windows and the square root spike count transform:

$$f_i^{(w)} = \sqrt{\sum_{t=(w-1) \cdot 80 + 1}^{w \cdot 80} s_i(t)}$$

This removes the Gaussian smoothing step entirely and operates on raw counts per window. The sqrt transform stabilises Poisson variance. Each 80 ms window gives an $N = 98$ dimensional feature vector, preserving the full neural population at each time epoch.

**Per-window LDA with shrinkage.** A separate LDA model is trained for each window index $w$. The within-class scatter matrix $S_w \in \mathbb{R}^{98 \times 98}$ is computed from the 98-dimensional feature vectors. Because $N = 98 > N_{\text{train}}/K \approx 6$ (trials per class per window), $S_w$ is rank-deficient. Ledoit-Wolf style shrinkage is applied:

$$\hat{S}_w = (1 - \alpha) S_w + \alpha \cdot \frac{\operatorname{tr}(S_w)}{N} I_N, \quad \alpha = 0.2$$

This guarantees $\hat{S}_w$ is positive definite and invertible. The inverse $\hat{S}_w^{-1}$ is stored. At inference, log-likelihoods are accumulated across all available complete windows:

$$\log P(k) \mathrel{+}= -\frac{1}{2} (\mathbf{f}^{(w)} - \boldsymbol{\mu}_k^{(w)})^\top \hat{S}_w^{-1} (\mathbf{f}^{(w)} - \boldsymbol{\mu}_k^{(w)})$$

Temperature scaling (5.0 early → 1.0 late) was introduced at this point.

**Kernel PCA regression.** For each $(n_w, k)$ pair, a kernel PCA regression is trained. The key idea is to use the kernel trick to perform PCA in the $(N \cdot n_w)$-dimensional feature space without explicitly forming the $(N \cdot n_w) \times (N \cdot n_w)$ covariance matrix. Instead, the $N_{\text{train}} \times N_{\text{train}}$ kernel matrix $K = X_c X_c^\top$ is eigendecomposed, giving kernel PCA scores in $\mathbb{R}^{N_{\text{train}} \times p}$ with $p = 10$. Ridge regression from these scores to absolute hand position at each evaluation timepoint was trained with $\lambda = 1$.

**Per-direction ridge regression.** A causal ridge regressor was trained separately per direction, using the first 4 windows plus a current-window feature as input, and regressing to absolute position. This provided a complementary prediction that captured trial-by-trial variability not resolved by the kernel PCA model.

**KNN on absolute trajectories.** KNN retrieval was switched from displacement-based to absolute-trajectory-based: the $k$ nearest training trials (in feature space) have their actual hand position trajectories retrieved and weighted-averaged. This avoids the start-position sensitivity of displacement-based retrieval (where the displacement from start depends on knowing an accurate start position for each training trial).

**Three-way blend.** PCA regression + KNN trajectory + ridge regression, with per-direction weights determined by trajectory variability.

### Why This Worked

The key improvement over v1–v6 was the combination of two changes:

1. The per-window LDA with shrinkage directly uses the 98-dimensional neural population at each epoch, avoiding lossy PCA compression before classification. With proper regularisation, this gives substantially better classification accuracy than the PCA-compressed LDA, especially when direction selectivity is strongest in specific windows.

2. The kernel PCA regression avoids the feature-dimension explosion of direct PCA in the concatenated $(98 \cdot n_w)$-dimensional space by working in the dual (trial) space. This is computationally tractable ($50 \times 50$ eigendecomposition) while still capturing all linear structure in the high-dimensional feature space.

### Performance

RMSE_cm $\approx 0.73$ (sub-7.3 NRMSE, from commit message)

---

## v8 — Incremental Refinement of v7

### Overview

Version 8 is built entirely on the v7 architecture, with targeted improvements to each module based on empirical tuning and principled reasoning about failure modes of v7.

### 2.1 Stronger Shrinkage Regularisation ($\alpha$: 0.20 → 0.45)

With only 82 training trials and 98 neurons per window, the sample within-class scatter $S_w$ is computed from at most $N_{\text{train}} / K = 10$ observations per class in each window. The rank of $S_w$ is at most 9 for a single class, meaning the population covariance matrix has 89 eigenvalues equal to zero from data. The shrinkage weight $\alpha = 0.20$ used in v7 leaves 80% weight on this highly rank-deficient matrix. Increasing to $\alpha = 0.45$ places substantially more weight on the spherical term $\frac{\operatorname{tr}(S_w)}{N} I$, which assigns equal variance to all 98 directions. While this introduces bias, it dramatically reduces the variance of the precision matrix estimate $\hat{S}_w^{-1}$, improving generalisation of the log-likelihood accumulation.

### 2.2 More Kernel PCA Components ($p$: 10 → 32)

With $N_{\text{train}} = 82$ training trials, the kernel matrix is at most rank 82. Retaining only 10 components in v7 discarded considerable variance. Increasing to $p = 32$ captures finer structure in the training set, enabling the ridge regression to fit more nuanced direction- and time-dependent position patterns. Diminishing returns are expected beyond $p \approx 32$ because the remaining kernel PCA components are increasingly dominated by noise.

### 2.3 Stronger PCA Ridge Regularisation ($\lambda$: 1 → 20)

The larger score matrix $\tilde{Z} \in \mathbb{R}^{N_{\text{train}} \times 34}$ (32 kernel PCA components + 2 startPos) requires proportionally stronger ridge regularisation to prevent overfitting. $\lambda = 20$ was found to be appropriate for the augmented feature set.

### 2.4 startHandPos Integration

The trial start position $\mathbf{s}_0 \in \mathbb{R}^2$ is a known quantity at inference time (provided by the test harness as `test_data.startHandPos`). Within each direction class, trials do not begin from exactly the same hand position — there is a small but non-negligible spread of starting locations. When regressing to absolute position targets, this introduces residual within-class variance that is not explained by the neural features alone. By appending the centred start position to the kernel PCA score vector:

$$\tilde{Z} = [Z,\ S_c], \quad S_c = S - \mathbf{1}\bar{\mathbf{s}}^\top$$

the regression can correct for this offset. The same augmentation is applied to the per-direction ridge regression features.

### 2.5 Displacement-from-Start Predictors (avg and KNN)

Two new predictors were added:

- **avgDisp**: for each direction $k$, the mean displacement $\bar{\boldsymbol{\delta}}_k(t) = \mathbb{E}[\mathbf{h}(t) - \mathbf{h}(1)]$ over training trials is stored. At inference, this is combined with the test start position: $\hat{\mathbf{y}}_{\text{disp},k}(t) = \mathbf{s}_0 + \bar{\boldsymbol{\delta}}_k(t)$. The soft-weighted prediction over directions gives $\hat{\mathbf{y}}_{\text{disp}}$.

- **knnDispPos**: the KNN displacement prediction transfers the displacement profile from retrieved training trials to the test start position.

These predictors provide a clean anchor to the known start position. Early in a trial, before enough windows have accumulated for confident regression, the displacement predictors are more reliable than the regression models.

### 2.6 Five-Way Blend with Uncertainty-Aware Displacement Boost (was 3-way)

The 3-way blend of v7 (PCA + KNN + ridge) was expanded to 5 components. The displacement weight $w_d$ decays from 0.20 (early) to 0.01 (late), reflecting the decreasing relative benefit of displacement anchoring as regression accuracy improves.

A critical addition is the uncertainty-aware boost: when $P(k^*) < 0.65$, $w_d$ is increased by $5.0 \times (0.65 - P(k^*))$, up to a cap of 0.85. When the classifier is uncertain — meaning the neural evidence does not strongly favour any single direction — the regression models are operating in a mixed regime where the soft-weighted predictions are degraded by probability mass on incorrect directions. The displacement predictors, which only require a known start position and do not depend on direction confidence, are more trustworthy in this regime.

### 2.7 Recency-Weighted KNN Distance

The KNN distance metric was upgraded from uniform Euclidean distance to a recency-weighted distance:

$$d_n = \sum_{w=1}^{W_{\text{feat}}} \left(\frac{w}{W_{\text{feat}}}\right)^2 \left\|\mathbf{f}_w^{(n)} - \mathbf{f}_w^{\text{test}}\right\|^2$$

The quadratic weighting $(w/W_{\text{feat}})^2$ means the most recent window has weight 1 while the earliest window has weight $(1/W_{\text{feat}})^2 \approx 0.02$. This reflects the temporal structure of reaches: earlier windows capture the preparatory phase (which is relatively direction-invariant), while later windows capture the active movement phase (which is highly direction-specific).

### 2.8 Kinematic Velocity Blend

A kinematic prediction from recent decoded velocity history was added:

$$\hat{\mathbf{y}}_{\text{kin}}(t) = \hat{\mathbf{y}}(t-20) + \frac{\hat{\mathbf{y}}(t-20) - \hat{\mathbf{y}}(t-40)}{20} \cdot 20$$

blended with neural predictions at weight $\gamma_{\text{kin}}$ decaying from 0.22 to 0.04. This provides temporal continuity and suppresses oscillations arising from frame-to-frame variation in the neural regression output.

### 2.9 Adaptive EMA + Outlier Clamping

EMA smoothing was made adaptive: $\gamma$ decays from 0.05 (early) to 0.00 (late). Outlier clamping was added: jumps $\|\hat{\mathbf{y}}(t) - \hat{\mathbf{y}}(t-20)\| > 80$ mm trigger a proportional increase of $\gamma$ toward $\gamma_{\text{clamp}} = 0.30$. These large jumps are characteristic of transient direction misclassification events: the decoder suddenly shifts to a trajectory from the wrong direction. The clamping mechanism acts as a low-pass filter on large positional jumps, blending them back toward the previous decoded position.

### 2.10 Temperature Sharpening ($\tau_{\text{late}}$: 1.0 → 0.5)

The late-stage classification temperature was reduced from 1.0 to 0.5, producing a sharper posterior when many windows are available. With high-quality evidence from 7+ windows, the Mahalanobis log-likelihood accumulation is already well-calibrated, and reducing the temperature makes the posterior more decisive without significantly increasing the misclassification rate.

### 2.11 Data-Driven Kalman Filter (Infrastructure)

A data-driven Kalman filter over the 4D state $[x, y, v_x, v_y]^\top$ was implemented. The transition matrix $F$ is estimated by least squares from consecutive state pairs in the training trajectories. The filter blend weight is currently set to 0.0 (disabled), so it does not contribute to the v8 output. The infrastructure is retained for potential future use.

### Performance

RMSE_cm = **0.6487** (improvement of ~11% over v7's 0.73 cm)

---

## v9 — Population Vector Algorithm (Ablation Study)

### Architecture

Version 9 implemented the classical Georgopoulos et al. (1986) Population Vector Algorithm (PVA) as a complete alternative to the v8 hybrid pipeline. This was motivated by the theoretical question: does the v8 complexity genuinely add value over the canonical neuroscience approach?

**Cosine tuning model.** For each neuron $i$, firing rate $r_i(\theta)$ is modelled as a raised cosine function of reaching direction $\theta$:

$$r_i(\theta) = b_i + m_{\cos,i} \cos\theta + m_{\sin,i} \sin\theta$$

The parameters $(b_i, m_{\cos,i}, m_{\sin,i})$ are estimated by ordinary least squares using the mean firing rates across all 8 directions and all training trials, giving a $3 \times 1$ coefficient vector per neuron.

**Preferred direction and modulation depth.** The preferred direction and modulation depth for each neuron are:

$$\theta_i^* = \operatorname{atan2}(m_{\sin,i},\ m_{\cos,i}), \quad m_i = \sqrt{m_{\cos,i}^2 + m_{\sin,i}^2}$$

**Population vector.** At each evaluation time $t$, the observed firing rate $r_i(t)$ (estimated from a causal 80 ms window) is used to form the population vector:

$$\mathbf{P}(t) = \sum_{i=1}^{N} \max(r_i(t) - b_i,\ 0) \cdot m_i \cdot \begin{bmatrix} \cos\theta_i^* \\ \sin\theta_i^* \end{bmatrix}$$

The $\max(\cdot, 0)$ threshold removes neurons contributing negatively (below baseline). The modulation depth $m_i$ weights neurons that have stronger directional tuning more heavily.

**Velocity integration.** The population vector $\mathbf{P}(t)$ is interpreted as a velocity signal:

$$\hat{\mathbf{y}}(t) = \hat{\mathbf{y}}(t-20) + \mathbf{P}(t) \cdot s_{\text{scale}} \cdot 20$$

where $s_{\text{scale}}$ is a scalar speed calibration factor fitted from training data (median speed / median $|\mathbf{P}|$).

### Limitations and Failure Modes

1. **Cosine tuning assumption is violated for many units.** Many of the 98 neurons do not have clean unimodal tuning curves. Multi-peaked tuning, broad tuning with no clear preference, and weakly modulated units all introduce noise into the population vector.

2. **Velocity integration accumulates drift.** Unlike the v8 regression approach (which predicts absolute position at each timestep), velocity integration from an uncertain starting estimate accumulates error over time. A single bad window with a transient firing rate increase produces a permanent offset in the decoded trajectory.

3. **Discards temporal structure within trials.** The PVA uses only the instantaneous (80 ms window) firing rate. It ignores the rich temporal evolution of population responses across the full trial — information that the v8 kernel PCA regression and per-window LDA exploit explicitly.

4. **No trial-specific adaptation.** The PVA uses a single fixed set of tuning curves, with no mechanism to adapt to the particular neural pattern in the current trial (unlike KNN, which retrieves the most similar training trials).

### Value of this Ablation

The result clearly demonstrates that assuming independent cosine tuning curves and integrating velocity is insufficient for the complexity of actual motor cortical responses in this dataset. The v8 approach — which makes no assumption about tuning curve shape and instead learns position-predicting regression functions directly — is strictly more powerful in this setting.

### Performance

RMSE_cm = **1.051** (62% worse than v8's 0.6487)

---

## v10 — 40ms Windows (Ablation Study)

### Architecture

Version 10 is architecturally identical to v8, with a single change: the window length is halved to $L = 40$ ms (from 80 ms), doubling the temporal resolution and doubling the number of windows per trial.

### Expected Behaviour

For a Poisson neuron firing at rate $\lambda$ spikes/ms, the expected spike count in a window of length $L$ ms is $c = \lambda L$. The sqrt-count feature therefore has expected value:

$$\mathbb{E}[f_i^{(w)}] = \mathbb{E}[\sqrt{c}] \approx \sqrt{\lambda L}$$

Halving $L$ reduces the expected feature value by $\sqrt{2}$. The variance of $\sqrt{c}$ is approximately $1/4$ regardless of $L$ (by the Anscombe approximation), so the signal-to-noise ratio of each feature:

$$\text{SNR} = \frac{\mathbb{E}[f_i^{(w)}]}{\sqrt{\text{Var}(f_i^{(w)})}} \approx \frac{\sqrt{\lambda L}}{1/2} = 2\sqrt{\lambda L}$$

is proportional to $\sqrt{L}$. Halving $L$ from 80 to 40 ms reduces the SNR of each feature by $\sqrt{2} \approx 1.41$.

### Why Smaller Windows Hurt Performance

The v8 design trains one LDA model per window using only $N_{\text{train}} \approx 82$ samples (all training trials of one direction). With 40 ms windows, each window's features are noisier — the Mahalanobis log-likelihoods accumulated per window have higher variance. While there are twice as many windows (and thus twice as many log-likelihood terms accumulated), the individual contributions are noisier, and the net effect is a reduction in classification accuracy.

For the kernel PCA regression, the feature matrix $X \in \mathbb{R}^{N_{\text{train}} \times (N \cdot n_w)}$ has twice as many columns for the same $n_w$, but the individual features carry less signal. The ridge regression must now fit a larger, noisier feature space with the same $N_{\text{train}} = 82$ observations, increasing overfitting risk.

The result confirms that $L = 80$ ms is closer to the optimal tradeoff between temporal resolution and per-window SNR given the available training set size of 82 trials per direction. A larger training set would shift this tradeoff toward finer temporal resolution.

### Performance

RMSE_cm = **0.7841** (21% worse than v8's 0.6487, worse than v7's 0.73)

---

## Summary Table

| Version | Key Algorithm | RMSE_cm | Key Decision |
|---------|--------------|---------|-------------|
| v1 | Gaussian smooth → PCA → LDA → kNN → ridge (pos + vel) | NRMSE ~0.0948 (not cm) | Establish baseline: classical dimensionality reduction + regression |
| v2 | + Displacement anchoring, softmax temperature | Not recorded | Absolute position drift → displace from startHandPos |
| v3 | + Per-bin classifier bank, temperature ramp | Not recorded | Single-time LDA ignores temporal evolution of selectivity |
| v4 | + Per-dir per-bin displacement regression bank | Not recorded | Template displacement → personalised displacement regression |
| v5 | + Two-stage preprocessing, velocity integration | Not recorded | Velocity targets more stationary; step-size constraint for integration |
| v6 | + Step + displacement trajectory, step-size EMA | Not recorded | Revert preprocessing; add step trajectory as additional signal |
| v7 | 80ms windows + per-window shrinkage LDA + kernel PCA + ridge (3-way blend) | ~0.73 | Ground-up redesign: skip PCA compression; use full 98D per window with regularisation |
| v8 | v7 + stronger shrinkage (α=0.45) + nPC=32 + startPos augment + 5-way blend + uncertainty disp boost + recency KNN + kinematic blend + EMA clamping | **0.6487** | Targeted refinements to each v7 module; uncertainty-aware displacement anchoring |
| v9 | Population Vector Algorithm (cosine tuning + velocity integration) | 1.051 | Ablation: cosine tuning assumption fails; velocity integration drifts |
| v10 | v8 with wLen=40ms | 0.7841 | Ablation: 40ms windows lower per-window SNR, hurting classification and regression |

*Note: RMSE values for v2–v6 were not recorded during development. The NRMSE value for v1 (0.0948) is reported in normalised units rather than cm, and is not directly comparable to the RMSE_cm values for v7–v10.*
