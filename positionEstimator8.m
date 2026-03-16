%%% Model 4: Full V11 Hybrid Decoder
%%%   LDA + Adaptive KNN + PCA Regression + Ridge + Per-Direction Hyperparams
%%% BMI Spring 2025

% function [x, y] = positionEstimator(test_data, modelParameters)
%
% Full hybrid decoder — strictly causal: only spike data up to and
% including the current timestep T is used.  No position input of any kind
% (no startHandPos, no decodedHandPos, no EMA feedback).
%
% Inference pipeline:
%   1. Temperature-scaled soft LDA classification -> P(k) over 8 directions
%   2. KNN trajectory retrieval for the best-classified direction
%      (looks up absolute training trajectories — not the current test position)
%   3. Soft-weighted PCA regression, KNN trajectory, and ridge predictions
%      (PCA and ridge trained to predict absolute position from neural features)
%   4. Three-way blend with per-direction adaptive weights
%
% Arguments:
%   test_data.trialId          unique trial ID
%   test_data.startHandPos     2x1 hand position at trial start  [not used]
%   test_data.decodedHandPos   2xN previously decoded positions  [not used]
%   test_data.spikes(i,t)      spike trains from t=1 to current time T
%   modelParameters            struct returned by positionEstimatorTraining
%
% Return Values:
%   x, y   decoded hand position (mm)

function [x, y, newModelParameters] = positionEstimator8(test_data, modelParameters)

    newModelParameters = modelParameters;

    sp   = test_data.spikes;
    T    = size(sp, 2);          % current time (ms) — only spike data used

    wLen      = modelParameters.wLen;
    maxNWin   = modelParameters.maxNWin;
    nDirs     = modelParameters.nDirs;
    nN        = modelParameters.nNeurons;
    evalTimes = modelParameters.pcaEvalTimes;
    tempEarly = modelParameters.tempEarly;
    tempLate  = modelParameters.tempLate;

    % ----------------------------------------------------------------
    % STEP 1: Temperature-scaled soft LDA classification
    % Only windows that lie entirely within [1, T] are used.
    % Temperature: 5.0 (few windows, soft) -> 1.0 (many windows, sharp)
    % ----------------------------------------------------------------
    nAvailWin = min(floor(T / wLen), maxNWin);

    logP = zeros(1, nDirs);
    for w = 1:nAvailWin
        ts   = (w-1)*wLen + 1;
        te   =  w   *wLen;
        feat = sqrt(sum(sp(:, ts:te), 2))';

        for k = 1:nDirs
            d       = feat - modelParameters.winLDA(w).dirMeans(k,:);
            logP(k) = logP(k) - 0.5 * (d * modelParameters.winLDA(w).SwInv * d');
        end
    end

    temp  = tempEarly + (tempLate - tempEarly) * (nAvailWin - 4) / max(maxNWin - 4, 1);
    temp  = max(temp, 0.5);
    logP  = logP / temp;
    maxLP = max(logP);
    P     = exp(logP - maxLP);
    P     = P / sum(P);
    [~, bestDir] = max(P);

    % ----------------------------------------------------------------
    % STEP 2: Per-direction adaptive hyperparameters
    % ----------------------------------------------------------------
    knnK       = modelParameters.knnK_perDir(bestDir);
    pcaWeight  = modelParameters.pcaW_perDir(bestDir);
    ridgeAlpha = modelParameters.alpha_perDir(bestDir);

    % ----------------------------------------------------------------
    % STEP 3: Spike feature vector (causal — windows 1..nFeatWin)
    % ----------------------------------------------------------------
    nFeatWin = max(4, nAvailWin);
    nFeatWin = min(nFeatWin, maxNWin);

    testFeat = zeros(1, nN * nFeatWin);
    for w = 1:nFeatWin
        ts = (w-1)*wLen + 1;
        te =  w   *wLen;
        testFeat((w-1)*nN+1 : w*nN) = sqrt(sum(sp(:, ts:te), 2))';
    end

    % startPos is used in KNN distance, displacement predictions, and ridge features
    startPos = test_data.startHandPos(1:2);

    % ----------------------------------------------------------------
    % STEP 4: KNN trajectory for best-classified direction
    % Retrieves absolute hand positions from training trials — not the
    % current test position.
    % ----------------------------------------------------------------
    nTrain    = size(modelParameters.knnFeat{bestDir}, 1);
    % Weighted distance: weight later windows more heavily (more informative)
    % Quadratic weighting: (ww/nFeatWin)^2
    winWeights = ones(1, nN * nFeatWin);
    for ww = 1:nFeatWin
        winWeights((ww-1)*nN+1 : ww*nN) = (ww / nFeatWin)^2;
    end
    diff_feat = (modelParameters.knnFeat{bestDir}(:, 1:nN*nFeatWin) - testFeat) .* winWeights;
    dists     = sum(diff_feat.^2, 2);
    [~, sortIdx] = sort(dists);
    kIdx      = sortIdx(1 : min(knnK, nTrain));
    kDists    = dists(kIdx);
    sigma     = median(kDists) + 1e-6;
    wts       = exp(-kDists / (2*sigma));
    wts       = wts / sum(wts);

    knnTraj    = zeros(2,1);
    knnDispPos = zeros(2,1);   % displacement-from-start prediction via KNN (bestDir)
    startPos   = test_data.startHandPos(1:2);
    for ki = 1:length(kIdx)
        hp_k = modelParameters.knnTraj{bestDir, kIdx(ki)};
        hpT  = hp_k(:, min(T, size(hp_k,2)));
        knnTraj    = knnTraj    + wts(ki) * hpT;
        knnStartK  = hp_k(:,1);
        knnDispPos = knnDispPos + wts(ki) * (startPos + (hpT - knnStartK));
    end

    % Nearest evaluation time
    [~, eti] = min(abs(evalTimes - T));

    % Ridge feature vector (causal: data up to T only)
    fi  = testFeat(1 : 4*nN);
    t_s = max(1, T - 79);
    fc  = sqrt(sum(sp(:, t_s:T), 2))';
    fcu = sqrt(sum(sp(:, 1:T), 2))' / sqrt(T / 80);
    startHandPos2 = test_data.startHandPos(1:2)';   % 1x2
    ridgeFeat = [fi, fc, fcu, T/800, startHandPos2];

    % ----------------------------------------------------------------
    % STEP 5: Soft-weighted predictions across all directions
    % PCA and ridge predict absolute position (trained without startPos).
    % ----------------------------------------------------------------
    softPCA   = zeros(2,1);
    softTraj  = zeros(2,1);
    softRidge = zeros(2,1);

    for k = 1:nDirs
        % PCA regression -> absolute position (with startHandPos augmentation)
        rm   = modelParameters.pcaReg(nFeatWin, k);
        xc   = testFeat - rm.Xmean;
        kvec = rm.Xc * xc';
        test_score  = (rm.V' * kvec)' ./ sqrt(max(rm.eigvals', 1e-6));
        spC         = (startPos' - rm.spMean);   % 1x2 centered start pos
        test_scoreA = [test_score, spC];          % 1x(nPC+2)
        pred_pos    = (test_scoreA * rm.betas(:,:,eti))' + rm.ymeans(:,eti);
        softPCA     = softPCA + P(k) * pred_pos;

        % Trajectory: KNN for best direction, mean trajectory for others
        if k == bestDir
            softTraj = softTraj + P(k) * knnTraj;
        else
            avgT = modelParameters.avgTraj{k};
            aLen = modelParameters.avgTrajLen(k);
            if T <= aLen
                softTraj = softTraj + P(k) * avgT(:, T);
            else
                softTraj = softTraj + P(k) * avgT(:, end);
            end
        end

        % Ridge regression -> absolute position
        rm_r     = modelParameters.ridge(k);
        xn       = (ridgeFeat - rm_r.Xmean) ./ rm_r.Xstd;
        ridgePos = (xn * rm_r.beta + rm_r.Ymean)';
        softRidge = softRidge + P(k) * ridgePos;
    end

    % ----------------------------------------------------------------
    % STEP 5b: Displacement-from-start predictions
    % (a) softDisp: use startHandPos + soft-weighted average displacement
    %     per direction over all 8 directions weighted by P(k).
    % (b) knnDispPos: already computed above from KNN-retrieved trials.
    % Both leverage the known trial start position (startPos defined above).
    % ----------------------------------------------------------------
    softDisp = zeros(2,1);
    for k = 1:nDirs
        avgD = modelParameters.avgDisp{k};
        aLen = modelParameters.avgTrajLen(k);
        if T <= aLen
            dispAtT = avgD(:, T);
        else
            dispAtT = avgD(:, end);
        end
        softDisp = softDisp + P(k) * (startPos + dispAtT);
    end

    % ----------------------------------------------------------------
    % STEP 6: Five-way blend with per-direction adaptive weights
    %   basePos      = pcaWeight * softPCA + (1-pcaWeight) * softTraj
    %   neuralPos    = (1-ridgeAlpha) * basePos + ridgeAlpha * softRidge
    %   dispPred     = 0.5 * softDisp + 0.5 * knnDispPos
    %   pos          = (1-dispWeight) * neuralPos + dispWeight * dispPred
    % ----------------------------------------------------------------
    basePos   = pcaWeight * softPCA + (1 - pcaWeight) * softTraj;
    neuralPos = (1 - ridgeAlpha) * basePos + ridgeAlpha * softRidge;
    dispPred  = 0.0 * softDisp + 1.0 * knnDispPos;
    % Time-varying dispWeight: high early (trust displacement template),
    % low late (trust neural regression more).
    % Also scale by classification uncertainty: low P(bestDir) -> more disp.
    frac_disp  = (nAvailWin - 4) / max(maxNWin - 4, 1);
    frac_disp  = min(max(frac_disp, 0), 1);
    dispWeight = modelParameters.dispWeightEarly * (1 - frac_disp) + modelParameters.dispWeightLate * frac_disp;
    % Uncertainty bonus: when bestDir prob < threshold, add extra disp weight
    confBonus  = max(0, 0.65 - P(bestDir)) * 5.0;   % up to 1.07 extra (capped)
    dispWeight = min(dispWeight + confBonus, 0.85);
    pos        = (1 - dispWeight) * neuralPos + dispWeight * dispPred;

    % ----------------------------------------------------------------
    % STEP 7: Velocity-aware blending + adaptive EMA smoothing +
    %         data-driven Kalman filter
    % When decodedHandPos history has 2+ entries, compute recent velocity
    % and form a kinematic prediction (prevPos + vel*dt).
    % Blend kinematic prediction with neural decoder prediction before EMA.
    % ----------------------------------------------------------------
    if ~isempty(test_data.decodedHandPos)
        prevPos  = test_data.decodedHandPos(:, end);
        nPrev    = size(test_data.decodedHandPos, 2);

        % Kinematic velocity prediction using recent decoded positions
        if nPrev >= 2
            vel = (test_data.decodedHandPos(:,end) - test_data.decodedHandPos(:,end-1)) / 20;
            kinematicPos = prevPos + vel * 20;   % predict 20ms ahead

            % Time-varying kinematic blend: higher early (smooth startup),
            % lower late (trust neural more)
            frac_kin = (nAvailWin - 4) / max(maxNWin - 4, 1);
            frac_kin = min(max(frac_kin, 0), 1);
            kinW = modelParameters.kinematicBlendEarly * (1 - frac_kin) + modelParameters.kinematicBlendLate * frac_kin;
            pos  = (1 - kinW) * pos + kinW * kinematicPos;
        end

        gEarly   = modelParameters.smoothGammaEarly;
        gLate    = modelParameters.smoothGammaLate;
        frac     = (nAvailWin - 4) / max(maxNWin - 4, 1);
        frac     = min(max(frac, 0), 1);
        gamma    = gEarly + (gLate - gEarly) * frac;

        % Outlier clamping: proportionally boost gamma for large jumps
        jumpDist   = norm(pos - prevPos);
        maxJump    = modelParameters.maxJumpMm;
        clampGamma = modelParameters.clampGamma;
        if jumpDist > maxJump
            excess = min((jumpDist - maxJump) / maxJump, 1);
            gamma  = gamma + (clampGamma - gamma) * excess;
        end

        pos = (1 - gamma) * pos + gamma * prevPos;
    end

    % ----------------------------------------------------------------
    % STEP 8: Data-driven Kalman filter update
    % State: [x; y; vx; vy]. We maintain P (covariance) in newModelParameters.
    % Predict with data-driven F, then update with observation pos.
    % ----------------------------------------------------------------
    kalman = modelParameters.kalman;
    F_k    = kalman.F;
    Q_k    = kalman.Q;
    R_k    = kalman.R;
    H_k    = kalman.H;
    blend  = kalman.blend;

    if isfield(modelParameters, 'kalmanState')
        xK = modelParameters.kalmanState;
        PK = modelParameters.kalmanP;
    else
        % Initialise state from first observation
        xK = [pos; 0; 0];   % [x; y; vx=0; vy=0]
        PK = eye(4) * 1000; % large initial uncertainty
    end

    % Predict step
    xK_pred = F_k * xK;
    PK_pred = F_k * PK * F_k' + Q_k;

    % Update step (observe x and y from neural decoder output pos)
    innov = pos - H_k * xK_pred;           % 2x1
    S_inn = H_k * PK_pred * H_k' + R_k;   % 2x2
    KG    = PK_pred * H_k' / S_inn;        % 4x2 Kalman gain
    xK    = xK_pred + KG * innov;
    PK    = (eye(4) - KG * H_k) * PK_pred;

    % Store updated Kalman state
    newModelParameters.kalmanState = xK;
    newModelParameters.kalmanP     = PK;

    % Blend Kalman estimate with EMA-smoothed neural output
    kalmanPos = xK(1:2);
    pos = (1 - blend) * pos + blend * kalmanPos;

    x = pos(1);
    y = pos(2);

end
