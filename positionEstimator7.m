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

function [x, y, newModelParameters] = positionEstimator7(test_data, modelParameters)

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

    % ----------------------------------------------------------------
    % STEP 4: KNN trajectory for best-classified direction
    % Retrieves absolute hand positions from training trials — not the
    % current test position.
    % ----------------------------------------------------------------
    nTrain    = size(modelParameters.knnFeat{bestDir}, 1);
    dists     = sum((modelParameters.knnFeat{bestDir}(:, 1:nN*nFeatWin) - testFeat).^2, 2);
    [~, sortIdx] = sort(dists);
    kIdx      = sortIdx(1 : min(knnK, nTrain));
    kDists    = dists(kIdx);
    sigma     = median(kDists) + 1e-6;
    wts       = exp(-kDists / (2*sigma));
    wts       = wts / sum(wts);

    knnTraj = zeros(2,1);
    for ki = 1:length(kIdx)
        hp_k = modelParameters.knnTraj{bestDir, kIdx(ki)};
        if T <= size(hp_k, 2)
            knnTraj = knnTraj + wts(ki) * hp_k(:, T);
        else
            knnTraj = knnTraj + wts(ki) * hp_k(:, end);
        end
    end

    % Nearest evaluation time
    [~, eti] = min(abs(evalTimes - T));

    % Ridge feature vector (causal: data up to T only)
    fi  = testFeat(1 : 4*nN);
    t_s = max(1, T - 79);
    fc  = sqrt(sum(sp(:, t_s:T), 2))';
    fcu = sqrt(sum(sp(:, 1:T), 2))' / sqrt(T / 80);
    ridgeFeat = [fi, fc, fcu, T/800];

    % ----------------------------------------------------------------
    % STEP 5: Soft-weighted predictions across all directions
    % PCA and ridge predict absolute position (trained without startPos).
    % ----------------------------------------------------------------
    softPCA   = zeros(2,1);
    softTraj  = zeros(2,1);
    softRidge = zeros(2,1);

    for k = 1:nDirs
        % PCA regression -> absolute position
        rm   = modelParameters.pcaReg(nFeatWin, k);
        xc   = testFeat - rm.Xmean;
        kvec = rm.Xc * xc';
        test_score = (rm.V' * kvec)' ./ sqrt(max(rm.eigvals', 1e-6));
        pred_pos   = (test_score * rm.betas(:,:,eti))' + rm.ymeans(:,eti);
        softPCA    = softPCA + P(k) * pred_pos;

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
    % STEP 6: Three-way blend with per-direction adaptive weights
    %   basePos = pcaWeight * softPCA + (1-pcaWeight) * softTraj
    %   pos     = (1-ridgeAlpha) * basePos + ridgeAlpha * softRidge
    % All components derived purely from neural features.
    % ----------------------------------------------------------------
    basePos = pcaWeight * softPCA + (1 - pcaWeight) * softTraj;
    pos     = (1 - ridgeAlpha) * basePos + ridgeAlpha * softRidge;

    x = pos(1);
    y = pos(2);

end
