%%% Model 9: Population Vector Algorithm (PVA) Decoder
%%%   PVA-based velocity integration + average trajectory fallback
%%% BMI Spring 2025
%%% Team: Edward Goh

% function [x, y, newModelParameters] = positionEstimator9(test_data, modelParameters)
%
% PVA inference pipeline — strictly causal.
%
%   1. Compute causal firing rates in a 150ms window ending at current time T
%   2. Compute population vector (PVA) weighted by modulation depth
%   3. Classify reaching direction via LDA on first 320ms of spikes
%   4. Convert PVA to velocity; integrate from startHandPos
%   5. Blend PVA-integrated position with average trajectory (fallback)
%   6. Apply EMA smoothing using decodedHandPos history
%
% Arguments:
%   test_data.trialId          unique trial ID
%   test_data.startHandPos     2x1 hand position at trial start
%   test_data.decodedHandPos   2xN previously decoded positions
%   test_data.spikes(i,t)      spike trains from t=1 to current time T
%   modelParameters            struct returned by positionEstimatorTraining9
%
% Return Values:
%   x, y                 decoded hand position (mm)
%   newModelParameters   updated model parameters (carries PVA state)

function [x, y, newModelParameters] = positionEstimator9(test_data, modelParameters)

    newModelParameters = modelParameters;

    sp         = test_data.spikes;
    T          = size(sp, 2);     % current time in ms
    startPos   = test_data.startHandPos(1:2);

    nDirs      = modelParameters.nDirs;
    nNeurons   = modelParameters.nNeurons;
    prefDir    = modelParameters.prefDir;     % 2 x nNeurons
    baseline   = modelParameters.baseline;   % nNeurons x 1
    modDepth   = modelParameters.modDepth;   % nNeurons x 1
    speedScale = modelParameters.speedScale;
    windowLen  = modelParameters.windowLen;  % 150ms

    % ----------------------------------------------------------------
    % STEP 1: Compute causal firing rates in [T-windowLen+1 : T]
    % ----------------------------------------------------------------
    t_start  = max(1, T - windowLen + 1);
    wL       = T - t_start + 1;
    rates_now = sum(sp(:, t_start:T), 2) / (wL / 1000);   % spikes/sec, nNeurons x 1

    % ----------------------------------------------------------------
    % STEP 2: Compute population vector
    %   r_above = rectified deviation from baseline
    %   popVec  = sum_i [ r_above_i * m_i * C_i ]
    % ----------------------------------------------------------------
    r_above = max(rates_now - baseline, 0);                    % nNeurons x 1
    popVec  = prefDir * (r_above .* modDepth);                 % 2x1
    pva_mag = norm(popVec);

    % ----------------------------------------------------------------
    % STEP 3: LDA direction classification (use first 320ms = 4 windows)
    % ----------------------------------------------------------------
    wLen    = modelParameters.wLen;
    maxNWin = modelParameters.maxNWin;   % 4 windows of 80ms
    nFeat   = nNeurons * maxNWin;

    testFeat = zeros(1, nFeat);
    for w = 1:maxNWin
        ts = (w-1)*wLen + 1;
        te =  w   *wLen;
        if te <= T
            testFeat((w-1)*nNeurons+1 : w*nNeurons) = sqrt(sum(sp(:, ts:te), 2))';
        elseif ts <= T
            % partial last window — use what's available
            testFeat((w-1)*nNeurons+1 : w*nNeurons) = sqrt(sum(sp(:, ts:T), 2))';
        end
    end

    logP = zeros(1, nDirs);
    for k = 1:nDirs
        d       = testFeat - modelParameters.ldaDirMeans(k,:);
        logP(k) = -0.5 * (d * modelParameters.ldaSwInv * d');
    end
    maxLP = max(logP);
    P     = exp(logP - maxLP);
    P     = P / sum(P);
    [~, bestDir] = max(P);

    % ----------------------------------------------------------------
    % STEP 4: PVA velocity integration
    %   velocity = popVec * speedScale  (mm/ms)
    %   Integrate from startHandPos cumulatively via decodedHandPos
    % ----------------------------------------------------------------
    dt = 20;   % ms between evaluation steps

    % Get current integrated position from previous decoded positions
    if ~isempty(test_data.decodedHandPos)
        prevPos = test_data.decodedHandPos(:, end);
    else
        prevPos = startPos;
    end

    % PVA-predicted velocity increment over dt ms
    pva_vel     = popVec * speedScale;        % 2x1 mm/ms
    pva_pos_new = prevPos + pva_vel * dt;     % integrate

    % ----------------------------------------------------------------
    % STEP 5: Average trajectory prediction (fallback / blend anchor)
    % ----------------------------------------------------------------
    avgT = modelParameters.avgTraj{bestDir};
    aLen = modelParameters.avgTrajLen(bestDir);

    if T <= aLen
        avgPos = avgT(:, T);
    else
        avgPos = avgT(:, end);
    end

    % Soft-weighted average over all directions for robustness
    % Also compute displacement-from-start prediction using known startHandPos
    softAvg  = zeros(2,1);
    softDisp = zeros(2,1);
    for k = 1:nDirs
        aT = modelParameters.avgTraj{k};
        aL = modelParameters.avgTrajLen(k);
        aD = modelParameters.avgDisp{k};
        if T <= aL
            softAvg  = softAvg  + P(k) * aT(:, T);
            softDisp = softDisp + P(k) * (startPos + aD(:, T));
        else
            softAvg  = softAvg  + P(k) * aT(:, end);
            softDisp = softDisp + P(k) * (startPos + aD(:, end));
        end
    end

    % ----------------------------------------------------------------
    % STEP 6: Blend PVA-integrated position with average trajectory
    %
    % alpha = PVA blend weight. When PVA magnitude is large (strong
    % neural signal), trust PVA more. When small, fall back to avgTraj.
    % Time-varying: trust avgTraj more early (T < 400), PVA more later.
    % Displacement prediction uses known startHandPos — anchored to actual start.
    % ----------------------------------------------------------------

    % Base alpha: increases with time
    t_frac = min(max((T - 320) / 480, 0), 1);   % 0 at T=320, 1 at T=800

    % PVA confidence: sigmoidal in pva_mag
    pva_conf = pva_mag / max(pva_mag + 50, 1e-6);  % 0 when silent, approaches 1

    % Blended alpha — keep PVA weight moderate; avgTraj is a reliable anchor
    alpha_pva = 0.10 + 0.15 * t_frac + 0.05 * pva_conf;   % 0.10 .. 0.30
    alpha_pva = min(max(alpha_pva, 0.0), 0.45);

    % Mix absolute avgTraj with displacement-based prediction:
    % Early: trust displacement more (it uses startHandPos which is known)
    % Late: trust absolute avgTraj more (longer trajectories are reliable)
    disp_w = 0.75 * (1 - t_frac);   % 0.75 early, 0 late
    fallback = (1 - disp_w) * softAvg + disp_w * softDisp;

    pos = alpha_pva * pva_pos_new + (1 - alpha_pva) * fallback;

    % ----------------------------------------------------------------
    % STEP 7: EMA smoothing using decoded history
    % ----------------------------------------------------------------
    if ~isempty(test_data.decodedHandPos)
        prevPos  = test_data.decodedHandPos(:, end);
        nPrev    = size(test_data.decodedHandPos, 2);

        % Adaptive EMA: more smoothing early, less late
        gamma_early = 0.10;
        gamma_late  = 0.02;
        gamma = gamma_early + (gamma_late - gamma_early) * t_frac;

        % Outlier clamping: if jump too large, pull back toward prevPos
        jumpDist = norm(pos - prevPos);
        maxJump  = 60;   % mm
        if jumpDist > maxJump
            excess = min((jumpDist - maxJump) / maxJump, 1.0);
            gamma  = gamma + (0.5 - gamma) * excess;
        end

        pos = (1 - gamma) * pos + gamma * prevPos;

        % Kinematic velocity blending (only after >=2 decoded steps)
        if nPrev >= 2
            vel          = (test_data.decodedHandPos(:,end) - test_data.decodedHandPos(:,end-1)) / 20;
            kinematicPos = prevPos + vel * 20;
            % Low kinematic weight — just for stability
            kinW = 0.05 * (1 - t_frac);   % 0.05 early, 0 late
            pos  = (1 - kinW) * pos + kinW * kinematicPos;
        end
    end

    x = pos(1);
    y = pos(2);

end
