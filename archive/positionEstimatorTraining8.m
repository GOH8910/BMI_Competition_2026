function modelParameters = positionEstimatorTraining8(training_data)

    [nTrials, nDirs] = size(training_data);
    nNeurons = size(training_data(1,1).spikes, 1);
    wLen = 80;   % 80 ms non-overlapping windows

    modelParameters.nNeurons = nNeurons;
    modelParameters.nDirs    = nDirs;
    modelParameters.wLen     = wLen;

    % Find how many complete 80 ms windows fit in the shortest trial
    minLen = inf;
    for k = 1:nDirs
        for n = 1:nTrials
            minLen = min(minLen, size(training_data(n,k).spikes, 2));
        end
    end
    maxNWin = floor(minLen / wLen);
    modelParameters.maxNWin = maxNWin;

    % ====================================================================
    % STEP 1: Per-window regularised LDA classifiers
    % Feature: sqrt(spike count) per neuron in each 80 ms window
    % Regularisation: Sw <- (1-a)Sw + a(trace(Sw)/N)*I  (a = 0.2)
    % ====================================================================
    alpha_reg = 0.45;

    for w = 1:maxNWin
        ts = (w-1)*wLen + 1;
        te =  w   *wLen;

        allFeats = zeros(nTrials * nDirs, nNeurons);
        dirMeans = zeros(nDirs, nNeurons);

        for k = 1:nDirs
            df = zeros(nTrials, nNeurons);
            for n = 1:nTrials
                sp = training_data(n,k).spikes(:, ts:te);
                df(n,:) = sqrt(sum(sp, 2))';
            end
            dirMeans(k,:) = mean(df, 1);
            allFeats((k-1)*nTrials+1 : k*nTrials, :) = df;
        end

        Sw = zeros(nNeurons);
        for k = 1:nDirs
            C  = allFeats((k-1)*nTrials+1 : k*nTrials, :) - dirMeans(k,:);
            Sw = Sw + C' * C;
        end
        Sw = Sw / (nTrials*nDirs - nDirs);
        Sw = (1 - alpha_reg) * Sw + alpha_reg * (trace(Sw) / nNeurons) * eye(nNeurons);

        modelParameters.winLDA(w).dirMeans = dirMeans;
        modelParameters.winLDA(w).SwInv    = inv(Sw);
    end

    % ====================================================================
    % STEP 2: Average trajectories per direction (absolute + displacement)
    % ====================================================================
    for k = 1:nDirs
        maxLen = 0;
        for n = 1:nTrials
            maxLen = max(maxLen, size(training_data(n,k).handPos, 2));
        end

        sumT    = zeros(2, maxLen);
        sumDisp = zeros(2, maxLen);
        cntT    = zeros(1, maxLen);

        for n = 1:nTrials
            hp      = training_data(n,k).handPos(1:2,:);
            L       = size(hp, 2);
            startP  = hp(:,1);
            sumT(:, 1:L)    = sumT(:, 1:L)    + hp(:, 1:L);
            sumDisp(:, 1:L) = sumDisp(:, 1:L) + (hp(:, 1:L) - startP);
            cntT(1:L)       = cntT(1:L) + 1;
        end

        avgT = zeros(2, maxLen);
        avgD = zeros(2, maxLen);
        for t = 1:maxLen
            if cntT(t) > 0
                avgT(:,t) = sumT(:,t)    / cntT(t);
                avgD(:,t) = sumDisp(:,t) / cntT(t);
            elseif t > 1
                avgT(:,t) = avgT(:,t-1);
                avgD(:,t) = avgD(:,t-1);
            end
        end
        % Average start position per direction (for displacement-based prediction)
        avgStart = zeros(2,1);
        for n = 1:nTrials
            avgStart = avgStart + training_data(n,k).handPos(1:2,1);
        end
        avgStart = avgStart / nTrials;

        modelParameters.avgTraj{k}    = avgT;
        modelParameters.avgDisp{k}    = avgD;
        modelParameters.avgTrajLen(k) = maxLen;
        modelParameters.avgStart{k}   = avgStart;
    end

    % ====================================================================
    % STEP 3: Adaptive kernel-PCA + ridge regression (per nw, per dir)
    %
    % For each number of available windows nw=4..maxNWin and each
    % direction k, train:
    %   - Kernel PCA (10 PCs, economy kernel trick)
    %   - Ridge regression (lambda=1) from PCA scores to hand displacement
    %     at each evaluation timepoint in 320:20:800 ms
    % ====================================================================
    nPC       = 32;
    lambda    = 20;
    evalTimes = 320:20:800;
    modelParameters.pcaEvalTimes = evalTimes;

    trainTraj = cell(1, nDirs);
    for k = 1:nDirs
        trajCell = cell(1, nTrials);
        for n = 1:nTrials
            trajCell{n} = training_data(n,k).handPos(1:2,:);   % absolute position
        end
        trainTraj{k} = trajCell;
    end

    for nw = 4:maxNWin
        nFeat = nNeurons * nw;

        for k = 1:nDirs
            X = zeros(nTrials, nFeat);
            for n = 1:nTrials
                sp = training_data(n,k).spikes;
                for w = 1:nw
                    ts = (w-1)*wLen + 1;
                    te =  w   *wLen;
                    X(n, (w-1)*nNeurons+1 : w*nNeurons) = sqrt(sum(sp(:, ts:te), 2))';
                end
            end

            Xmean = mean(X, 1);
            Xc    = X - Xmean;

            K_mat = Xc * Xc';
            [V, D] = eig(K_mat);
            [eigvals, idx] = sort(diag(D), 'descend');
            V       = V(:, idx);
            nPC_use = min(nPC, sum(eigvals > 1e-6));
            scores  = V(:, 1:nPC_use) .* sqrt(max(eigvals(1:nPC_use), 0))';

            % Augment scores with startHandPos per trial for regression
            % This allows the regression to use trial-specific start position
            startPosAll = zeros(nTrials, 2);
            for n = 1:nTrials
                startPosAll(n,:) = training_data(n,k).handPos(1:2,1)';
            end
            spMean = mean(startPosAll, 1);
            startPosC = startPosAll - spMean;   % center startHandPos features
            scoresAug = [scores, startPosC];     % nTrials x (nPC_use+2)
            nFeatAug  = nPC_use + 2;

            betas  = zeros(nFeatAug, 2, length(evalTimes));
            ymeans = zeros(2, length(evalTimes));

            for ti = 1:length(evalTimes)
                t  = evalTimes(ti);
                Y  = zeros(nTrials, 2);
                for n = 1:nTrials
                    traj = trainTraj{k}{n};   % absolute position
                    if t <= size(traj,2)
                        Y(n,:) = traj(:,t)';
                    else
                        Y(n,:) = traj(:,end)';
                    end
                    % Target is absolute position
                end
                ym   = mean(Y, 1);
                Yc   = Y - ym;
                S2   = scoresAug' * scoresAug;
                beta = (S2 + lambda * eye(nFeatAug)) \ (scoresAug' * Yc);
                betas(:,:,ti)  = beta;
                ymeans(:,ti)   = ym';
            end

            modelParameters.pcaReg(nw, k).Xmean   = Xmean;
            modelParameters.pcaReg(nw, k).Xc       = Xc;
            modelParameters.pcaReg(nw, k).V        = V(:, 1:nPC_use);
            modelParameters.pcaReg(nw, k).eigvals  = eigvals(1:nPC_use);
            modelParameters.pcaReg(nw, k).nPC      = nPC_use;
            modelParameters.pcaReg(nw, k).betas    = betas;
            modelParameters.pcaReg(nw, k).ymeans   = ymeans;
            modelParameters.pcaReg(nw, k).spMean   = spMean;   % mean start pos for centering
        end
    end

    % ====================================================================
    % STEP 4: Per-direction ridge regression with causal time-varying features
    %
    % Features (only data up to t):
    %   [sqrt counts: first 4 windows (4*nN) | current 80ms window (nN) |
    %    cumulative sqrt counts normalised (nN) | t/800 (1)]
    % Target: absolute hand position at time t (no position input at inference)
    % lambda = 500, features standardised
    % ====================================================================
    ridgeLambda = 650;
    nInitWin    = 4;

    for k = 1:nDirs
        X = [];
        Y = [];
        for n = 1:nTrials
            sp   = training_data(n,k).spikes;
            hp   = training_data(n,k).handPos(1:2,:);
            Tmax = size(sp, 2);
            times  = 320:20:Tmax;

            % Precompute all window features for this trial
            allWinFeat = zeros(maxNWin, nNeurons);
            for w = 1:maxNWin
                ts = (w-1)*wLen + 1;
                te =  w   *wLen;
                if te <= size(sp, 2)
                    allWinFeat(w,:) = sqrt(sum(sp(:, ts:te), 2))';
                end
            end

            % Precompute fi (first 4 windows) - constant for this trial
            fi = reshape(allWinFeat(1:nInitWin, :)', 1, nInitWin*nNeurons);

            startP = hp(1:2, 1)';   % 1x2 start hand position
            for ti = 1:length(times)
                t   = times(ti);
                t_s = max(1, t - 79);
                fc  = sqrt(sum(sp(:, t_s:t), 2))';
                fcu = sqrt(sum(sp(:, 1:t), 2))' / sqrt(t / 80);
                X   = [X; fi, fc, fcu, t/800, startP];            %#ok<AGROW>
                Y   = [Y; hp(:, min(t,size(hp,2)))']; %#ok<AGROW>  absolute position
            end
        end

        Xm = mean(X, 1);
        Xs = std(X, 0, 1);
        Xs(Xs < 1e-8) = 1;
        Xn = (X - Xm) ./ Xs;
        Ym = mean(Y, 1);

        nFeat = size(Xn, 2);
        beta  = (Xn'*Xn + ridgeLambda*eye(nFeat)) \ (Xn' * (Y - Ym));

        modelParameters.ridge(k).beta  = beta;
        modelParameters.ridge(k).Xmean = Xm;
        modelParameters.ridge(k).Xstd  = Xs;
        modelParameters.ridge(k).Ymean = Ym;
    end

    % ====================================================================
    % STEP 5: KNN data store (spike features + trajectories per direction)
    % ====================================================================
    for k = 1:nDirs
        knnFeat = zeros(nTrials, nNeurons * maxNWin);
        for n = 1:nTrials
            sp = training_data(n,k).spikes;
            for w = 1:maxNWin
                ts = (w-1)*wLen + 1;
                te =  w   *wLen;
                knnFeat(n, (w-1)*nNeurons+1 : w*nNeurons) = sqrt(sum(sp(:, ts:te), 2))';
            end
        end
        modelParameters.knnFeat{k} = knnFeat;
        for n = 1:nTrials
            modelParameters.knnTraj{k,n} = training_data(n,k).handPos(1:2,:);
        end
    end
    modelParameters.knnStartScale = 0;   % unused, kept for compatibility

    % ====================================================================
    % STEP 6: Per-direction hyperparameters based on trajectory difficulty
    %
    % Difficulty = RMS deviation of individual trials from mean trajectory.
    % Hard directions (high variability): fewer KNN neighbours, less PCA
    %   weight, more ridge weight.
    % Easy directions (low variability): more neighbours, more PCA weight,
    %   less ridge weight.
    %
    % Linear map from dirFrac in [0,1] (easiest to hardest):
    %   K       : 20 -> 3
    %   pcaW    : 0.40 -> 0.05
    %   alpha   : 0.15 -> 0.35
    % ====================================================================
    dirDifficulty = zeros(1, nDirs);
    for k = 1:nDirs
        avgT  = modelParameters.avgTraj{k};
        aLen  = modelParameters.avgTrajLen(k);
        totV  = 0;
        nPts  = 0;
        for n = 1:nTrials
            hp = training_data(n,k).handPos(1:2,:);
            L  = min(size(hp,2), aLen);
            for t = 320:20:L
                totV = totV + sum((hp(:,t) - avgT(:,t)).^2);
                nPts = nPts + 1;
            end
        end
        dirDifficulty(k) = sqrt(totV / max(nPts, 1));
    end

    dMin   = min(dirDifficulty);
    dMax   = max(dirDifficulty);
    dRange = max(dMax - dMin, 1e-6);
    dirFrac = (dirDifficulty - dMin) / dRange;   % 0 = easiest, 1 = hardest

    modelParameters.knnK_perDir  = round(10 - 7 * dirFrac);     % 10..3
    modelParameters.pcaW_perDir  = 1.00 * ones(1, nDirs);   % 1.00 = all PCA
    modelParameters.alpha_perDir = 0.25 + 0.10 * dirFrac;       % 0.25..0.35
    modelParameters.dirDifficulty = dirDifficulty;

    % EMA and temperature parameters
    modelParameters.tempEarly  = 6.0;
    modelParameters.tempLate   = 0.5;
    modelParameters.emaGamma   = 0.10;
    modelParameters.smoothGamma      = 0.05;   % EMA smoothing via decodedHandPos (base)
    modelParameters.smoothGammaEarly = 0.05;   % gamma at nAvailWin=4 (early)
    modelParameters.smoothGammaLate  = 0.000;  % gamma at nAvailWin=maxNWin (late)
    modelParameters.maxJumpMm        = 80;     % jump threshold for outlier clamping (mm)
    modelParameters.clampGamma       = 0.30;   % gamma applied when jump > maxJumpMm
    modelParameters.kinematicBlend      = 0.15;   % weight of kinematic velocity prediction (base)
    modelParameters.kinematicBlendEarly = 0.22;   % kinematic weight at nAvailWin=4 (early)
    modelParameters.kinematicBlendLate  = 0.04;   % kinematic weight at nAvailWin=maxNWin (late)
    modelParameters.dispWeight       = 0.12;   % weight of displacement-from-start prediction (kept for compat)
    modelParameters.dispWeightEarly  = 0.20;   % dispWeight at nAvailWin=4 (early)
    modelParameters.dispWeightLate   = 0.01;   % dispWeight at nAvailWin=maxNWin (late)

    % ====================================================================
    % STEP 7: Data-driven Kalman filter matrices
    %
    % State: [x; y; vx; vy] (4D)
    % Evaluation timestep dt = 20ms (one call = one 20ms eval step)
    % F: fit by least squares from training state sequences
    % Q: covariance of residuals from F fit
    % R: decoder RMSE^2 (approx 45 mm^2 per axis, diagonal)
    % H: observe x and y only
    % blend: weight of Kalman output vs raw EMA output (0=all EMA, 1=all KF)
    % ====================================================================
    dt_eval = 20;   % ms between evaluation steps

    % Build state sequences from training data
    % State at each eval step: [x; y; vx; vy]
    % Velocity estimated as (pos(t) - pos(t-20)) / 20
    stateSeqCur  = [];   % state at time t
    stateSeqNext = [];   % state at time t+20ms

    for k = 1:nDirs
        for n = 1:nTrials
            hp = training_data(n,k).handPos(1:2,:);
            Tmax = size(hp, 2);
            evalT = 320:20:Tmax;
            for ei = 1:length(evalT)-1
                t1 = evalT(ei);
                t2 = evalT(ei+1);
                if t1 < 1 || t2 > Tmax
                    continue;
                end
                % velocity: backward difference (causal)
                if t1 > dt_eval
                    vx1 = (hp(1,t1) - hp(1,t1-dt_eval)) / dt_eval;
                    vy1 = (hp(2,t1) - hp(2,t1-dt_eval)) / dt_eval;
                else
                    vx1 = 0;
                    vy1 = 0;
                end
                if t2 > dt_eval
                    vx2 = (hp(1,t2) - hp(1,t2-dt_eval)) / dt_eval;
                    vy2 = (hp(2,t2) - hp(2,t2-dt_eval)) / dt_eval;
                else
                    vx2 = 0;
                    vy2 = 0;
                end
                s1 = [hp(1,t1); hp(2,t1); vx1; vy1];
                s2 = [hp(1,t2); hp(2,t2); vx2; vy2];
                stateSeqCur  = [stateSeqCur,  s1]; %#ok<AGROW>
                stateSeqNext = [stateSeqNext, s2]; %#ok<AGROW>
            end
        end
    end

    % Fit F by least squares: S_next = F * S_cur
    % F = S_next * S_cur' * inv(S_cur * S_cur')
    SScur  = stateSeqCur  * stateSeqCur';   % 4x4
    SSmix  = stateSeqNext * stateSeqCur';   % 4x4
    F_kalman = SSmix / (SScur + 1e-6 * eye(4));

    % Compute Q as covariance of residuals
    residuals = stateSeqNext - F_kalman * stateSeqCur;   % 4 x N
    Q_kalman  = (residuals * residuals') / size(residuals, 2);
    % Add small regularisation for numerical stability
    Q_kalman  = Q_kalman + 1e-4 * eye(4);

    % Observation noise R: approximate decoder RMSE^2 ~45 mm^2 per axis
    R_kalman = diag([45, 45]);

    H_kalman = [1 0 0 0; 0 1 0 0];

    modelParameters.kalman.F      = F_kalman;
    modelParameters.kalman.Q      = Q_kalman;
    modelParameters.kalman.R      = R_kalman;
    modelParameters.kalman.H      = H_kalman;
    modelParameters.kalman.blend  = 0.0;    % disabled

end
