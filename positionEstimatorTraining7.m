%%% Model 4: Full V11 Hybrid Decoder
%%%   LDA + Adaptive KNN + PCA Regression + Ridge + Per-Direction Hyperparams
%%% BMI Spring 2025

% function modelParameters = positionEstimatorTraining(training_data)
%
% Full pipeline:
%   1. Per-window regularised LDA classifiers (one per 80 ms window)
%   2. Average trajectories per direction (absolute + displacement)
%   3. Adaptive kernel-PCA + ridge regression (separate model per window count)
%   4. Per-direction ridge regression with time-varying causal features
%   5. KNN data store (spike features + trajectories per direction)
%   6. Per-direction hyperparameters (K, pcaWeight, ridgeAlpha) based on
%      trajectory variability ("difficulty") of each direction
%
% Arguments:
%   training_data(n,k)              n = trial index, k = reaching direction
%   training_data(n,k).trialId      unique trial number
%   training_data(n,k).spikes(i,t)  i = neuron, t = time (ms)
%   training_data(n,k).handPos(d,t) d = dimension [1-3], t = time (ms)
%
% Return Value:
%   modelParameters  struct with all learned parameters for positionEstimator

function modelParameters = positionEstimatorTraining7(training_data)

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
    alpha_reg = 0.2;

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
        modelParameters.avgTraj{k}    = avgT;
        modelParameters.avgDisp{k}    = avgD;
        modelParameters.avgTrajLen(k) = maxLen;
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
    nPC       = 10;
    lambda    = 1;
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

            betas  = zeros(nPC_use, 2, length(evalTimes));
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
                end
                ym   = mean(Y, 1);
                Yc   = Y - ym;
                S2   = scores' * scores;
                beta = (S2 + lambda * eye(nPC_use)) \ (scores' * Yc);
                betas(:,:,ti)  = beta;
                ymeans(:,ti)   = ym';
            end

            modelParameters.pcaReg(nw, k).Xmean  = Xmean;
            modelParameters.pcaReg(nw, k).Xc      = Xc;
            modelParameters.pcaReg(nw, k).V        = V(:, 1:nPC_use);
            modelParameters.pcaReg(nw, k).eigvals  = eigvals(1:nPC_use);
            modelParameters.pcaReg(nw, k).nPC      = nPC_use;
            modelParameters.pcaReg(nw, k).betas    = betas;
            modelParameters.pcaReg(nw, k).ymeans   = ymeans;
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
    ridgeLambda = 500;
    nInitWin    = 4;

    for k = 1:nDirs
        X = [];
        Y = [];
        for n = 1:nTrials
            sp   = training_data(n,k).spikes;
            hp   = training_data(n,k).handPos(1:2,:);
            Tmax = size(sp, 2);
            times  = 320:20:Tmax;

            fi = zeros(1, nInitWin * nNeurons);
            for w = 1:nInitWin
                ts = (w-1)*wLen + 1;
                te =  w   *wLen;
                fi((w-1)*nNeurons+1 : w*nNeurons) = sqrt(sum(sp(:, ts:te), 2))';
            end

            for ti = 1:length(times)
                t   = times(ti);
                t_s = max(1, t - 79);
                fc  = sqrt(sum(sp(:, t_s:t), 2))';
                fcu = sqrt(sum(sp(:, 1:t), 2))' / sqrt(t / 80);
                X   = [X; fi, fc, fcu, t/800];                    %#ok<AGROW>
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

    modelParameters.knnK_perDir  = round(20 - 17 * dirFrac);    % 20..3
    modelParameters.pcaW_perDir  = 0.40 - 0.35 * dirFrac;       % 0.40..0.05
    modelParameters.alpha_perDir = 0.15 + 0.20 * dirFrac;       % 0.15..0.35
    modelParameters.dirDifficulty = dirDifficulty;

    % EMA and temperature parameters
    modelParameters.tempEarly = 5.0;
    modelParameters.tempLate  = 1.0;
    modelParameters.emaGamma  = 0.10;

end
