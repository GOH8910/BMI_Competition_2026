function tuneHyperparameters()
    load monkeydata0.mat

    rng(2013);
    ix = randperm(length(trial));

    allTraining = trial(ix(1:50), :);
    nFolds = 3;
    foldSize = floor(size(allTraining, 1) / nFolds);

    %% --- Phase 1: Training-time params (sigma, nPCs, lambda, phaseSplit) ---
    sigma_grid  = [0, 25, 50];
    nPCs_grid   = [15, 20, 30];
    lambda_grid = [1e-3, 1e-2, 1e-1];
    phaseSplit_fixed = 9;

    fprintf('=== Phase 1: Tuning sigma, nPCs, lambda (3-fold CV) ===\n');
    bestRMSE1 = inf;
    bestSigma = 25; bestNPCs = 20; bestLambda = 0.01;

    for si = 1:length(sigma_grid)
        for pi = 1:length(nPCs_grid)
            for li = 1:length(lambda_grid)
                hp.sigma = sigma_grid(si);
                hp.nPCs  = nPCs_grid(pi);
                hp.lambda = lambda_grid(li);
                hp.phaseSplit = phaseSplit_fixed;

                foldErrors = zeros(nFolds, 1);
                for fold = 1:nFolds
                    valIdx = (fold-1)*foldSize+1 : fold*foldSize;
                    trainIdx = setdiff(1:size(allTraining,1), valIdx);
                    params = trainWithParams(allTraining(trainIdx,:), hp);
                    foldErrors(fold) = evaluateModel(allTraining(valIdx,:), params);
                end

                avgRMSE = mean(foldErrors);
                fprintf('  sigma=%2d nPCs=%2d lambda=%.0e -> RMSE=%.4f\n', ...
                    hp.sigma, hp.nPCs, hp.lambda, avgRMSE);

                if avgRMSE < bestRMSE1
                    bestRMSE1 = avgRMSE;
                    bestSigma = hp.sigma;
                    bestNPCs  = hp.nPCs;
                    bestLambda = hp.lambda;
                end
            end
        end
    end
    fprintf('Best Phase 1: sigma=%d, nPCs=%d, lambda=%.0e, RMSE=%.4f\n\n', ...
        bestSigma, bestNPCs, bestLambda, bestRMSE1);

    %% --- Phase 2: phaseSplit (CV with best Phase 1 params) ---
    phaseSplit_grid = [0, 7, 9, 12];
    fprintf('=== Phase 2: Tuning phaseSplit (3-fold CV) ===\n');
    bestPhaseSplit = 9;
    bestRMSE2 = inf;

    for psi = 1:length(phaseSplit_grid)
        hp.sigma = bestSigma;
        hp.nPCs  = bestNPCs;
        hp.lambda = bestLambda;
        hp.phaseSplit = phaseSplit_grid(psi);

        foldErrors = zeros(nFolds, 1);
        for fold = 1:nFolds
            valIdx = (fold-1)*foldSize+1 : fold*foldSize;
            trainIdx = setdiff(1:size(allTraining,1), valIdx);
            params = trainWithParams(allTraining(trainIdx,:), hp);
            foldErrors(fold) = evaluateModel(allTraining(valIdx,:), params);
        end

        avgRMSE = mean(foldErrors);
        fprintf('  phaseSplit=%2d -> RMSE=%.4f\n', hp.phaseSplit, avgRMSE);

        if avgRMSE < bestRMSE2
            bestRMSE2 = avgRMSE;
            bestPhaseSplit = hp.phaseSplit;
        end
    end
    fprintf('Best Phase 2: phaseSplit=%d, RMSE=%.4f\n\n', bestPhaseSplit, bestRMSE2);

    %% --- Phase 3: Prediction-time params (single fold, fast) ---
    k_grid        = [3, 5, 7, 11];
    alphaMin_grid = [0.3, 0.4, 0.5];
    alphaMax_grid = [0.75, 0.85, 0.95];
    ensW_grid     = [0.3, 0.5, 0.7];
    rampSteps_fixed = 15;

    fprintf('=== Phase 3: Tuning k, alphaMin, alphaMax, ensembleW ===\n');
    hp.sigma = bestSigma;
    hp.nPCs  = bestNPCs;
    hp.lambda = bestLambda;
    hp.phaseSplit = bestPhaseSplit;

    trainData = allTraining(1:40, :);
    valData   = allTraining(41:end, :);
    baseParams = trainWithParams(trainData, hp);

    bestRMSE3 = inf;
    bestK = 5; bestAlphaMin = 0.4; bestAlphaMax = 0.85; bestEnsW = 0.5;

    for ki = 1:length(k_grid)
        for ami = 1:length(alphaMin_grid)
            for axi = 1:length(alphaMax_grid)
                if alphaMax_grid(axi) <= alphaMin_grid(ami), continue; end
                for ei = 1:length(ensW_grid)
                    p = baseParams;
                    p.kNN_k    = k_grid(ki);
                    p.alphaMin = alphaMin_grid(ami);
                    p.alphaMax = alphaMax_grid(axi);
                    p.rampSteps = rampSteps_fixed;
                    p.ensembleW = ensW_grid(ei);

                    rmse = evaluateModel(valData, p);
                    fprintf('  k=%2d aMin=%.1f aMax=%.2f ensW=%.1f -> RMSE=%.4f\n', ...
                        k_grid(ki), alphaMin_grid(ami), alphaMax_grid(axi), ...
                        ensW_grid(ei), rmse);

                    if rmse < bestRMSE3
                        bestRMSE3 = rmse;
                        bestK = k_grid(ki);
                        bestAlphaMin = alphaMin_grid(ami);
                        bestAlphaMax = alphaMax_grid(axi);
                        bestEnsW = ensW_grid(ei);
                    end
                end
            end
        end
    end

    fprintf('\n=============== BEST HYPERPARAMETERS ===============\n');
    fprintf('  gaussSigma  = %d\n', bestSigma);
    fprintf('  nPCs        = %d\n', bestNPCs);
    fprintf('  lambda      = %.0e\n', bestLambda);
    fprintf('  phaseSplit  = %d\n', bestPhaseSplit);
    fprintf('  kNN_k       = %d\n', bestK);
    fprintf('  alphaMin    = %.1f\n', bestAlphaMin);
    fprintf('  alphaMax    = %.2f\n', bestAlphaMax);
    fprintf('  rampSteps   = %d\n', rampSteps_fixed);
    fprintf('  ensembleW   = %.1f\n', bestEnsW);
    fprintf('  Phase1 CV RMSE  = %.4f\n', bestRMSE1);
    fprintf('  Phase2 CV RMSE  = %.4f\n', bestRMSE2);
    fprintf('  Phase3 RMSE     = %.4f\n', bestRMSE3);
    fprintf('====================================================\n');
end


%% ================================================================
%  Training function (mirrors positionEstimatorTraining with all features)
%  ================================================================
function params = trainWithParams(training_data, hp)
    binSize = 20;
    nDirections = size(training_data, 2);
    nTrials = size(training_data, 1);
    nNeurons = size(training_data(1,1).spikes, 1);
    classifyBins = 320 / binSize;
    nLags = 2;

    sigma = hp.sigma;
    lambda = hp.lambda;
    nPCs_target = hp.nPCs;
    phaseSplit = hp.phaseSplit;

    % Gaussian kernel (sigma=0 means hard binning)
    if sigma > 0
        kernelHW = ceil(3 * sigma);
        kernelX = -kernelHW:kernelHW;
        gKernel = exp(-kernelX.^2 / (2 * sigma^2));
        gKernel = gKernel / sum(gKernel);
        useGauss = true;
    else
        gKernel = [];
        useGauss = false;
    end

    %% Feature extraction
    allRates = [];
    trialRates = cell(nTrials, nDirections);
    trialHP = cell(nTrials, nDirections);

    for k = 1:nDirections
        for n = 1:nTrials
            spikes = training_data(n, k).spikes;
            T = size(spikes, 2);

            if useGauss
                samplePts = binSize:binSize:T;
                nBins = length(samplePts);
                rates = zeros(nNeurons, nBins);
                for i = 1:nNeurons
                    smoothed = conv(spikes(i,:), gKernel, 'same');
                    rates(i,:) = sqrt(smoothed(samplePts));
                end
            else
                nBins = floor(T / binSize);
                rates = zeros(nNeurons, nBins);
                for b = 1:nBins
                    rates(:, b) = sqrt(sum(spikes(:, (b-1)*binSize+1 : b*binSize), 2));
                end
            end

            trialRates{n, k} = rates;
            trialHP{n, k} = training_data(n, k).handPos;
            allRates = [allRates, rates];
        end
    end

    %% PCA
    mu = mean(allRates, 2);
    centered = allRates - mu;
    C = (centered * centered') / (size(centered, 2) - 1);
    [V, D] = eig(C);
    d = diag(D);
    [d, idx] = sort(d, 'descend');
    V = V(:, idx);

    nPCs = min(nPCs_target, size(V, 2));
    W_pca = V(:, 1:nPCs);

    %% LDA + kNN
    nSamples = nTrials * nDirections;
    X_lda = zeros(nSamples, nPCs);
    labels = zeros(nSamples, 1);
    si = 0;
    for k = 1:nDirections
        for n = 1:nTrials
            si = si + 1;
            rates = trialRates{n, k};
            meanR = mean(rates(:, 1:classifyBins), 2);
            X_lda(si, :) = (W_pca' * (meanR - mu))';
            labels(si) = k;
        end
    end

    classMu = zeros(nDirections, nPCs);
    for k = 1:nDirections
        classMu(k, :) = mean(X_lda(labels == k, :), 1);
    end
    globalMu = mean(X_lda, 1);

    Sw = zeros(nPCs);
    Sb = zeros(nPCs);
    for k = 1:nDirections
        Xk = X_lda(labels == k, :) - classMu(k, :);
        Sw = Sw + Xk' * Xk;
        dk = classMu(k, :) - globalMu;
        Sb = Sb + sum(labels == k) * (dk' * dk);
    end

    Sw = Sw + 1e-6 * eye(nPCs);
    [Vlda, Dlda] = eig(Sw \ Sb);
    dlda = real(diag(Dlda));
    Vlda = real(Vlda);
    [~, ldaIdx] = sort(dlda, 'descend');
    nLDA = min(nDirections - 1, nPCs);
    W_lda = Vlda(:, ldaIdx(1:nLDA));

    knnRef = X_lda * W_lda;
    knnLabels = labels;

    %% Per-direction, per-phase ridge regression with neural velocity
    featureDim = nPCs * (nLags + 1) + nPCs + 1;
    phaseBoundary = classifyBins + phaseSplit;

    betaPosEarly = cell(nDirections, 1);
    betaVelEarly = cell(nDirections, 1);
    betaPosLate  = cell(nDirections, 1);
    betaVelLate  = cell(nDirections, 1);

    for k = 1:nDirections
        Xreg_e = []; Ypos_e = []; Yvel_e = [];
        Xreg_l = []; Ypos_l = []; Yvel_l = [];

        for n = 1:nTrials
            rates = trialRates{n, k};
            hp_mat = trialHP{n, k};
            nBins = size(rates, 2);
            pc = W_pca' * (rates - mu);
            startBin = max(classifyBins, nLags + 1);

            for b = startBin:nBins
                t = b * binSize;
                if t > size(hp_mat, 2), break; end

                feat = zeros(featureDim, 1);
                for lag = 0:nLags
                    feat(lag*nPCs+1 : (lag+1)*nPCs) = pc(:, b - lag);
                end
                dpcOff = nPCs * (nLags + 1);
                if b >= 2
                    feat(dpcOff+1 : dpcOff+nPCs) = pc(:, b) - pc(:, b-1);
                end
                feat(end) = 1;

                posT = hp_mat(1:2, t);
                tPrev = (b - 1) * binSize;
                if tPrev < 1, tPrev = 1; end
                velT = hp_mat(1:2, t) - hp_mat(1:2, tPrev);

                if phaseSplit == 0 || b <= phaseBoundary
                    Xreg_e = [Xreg_e; feat'];
                    Ypos_e = [Ypos_e; posT'];
                    Yvel_e = [Yvel_e; velT'];
                else
                    Xreg_l = [Xreg_l; feat'];
                    Ypos_l = [Ypos_l; posT'];
                    Yvel_l = [Yvel_l; velT'];
                end
            end
        end

        % Early
        XtX = Xreg_e' * Xreg_e;
        R = XtX + lambda * eye(featureDim);
        betaPosEarly{k} = R \ (Xreg_e' * Ypos_e);
        betaVelEarly{k} = R \ (Xreg_e' * Yvel_e);

        % Late (fall back to early if no late data or phaseSplit=0)
        if phaseSplit > 0 && size(Xreg_l, 1) > 0
            XtX = Xreg_l' * Xreg_l;
            R = XtX + lambda * eye(featureDim);
            betaPosLate{k} = R \ (Xreg_l' * Ypos_l);
            betaVelLate{k} = R \ (Xreg_l' * Yvel_l);
        else
            betaPosLate{k} = betaPosEarly{k};
            betaVelLate{k} = betaVelEarly{k};
        end
    end

    %% Store params
    params.binSize = binSize;
    params.nPCs = nPCs;
    params.nLags = nLags;
    params.mu = mu;
    params.W_pca = W_pca;
    params.W_lda = W_lda;
    params.knnRef = knnRef;
    params.knnLabels = knnLabels;
    params.betaPosEarly = betaPosEarly;
    params.betaVelEarly = betaVelEarly;
    params.betaPosLate  = betaPosLate;
    params.betaVelLate  = betaVelLate;
    params.classifyBins = classifyBins;
    params.nDirections = nDirections;
    params.phaseSplit = phaseSplit;
    params.kernel = gKernel;
    params.useGauss = useGauss;
    params.ensembleW = 0.5;
    params.kNN_k = 5;
    params.alphaMin = 0.4;
    params.alphaMax = 0.85;
    params.rampSteps = 15;
    params.cachedWeights = [];
end


%% ================================================================
%  Evaluate model RMSE on held-out data
%  ================================================================
function rmse = evaluateModel(testData, modelParameters)
    meanSqError = 0;
    nPredictions = 0;

    for tr = 1:size(testData, 1)
        for direc = 1:8
            decodedHandPos = [];
            times = 320:20:size(testData(tr, direc).spikes, 2);

            for t = times
                past_current_trial.trialId = testData(tr, direc).trialId;
                past_current_trial.spikes = testData(tr, direc).spikes(:, 1:t);
                past_current_trial.decodedHandPos = decodedHandPos;
                past_current_trial.startHandPos = testData(tr, direc).handPos(1:2, 1);

                [dx, dy, modelParameters] = positionEstimatorLocal( ...
                    past_current_trial, modelParameters);

                decodedPos = [dx; dy];
                decodedHandPos = [decodedHandPos, decodedPos];

                meanSqError = meanSqError + ...
                    norm(testData(tr, direc).handPos(1:2, t) - decodedPos)^2;
            end
            nPredictions = nPredictions + length(times);
        end
    end

    rmse = sqrt(meanSqError / nPredictions);
end


%% ================================================================
%  Local predictor (mirrors positionEstimator with all features)
%  ================================================================
function [x, y, newParams] = positionEstimatorLocal(test_data, params)
    newParams = params;

    binSize = params.binSize;
    nPCs = params.nPCs;
    nLags = params.nLags;
    mu = params.mu;
    W_pca = params.W_pca;
    classifyBins = params.classifyBins;
    nDirections = params.nDirections;
    ensW = params.ensembleW;
    phaseSplit = params.phaseSplit;

    spikes = test_data.spikes;
    T = size(spikes, 2);
    nNeurons = size(spikes, 1);

    %% Feature extraction
    if params.useGauss && ~isempty(params.kernel)
        samplePts = binSize:binSize:T;
        nBins = length(samplePts);
        rates = zeros(nNeurons, nBins);
        for i = 1:nNeurons
            smoothed = conv(spikes(i,:), params.kernel, 'same');
            rates(i,:) = sqrt(smoothed(samplePts));
        end
    else
        nBins = floor(T / binSize);
        rates = zeros(nNeurons, nBins);
        for b = 1:nBins
            rates(:, b) = sqrt(sum(spikes(:, (b-1)*binSize+1 : b*binSize), 2));
        end
    end

    pc = W_pca' * (rates - mu);

    isNew = isempty(test_data.decodedHandPos);

    if isNew || isempty(params.cachedWeights)
        meanPC = mean(pc(:, 1:classifyBins), 2);
        ldaFeat = params.W_lda' * meanPC;
        ref = params.knnRef;
        refLabels = params.knnLabels;
        kVal = params.kNN_k;
        dists = sqrt(sum((ref - ldaFeat').^2, 2));
        [sDists, sIdx] = sort(dists, 'ascend');
        kDists = max(sDists(1:kVal), 1e-10);
        kLab = refLabels(sIdx(1:kVal));
        invD = 1 ./ kDists;
        w = zeros(nDirections, 1);
        for k = 1:nDirections
            w(k) = sum(invD(kLab == k));
        end
        w = w / sum(w);
        newParams.cachedWeights = w;
    else
        w = params.cachedWeights;
    end

    %% Feature vector with neural velocity
    currentBin = nBins;
    featureDim = nPCs * (nLags + 1) + nPCs + 1;
    feat = zeros(featureDim, 1);
    for lag = 0:nLags
        b = currentBin - lag;
        if b >= 1
            feat(lag*nPCs+1 : (lag+1)*nPCs) = pc(:, b);
        end
    end
    dpcOff = nPCs * (nLags + 1);
    if currentBin >= 2
        feat(dpcOff+1 : dpcOff+nPCs) = pc(:, currentBin) - pc(:, currentBin-1);
    end
    feat(end) = 1;

    %% Phase selection
    phaseBoundary = classifyBins + phaseSplit;
    useEarly = (phaseSplit == 0) || (currentBin <= phaseBoundary);

    if isempty(test_data.decodedHandPos)
        lastPos = test_data.startHandPos;
    else
        lastPos = test_data.decodedHandPos(:, end);
    end

    pPos = zeros(2, 1);
    pVel = zeros(2, 1);
    for k = 1:nDirections
        if w(k) > 0
            if useEarly
                pPos = pPos + w(k) * (params.betaPosEarly{k}' * feat);
                pVel = pVel + w(k) * (params.betaVelEarly{k}' * feat);
            else
                pPos = pPos + w(k) * (params.betaPosLate{k}' * feat);
                pVel = pVel + w(k) * (params.betaVelLate{k}' * feat);
            end
        end
    end

    if isNew
        pred = pPos;
    else
        posFromVel = lastPos + pVel;
        pred = ensW * pPos + (1 - ensW) * posFromVel;
    end

    %% Adaptive smoothing
    if ~isNew
        nDecoded = size(test_data.decodedHandPos, 2);
        alphaMin = params.alphaMin;
        alphaMax = params.alphaMax;
        rampSteps = params.rampSteps;
        alpha = alphaMin + (alphaMax - alphaMin) * min(nDecoded / rampSteps, 1);
        pred = alpha * pred + (1 - alpha) * lastPos;
    end

    x = pred(1);
    y = pred(2);
end
