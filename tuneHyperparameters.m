function tuneHyperparameters()
    load monkeydata0.mat

    rng(2013);
    ix = randperm(length(trial));

    allTraining = trial(ix(1:50), :);
    nFolds = 5;
    foldSize = floor(size(allTraining, 1) / nFolds);

    %% --- Phase 1: Tune training-time params (nPCs, lambda) ---
    nPCs_grid = [10, 15, 20, 25, 30];
    lambda_grid = [1e-4, 1e-3, 1e-2, 1e-1, 1];

    fprintf('=== Phase 1: Tuning nPCs and lambda ===\n');
    bestRMSE1 = inf;
    bestNPCs = 20;
    bestLambda = 0.01;

    for pi = 1:length(nPCs_grid)
        for li = 1:length(lambda_grid)
            nPCs_try = nPCs_grid(pi);
            lambda_try = lambda_grid(li);

            foldErrors = zeros(nFolds, 1);
            for fold = 1:nFolds
                valIdx = (fold-1)*foldSize+1 : fold*foldSize;
                trainIdx = setdiff(1:size(allTraining,1), valIdx);

                trainData = allTraining(trainIdx, :);
                valData = allTraining(valIdx, :);

                params = trainWithParams(trainData, nPCs_try, lambda_try);
                foldErrors(fold) = evaluateModel(valData, params);
            end

            avgRMSE = mean(foldErrors);
            fprintf('  nPCs=%2d  lambda=%.0e  -> RMSE=%.4f\n', ...
                nPCs_try, lambda_try, avgRMSE);

            if avgRMSE < bestRMSE1
                bestRMSE1 = avgRMSE;
                bestNPCs = nPCs_try;
                bestLambda = lambda_try;
            end
        end
    end
    fprintf('Best Phase 1: nPCs=%d, lambda=%.0e, RMSE=%.4f\n\n', ...
        bestNPCs, bestLambda, bestRMSE1);

    %% --- Phase 2: Tune prediction-time params (k, alpha, ensembleW) ---
    k_grid = [3, 5, 7, 11];
    alpha_grid = [0.3, 0.5, 0.7, 0.9];
    ensW_grid = [0.3, 0.5, 0.7];

    fprintf('=== Phase 2: Tuning k, alpha, ensembleW ===\n');
    fprintf('(Using best nPCs=%d, lambda=%.0e from Phase 1)\n', bestNPCs, bestLambda);
    bestRMSE2 = inf;
    bestK = 5;
    bestAlpha = 0.5;
    bestEnsW = 0.5;

    % Train once with best training params on a single fold for speed
    trainIdx = 1:40;
    valIdx = 41:50;
    trainData = allTraining(trainIdx, :);
    valData = allTraining(valIdx, :);
    baseParams = trainWithParams(trainData, bestNPCs, bestLambda);

    for ki = 1:length(k_grid)
        for ai = 1:length(alpha_grid)
            for ei = 1:length(ensW_grid)
                params = baseParams;
                params.kNN_k = k_grid(ki);
                params.alpha = alpha_grid(ai);
                params.ensembleW = ensW_grid(ei);

                rmse = evaluateModel(valData, params);
                fprintf('  k=%2d  alpha=%.1f  ensW=%.1f  -> RMSE=%.4f\n', ...
                    k_grid(ki), alpha_grid(ai), ensW_grid(ei), rmse);

                if rmse < bestRMSE2
                    bestRMSE2 = rmse;
                    bestK = k_grid(ki);
                    bestAlpha = alpha_grid(ai);
                    bestEnsW = ensW_grid(ei);
                end
            end
        end
    end

    fprintf('\n========== BEST HYPERPARAMETERS ==========\n');
    fprintf('  nPCs       = %d\n', bestNPCs);
    fprintf('  lambda     = %.0e\n', bestLambda);
    fprintf('  kNN_k      = %d\n', bestK);
    fprintf('  alpha      = %.1f\n', bestAlpha);
    fprintf('  ensembleW  = %.1f\n', bestEnsW);
    fprintf('  Phase1 CV RMSE = %.4f\n', bestRMSE1);
    fprintf('  Phase2 RMSE    = %.4f\n', bestRMSE2);
    fprintf('==========================================\n');
end


function params = trainWithParams(training_data, nPCs_target, lambda)
    binSize = 20;
    nDirections = size(training_data, 2);
    nTrials = size(training_data, 1);
    nNeurons = size(training_data(1,1).spikes, 1);
    classifyBins = 320 / binSize;
    nLags = 2;

    allRates = [];
    trialRates = cell(nTrials, nDirections);
    trialHP = cell(nTrials, nDirections);

    for k = 1:nDirections
        for n = 1:nTrials
            spikes = training_data(n, k).spikes;
            T = size(spikes, 2);
            nBins = floor(T / binSize);
            rates = zeros(nNeurons, nBins);
            for b = 1:nBins
                rates(:, b) = sqrt(sum(spikes(:, (b-1)*binSize+1 : b*binSize), 2));
            end
            trialRates{n, k} = rates;
            trialHP{n, k} = training_data(n, k).handPos;
            allRates = [allRates, rates];
        end
    end

    mu = mean(allRates, 2);
    centered = allRates - mu;
    C = (centered * centered') / (size(centered, 2) - 1);
    [V, D] = eig(C);
    d = diag(D);
    [d, idx] = sort(d, 'descend');
    V = V(:, idx);

    nPCs = min(nPCs_target, size(V, 2));
    W_pca = V(:, 1:nPCs);

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

    featureDim = nPCs * (nLags + 1) + 1;
    betaPos = cell(nDirections, 1);
    betaVel = cell(nDirections, 1);

    for k = 1:nDirections
        Xreg = [];
        Ypos = [];
        Yvel = [];
        for n = 1:nTrials
            rates = trialRates{n, k};
            hp = trialHP{n, k};
            nBins = size(rates, 2);
            pc = W_pca' * (rates - mu);
            startBin = max(classifyBins, nLags + 1);
            for b = startBin:nBins
                t = b * binSize;
                if t > size(hp, 2), break; end
                feat = zeros(featureDim, 1);
                for lag = 0:nLags
                    feat(lag*nPCs+1 : (lag+1)*nPCs) = pc(:, b - lag);
                end
                feat(end) = 1;
                posT = hp(1:2, t);
                tPrev = (b - 1) * binSize;
                if tPrev < 1, tPrev = 1; end
                velT = hp(1:2, t) - hp(1:2, tPrev);
                Xreg = [Xreg; feat'];
                Ypos = [Ypos; posT'];
                Yvel = [Yvel; velT'];
            end
        end
        XtX = Xreg' * Xreg;
        R = XtX + lambda * eye(featureDim);
        betaPos{k} = R \ (Xreg' * Ypos);
        betaVel{k} = R \ (Xreg' * Yvel);
    end

    params.binSize = binSize;
    params.nPCs = nPCs;
    params.nLags = nLags;
    params.mu = mu;
    params.W_pca = W_pca;
    params.W_lda = W_lda;
    params.knnRef = knnRef;
    params.knnLabels = knnLabels;
    params.betaPos = betaPos;
    params.betaVel = betaVel;
    params.classifyBins = classifyBins;
    params.nDirections = nDirections;
    params.alpha = 0.5;
    params.ensembleW = 0.5;
    params.kNN_k = 5;
    params.cachedWeights = [];
end


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


function [x, y, newParams] = positionEstimatorLocal(test_data, params)
    newParams = params;

    binSize = params.binSize;
    nPCs = params.nPCs;
    nLags = params.nLags;
    mu = params.mu;
    W_pca = params.W_pca;
    classifyBins = params.classifyBins;
    nDirections = params.nDirections;
    alpha = params.alpha;
    ensW = params.ensembleW;

    spikes = test_data.spikes;
    T = size(spikes, 2);
    nBins = floor(T / binSize);
    nNeurons = size(spikes, 1);

    rates = zeros(nNeurons, nBins);
    for b = 1:nBins
        rates(:, b) = sqrt(sum(spikes(:, (b-1)*binSize+1 : b*binSize), 2));
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

    currentBin = nBins;
    featureDim = nPCs * (nLags + 1) + 1;
    feat = zeros(featureDim, 1);
    for lag = 0:nLags
        b = currentBin - lag;
        if b >= 1
            feat(lag*nPCs+1 : (lag+1)*nPCs) = pc(:, b);
        end
    end
    feat(end) = 1;

    if isempty(test_data.decodedHandPos)
        lastPos = test_data.startHandPos;
    else
        lastPos = test_data.decodedHandPos(:, end);
    end

    pPos = zeros(2, 1);
    pVel = zeros(2, 1);
    for k = 1:nDirections
        if w(k) > 0
            pPos = pPos + w(k) * (params.betaPos{k}' * feat);
            pVel = pVel + w(k) * (params.betaVel{k}' * feat);
        end
    end

    if isNew
        pred = pPos;
    else
        posFromVel = lastPos + pVel;
        pred = ensW * pPos + (1 - ensW) * posFromVel;
    end

    if ~isNew
        pred = alpha * pred + (1 - alpha) * lastPos;
    end

    x = pred(1);
    y = pred(2);
end
