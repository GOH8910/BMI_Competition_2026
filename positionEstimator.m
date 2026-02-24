function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
    newModelParameters = modelParameters;

    binSize = modelParameters.binSize;
    nPCs = modelParameters.nPCs;
    nLags = modelParameters.nLags;
    mu = modelParameters.mu;
    W_pca = modelParameters.W_pca;
    classifyBins = modelParameters.classifyBins;
    nDirections = modelParameters.nDirections;
    alpha = modelParameters.alpha;
    ensW = modelParameters.ensembleW;

    spikes = test_data.spikes;
    T = size(spikes, 2);
    nBins = floor(T / binSize);
    nNeurons = size(spikes, 1);

    %% Bin + sqrt (identical to training)
    rates = zeros(nNeurons, nBins);
    for b = 1:nBins
        rates(:, b) = sqrt(sum(spikes(:, (b-1)*binSize+1 : b*binSize), 2));
    end

    %% PCA projection
    pc = W_pca' * (rates - mu);

    %% Direction classification via LDA + weighted kNN (once per trial)
    isNew = isempty(test_data.decodedHandPos);

    if isNew || isempty(modelParameters.cachedWeights)
        meanPC = mean(pc(:, 1:classifyBins), 2);
        ldaFeat = modelParameters.W_lda' * meanPC;

        ref = modelParameters.knnRef;
        refLabels = modelParameters.knnLabels;
        kVal = modelParameters.kNN_k;

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

        newModelParameters.cachedWeights = w;
    else
        w = modelParameters.cachedWeights;
    end

    %% Build feature vector (current bin + lagged bins + bias)
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

    %% Weighted regression prediction across directions
    if isempty(test_data.decodedHandPos)
        lastPos = test_data.startHandPos;
    else
        lastPos = test_data.decodedHandPos(:, end);
    end

    pPos = zeros(2, 1);
    pVel = zeros(2, 1);
    for k = 1:nDirections
        if w(k) > 0
            pPos = pPos + w(k) * (modelParameters.betaPos{k}' * feat);
            pVel = pVel + w(k) * (modelParameters.betaVel{k}' * feat);
        end
    end

    %% Ensemble position + velocity models
    if isNew
        % First call: velocity integration from startHandPos is unreliable,
        % so rely on the position model only
        pred = pPos;
    else
        posFromVel = lastPos + pVel;
        pred = ensW * pPos + (1 - ensW) * posFromVel;
    end

    %% Exponential smoothing with previous decoded position
    if ~isNew
        pred = alpha * pred + (1 - alpha) * lastPos;
    end

    x = pred(1);
    y = pred(2);
end
