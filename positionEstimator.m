function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
    newModelParameters = modelParameters;

    binSize = modelParameters.binSize;
    nPCs = modelParameters.nPCs;
    nLags = modelParameters.nLags;
    mu = modelParameters.mu;
    W_pca = modelParameters.W_pca;
    classifyBins = modelParameters.classifyBins;
    nDirections = modelParameters.nDirections;
    ensW = modelParameters.ensembleW;
    phaseSplit = modelParameters.phaseSplit;
    kernel = modelParameters.kernel;
    keptNeurons = modelParameters.keptNeurons;

    spikes = test_data.spikes;
    T = size(spikes, 2);
    nNeuronsRaw = size(spikes, 1);

    %% Gaussian smooth + sqrt + resample (identical to training)
    samplePts = binSize:binSize:T;
    nBins = length(samplePts);

    ratesRaw = zeros(nNeuronsRaw, nBins);
    for i = 1:nNeuronsRaw
        smoothed = conv(spikes(i,:), kernel, 'same');
        ratesRaw(i,:) = sqrt(smoothed(samplePts));
    end

    rates = ratesRaw(keptNeurons, :);

    %% PCA projection
    pc = W_pca' * (rates - mu);

    %% Direction classification via LDA + weighted kNN
    % Reclassify at every call using mean PC over all available bins
    meanPC = mean(pc, 2);
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

    %% Build feature vector (PC lags + neural velocity + bias)
    currentBin = nBins;
    featureDim = nPCs * (nLags + 1) + nPCs + 1;
    feat = zeros(featureDim, 1);

    for lag = 0:nLags
        b = currentBin - lag;
        if b >= 1
            feat(lag*nPCs+1 : (lag+1)*nPCs) = pc(:, b);
        end
    end

    dpcOffset = nPCs * (nLags + 1);
    if currentBin >= 2
        feat(dpcOffset+1 : dpcOffset+nPCs) = pc(:, currentBin) - pc(:, currentBin-1);
    end

    feat(end) = 1;

    %% Select early or late phase model
    phaseBoundary = classifyBins + phaseSplit;
    useEarly = (phaseSplit == 0) || (currentBin <= phaseBoundary);

    %% Weighted regression prediction across directions
    isNew = isempty(test_data.decodedHandPos);

    if isNew
        lastPos = test_data.startHandPos;
    else
        lastPos = test_data.decodedHandPos(:, end);
    end

    pPos = zeros(2, 1);
    pVel = zeros(2, 1);
    for k = 1:nDirections
        if w(k) > 0
            if useEarly
                pPos = pPos + w(k) * (modelParameters.betaPosEarly{k}' * feat);
                pVel = pVel + w(k) * (modelParameters.betaVelEarly{k}' * feat);
            else
                pPos = pPos + w(k) * (modelParameters.betaPosLate{k}' * feat);
                pVel = pVel + w(k) * (modelParameters.betaVelLate{k}' * feat);
            end
        end
    end

    %% Ensemble position + velocity models
    if isNew
        pred = pPos;
    else
        posFromVel = lastPos + pVel;
        pred = ensW * pPos + (1 - ensW) * posFromVel;
    end

    %% Adaptive exponential smoothing
    if ~isNew
        nDecoded = size(test_data.decodedHandPos, 2);
        alphaMin = modelParameters.alphaMin;
        alphaMax = modelParameters.alphaMax;
        rampSteps = modelParameters.rampSteps;
        alpha = alphaMin + (alphaMax - alphaMin) * min(nDecoded / rampSteps, 1);
        pred = alpha * pred + (1 - alpha) * lastPos;
    end

    x = pred(1);
    y = pred(2);
end
