function [x, y, newModelParameters] = positionEstimator2(test_data, modelParameters)
    newModelParameters = modelParameters;

    binSize = modelParameters.binSize;
    classifyBins = modelParameters.classifyBins;
    nDirections = modelParameters.nDirections;
    kTraj = modelParameters.knnTrajK;
    temp = modelParameters.temperature;
    hybridAlpha = modelParameters.hybridAlpha;

    % Build rates from currently available spikes
    spikes = test_data.spikes;
    T = size(spikes, 2);
    samplePts = binSize:binSize:T;
    nBins = length(samplePts);
    nNeuronsRaw = size(spikes, 1);
    ratesRaw = zeros(nNeuronsRaw, nBins);
    for i = 1:nNeuronsRaw
        smoothed = conv(spikes(i,:), modelParameters.kernel, 'same');
        ratesRaw(i,:) = sqrt(smoothed(samplePts));
    end
    rates = ratesRaw(modelParameters.keptNeurons, :);

    % PCA features
    pc = modelParameters.W_pca' * (rates - modelParameters.mu);
    nUseBins = min(classifyBins, size(pc, 2));
    feat = mean(pc(:, 1:nUseBins), 2)';

    % Direction probabilities via LDA+kNN reference set
    ldaFeat = (modelParameters.W_lda' * mean(pc(:, 1:nUseBins), 2))';
    dirProb = computeDirProb(ldaFeat, modelParameters.knnRef, ...
        modelParameters.knnLabels, modelParameters.kNN_k, nDirections);
    dirProb = softmaxWithTemp(dirProb, temp);

    % Predict displacement at current time by combining directions
    tNow = size(test_data.spikes, 2);
    softAvgDisp = zeros(2,1);
    softKnnDisp = zeros(2,1);
    for k = 1:nDirections
        % Average displacement for direction k
        avgD = modelParameters.avgDisp{k};
        if tNow <= size(avgD, 2)
            dAvg = avgD(:, tNow);
        else
            dAvg = avgD(:, end);
        end
        softAvgDisp = softAvgDisp + dirProb(k) * dAvg;

        % kNN displacement for direction k
        trainFeatK = modelParameters.trajFeats{k};
        dists = sqrt(sum((trainFeatK - feat).^2, 2));
        [~, idx] = sort(dists, 'ascend');
        kUse = min(kTraj, length(idx));
        idx = idx(1:kUse);
        w = 1 ./ (dists(idx) + 1e-8);
        w = w / sum(w);

        dKnn = zeros(2,1);
        for j = 1:kUse
            traj = modelParameters.trajDisp{k}{idx(j)};
            if tNow <= size(traj, 2)
                d = traj(:, tNow);
            else
                d = traj(:, end);
            end
            dKnn = dKnn + w(j) * d;
        end
        softKnnDisp = softKnnDisp + dirProb(k) * dKnn;
    end

    dFinal = (1 - hybridAlpha) * softAvgDisp + hybridAlpha * softKnnDisp;
    pred = test_data.startHandPos + dFinal;

    % Optional temporal smoothing against last decoded point
    if ~isempty(test_data.decodedHandPos)
        lastPos = test_data.decodedHandPos(:, end);
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

function probs = computeDirProb(ldaFeat, knnRef, knnLabels, kVal, nDirs)
    dists = sqrt(sum((knnRef - ldaFeat).^2, 2));
    [sDists, sIdx] = sort(dists, 'ascend');
    kUse = min(kVal, length(sDists));
    kDists = max(sDists(1:kUse), 1e-10);
    kLab = knnLabels(sIdx(1:kUse));
    invD = 1 ./ kDists;

    w = zeros(nDirs, 1);
    for k = 1:nDirs
        w(k) = sum(invD(kLab == k));
    end
    if sum(w) > 0
        probs = w / sum(w);
    else
        probs = ones(nDirs,1) / nDirs;
    end
end

function P = softmaxWithTemp(baseProb, temperature)
    logP = log(baseProb + 1e-12) / temperature;
    logP = logP - max(logP);
    P = exp(logP);
    P = P / sum(P);
end
