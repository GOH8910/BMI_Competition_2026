function [x, y, newModelParameters] = positionEstimator4(test_data, modelParameters)
    newModelParameters = modelParameters;

    binSize = modelParameters.binSize;
    classifyBins = modelParameters.classifyBins;
    nDirections = modelParameters.nDirections;

    spikes = test_data.spikes;
    T = size(spikes, 2);
    samplePts = binSize:binSize:T;
    nBins = length(samplePts);

    % Same preprocessing as training
    nNeuronsRaw = size(spikes, 1);
    ratesRaw = zeros(nNeuronsRaw, nBins);
    for i = 1:nNeuronsRaw
        smoothed = conv(spikes(i,:), modelParameters.kernel, 'same');
        ratesRaw(i,:) = sqrt(smoothed(samplePts));
    end
    rates = ratesRaw(modelParameters.keptNeurons, :);
    pc = modelParameters.W_pca' * (rates - modelParameters.mu);

    % Select nearest trained time-bin model
    bUse = min(max(classifyBins, nBins), modelParameters.timeBins(end));
    [~, bi] = min(abs(modelParameters.timeBins - bUse));
    classModel = modelParameters.classBank(bi);

    % Soft direction probabilities
    clsFeat = mean(pc(:,1:classModel.bin), 2);
    ldaFeat = (classModel.W_lda' * clsFeat)';
    rawProb = knnDirProb(ldaFeat, classModel.knnRef, classModel.knnLabels, ...
        modelParameters.kNN_k, nDirections);

    dBins = max(0, nBins - classifyBins);
    frac = min(dBins / max(modelParameters.temperatureRampBins, 1), 1);
    temp = modelParameters.temperatureEarly + ...
        (modelParameters.temperatureLate - modelParameters.temperatureEarly) * frac;
    dirProb = softmaxWithTemp(log(rawProb + 1e-12), temp);

    isNew = isempty(test_data.decodedHandPos);
    if isNew
        newModelParameters.prevDirProb = dirProb;
    else
        gamma = modelParameters.dirProbSmoothing;
        prev = modelParameters.prevDirProb;
        if isempty(prev)
            prev = dirProb;
        end
        dirProb = gamma * dirProb + (1 - gamma) * prev;
        dirProb = dirProb / sum(dirProb);
        newModelParameters.prevDirProb = dirProb;
    end

    % Feature for regression model at current bin
    f = mean(pc(:,1:classModel.bin), 2);
    f = [f; 1];

    % Soft blend of per-direction regression and avg displacement
    tNow = size(spikes, 2);
    softRegDisp = zeros(2,1);
    softAvgDisp = zeros(2,1);

    for k = 1:nDirections
        B = modelParameters.regBank{k, bi};
        dReg = B' * f;
        softRegDisp = softRegDisp + dirProb(k) * dReg;

        avgD = modelParameters.avgDisp{k};
        if tNow <= size(avgD, 2)
            dAvg = avgD(:, tNow);
        else
            dAvg = avgD(:, end);
        end
        softAvgDisp = softAvgDisp + dirProb(k) * dAvg;
    end

    conf = max(dirProb);
    conf0 = 1 / nDirections;
    wReg = modelParameters.regBlendBase + ...
           modelParameters.regBlendConfGain * max(0, conf - conf0) / (1 - conf0);
    wReg = min(max(wReg, 0), 1);

    dFinal = wReg * softRegDisp + (1 - wReg) * softAvgDisp;
    pred = test_data.startHandPos + dFinal;

    % Temporal smoothing
    if ~isNew
        lastPos = test_data.decodedHandPos(:, end);
        nDecoded = size(test_data.decodedHandPos, 2);
        alpha = modelParameters.alphaMin + ...
            (modelParameters.alphaMax - modelParameters.alphaMin) * ...
            min(nDecoded / max(modelParameters.rampSteps, 1), 1);
        pred = alpha * pred + (1 - alpha) * lastPos;
    end

    x = pred(1);
    y = pred(2);
end

function probs = knnDirProb(ldaFeat, knnRef, knnLabels, kVal, nDirs)
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

function P = softmaxWithTemp(logits, temperature)
    logits = logits / max(temperature, 1e-6);
    logits = logits - max(logits);
    P = exp(logits);
    P = P / sum(P);
end
