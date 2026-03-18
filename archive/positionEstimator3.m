function [x, y, newModelParameters] = positionEstimator3(test_data, modelParameters)
    newModelParameters = modelParameters;

    binSize = modelParameters.binSize;
    classifyBins = modelParameters.classifyBins;
    nDirections = modelParameters.nDirections;

    % Build rates from available spikes
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

    % PCA projection
    pc = modelParameters.W_pca' * (rates - modelParameters.mu);

    % Pick classification bin/model matching current available time
    bUse = min(max(classifyBins, nBins), modelParameters.classBins(end));
    [~, bi] = min(abs(modelParameters.classBins - bUse));
    classModel = modelParameters.classBank(bi);

    clsFeat = mean(pc(:,1:classModel.bin), 2);
    ldaFeat = (classModel.W_lda' * clsFeat)';
    rawProb = knnDirProb(ldaFeat, classModel.knnRef, classModel.knnLabels, ...
        modelParameters.kNN_k, nDirections);

    % Temperature schedule: higher early, lower later
    dBins = max(0, nBins - classifyBins);
    frac = min(dBins / max(modelParameters.temperatureRampBins, 1), 1);
    temp = modelParameters.temperatureEarly + ...
        (modelParameters.temperatureLate - modelParameters.temperatureEarly) * frac;
    dirProb = softmaxWithTemp(log(rawProb + 1e-12), temp);

    % Smooth direction probabilities over time within trial
    isNew = isempty(test_data.decodedHandPos);
    if isNew
        newModelParameters.prevDirProb = dirProb;
    else
        gamma = modelParameters.dirProbSmoothing;
        if isempty(modelParameters.prevDirProb)
            prev = dirProb;
        else
            prev = modelParameters.prevDirProb;
        end
        dirProb = gamma * dirProb + (1 - gamma) * prev;
        dirProb = dirProb / sum(dirProb);
        newModelParameters.prevDirProb = dirProb;
    end

    % Progressive trajectory feature at current bin
    bFeat = min(max(classifyBins, nBins), modelParameters.maxFeatBins);
    testFeat = mean(pc(:,1:bFeat), 2)';

    % Soft direction-weighted trajectory prediction
    tNow = size(spikes, 2);
    softAvgDisp = zeros(2,1);
    softKnnDisp = zeros(2,1);

    for k = 1:nDirections
        % Avg displacement by direction
        avgD = modelParameters.avgDisp{k};
        if tNow <= size(avgD, 2)
            dAvg = avgD(:, tNow);
        else
            dAvg = avgD(:, end);
        end
        softAvgDisp = softAvgDisp + dirProb(k) * dAvg;

        % kNN displacement by direction, using time-matched feature bank
        trainFeatK = modelParameters.trajFeatsByBin{k, bFeat};
        dists = sqrt(sum((trainFeatK - testFeat).^2, 2));
        [~, idx] = sort(dists, 'ascend');
        kUse = min(modelParameters.knnTrajK, length(idx));
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

    % Confidence-adaptive blend between avg and kNN trajectory
    conf = max(dirProb);
    conf0 = 1 / nDirections;
    alpha = modelParameters.hybridAlphaBase + ...
            modelParameters.hybridAlphaConfGain * max(0, conf - conf0) / (1 - conf0);
    alpha = min(max(alpha, 0), 1);

    dFinal = (1 - alpha) * softAvgDisp + alpha * softKnnDisp;
    pred = test_data.startHandPos + dFinal;

    % Temporal smoothing against last decoded point
    if ~isNew
        lastPos = test_data.decodedHandPos(:, end);
        nDecoded = size(test_data.decodedHandPos, 2);
        aMin = modelParameters.alphaMin;
        aMax = modelParameters.alphaMax;
        rampSteps = modelParameters.rampSteps;
        a = aMin + (aMax - aMin) * min(nDecoded / max(rampSteps, 1), 1);
        pred = a * pred + (1 - a) * lastPos;
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
