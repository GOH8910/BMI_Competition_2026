function [x, y, newModelParameters] = positionEstimator5(past_current_trial, modelParameters)
    newModelParameters = modelParameters;

    nDirections = modelParameters.nDirections;
    group = modelParameters.group;
    startBin = modelParameters.startBin;
    maxTrainBin = modelParameters.trainBins(end);

    % PRO preprocessing online
    trialProcess = bin_and_sqrt_single(past_current_trial, group, 1);
    trialFinal = get_firing_rates_single(trialProcess, group, modelParameters.smoothWin);
    rates = trialFinal.rates;

    % Remove low-firing neurons (PRO filter indices)
    rates(modelParameters.lowFirers, :) = [];

    nBins = size(rates, 2);
    bUse = min(max(startBin, nBins), maxTrainBin);
    [~, bi] = min(abs(modelParameters.trainBins - bUse));
    classModel = modelParameters.classBank(bi);

    % Z-score + soft direction probabilities (v3-style)
    ratesZ = (rates - modelParameters.muRate) ./ modelParameters.sdRate;
    pc = modelParameters.W_pca' * ratesZ;
    b0c = max(1, classModel.bin - modelParameters.regWinBins + 1);
    clsFeat = mean(pc(:,b0c:classModel.bin), 2);
    ldaFeat = (classModel.W_lda' * clsFeat)';
    rawProb = knnDirProb(ldaFeat, classModel.knnRef, classModel.knnLabels, ...
        modelParameters.kNN_k, nDirections);

    dBins = max(0, bUse - startBin);
    frac = min(dBins / max(modelParameters.temperatureRampBins, 1), 1);
    temp = modelParameters.temperatureEarly + ...
        (modelParameters.temperatureLate - modelParameters.temperatureEarly) * frac;
    dirProb = softmaxWithTemp(log(rawProb + 1e-12), temp);

    isNew = isempty(past_current_trial.decodedHandPos);
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

    % PCR delta-position predictions mixed by soft probabilities
    L = modelParameters.regWinBins;
    nKeep = size(ratesZ,1);
    b0 = max(1, classModel.bin - L + 1);
    rwin = ratesZ(:, b0:classModel.bin);
    if size(rwin,2) < L
        pad = zeros(nKeep, L - size(rwin,2));
        rwin = [pad, rwin];
    end
    rmean = mean(ratesZ(:,1:classModel.bin), 2);
    xFeat = [rwin(:); rmean]';

    tNow = size(past_current_trial.spikes, 2);
    softDp = zeros(2,1);
    softAvgDisp = zeros(2,1);
    for k = 1:nDirections
        regDp = modelParameters.regDpBank{k, bi};
        xc = xFeat - regDp.xMean;
        z = xc * regDp.V;
        predDp = (z * regDp.Bz) + regDp.yMean;
        softDp = softDp + dirProb(k) * predDp';

        avgD = modelParameters.avgDisp{k};
        if tNow <= size(avgD,2)
            dAvg = avgD(:,tNow);
        else
            dAvg = avgD(:,end);
        end
        softAvgDisp = softAvgDisp + dirProb(k) * dAvg;
    end

    % Robustify step by magnitude cap
    dpNorm = norm(softDp);
    if dpNorm > modelParameters.maxStepMm
        softDp = softDp * (modelParameters.maxStepMm / dpNorm);
    end

    % Integrate delta-position; let step prediction take over with confidence/time
    if isNew
        pred = past_current_trial.startHandPos + 0.2 * softDp + 0.8 * softAvgDisp;
    else
        lastPos = past_current_trial.decodedHandPos(:, end);
        predStep = lastPos + softDp;
        predAvg = past_current_trial.startHandPos + softAvgDisp;
        conf = max(dirProb);
        conf0 = 1 / nDirections;
        nDecoded = size(past_current_trial.decodedHandPos, 2);
        timeRamp = min(nDecoded / max(modelParameters.stepRampBins, 1), 1);
        wStep = (modelParameters.stepWeightBase + ...
            modelParameters.stepWeightConfGain * max(0, conf - conf0) / (1 - conf0)) * timeRamp;
        wStep = min(max(wStep, 0.05), 0.95);
        pred = wStep * predStep + (1 - wStep) * predAvg;
    end

    % Temporal smoothing
    if ~isNew
        lastPos = past_current_trial.decodedHandPos(:, end);
        nDecoded = size(past_current_trial.decodedHandPos, 2);
        alpha = modelParameters.alphaMin + ...
            (modelParameters.alphaMax - modelParameters.alphaMin) * ...
            min(nDecoded / max(modelParameters.rampSteps,1), 1);
        pred = alpha * pred + (1 - alpha) * lastPos;
    end

    x = pred(1);
    y = pred(2);
end

function trialProcessed = bin_and_sqrt_single(trial, group, to_sqrt)
    allSpikes = trial.spikes;
    nNeurons = size(allSpikes,1);
    nPoints = size(allSpikes,2);
    tNew = 1:group:nPoints+1;
    spikes = zeros(nNeurons, numel(tNew)-1);
    for k = 1:numel(tNew)-1
        spikes(:,k) = sum(allSpikes(:, tNew(k):tNew(k+1)-1), 2);
    end
    if to_sqrt
        spikes = sqrt(spikes);
    end
    trialProcessed.spikes = spikes;
end

function trialFinal = get_firing_rates_single(trialProcessed, group, scale_window)
    win = 10 * (scale_window / group);
    normstd = scale_window / group;
    alpha = (win - 1) / (2 * normstd);
    temp1 = -(win-1)/2 : (win-1)/2;
    gausstemp = exp((-1/2) * (alpha * temp1 / ((win-1)/2)).^2)';
    gaussian_window = gausstemp / sum(gausstemp);

    hold_rates = zeros(size(trialProcessed.spikes,1), size(trialProcessed.spikes,2));
    for k = 1:size(trialProcessed.spikes,1)
        hold_rates(k,:) = conv(trialProcessed.spikes(k,:), gaussian_window, 'same') / (group/1000);
    end
    trialFinal.rates = hold_rates;
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
