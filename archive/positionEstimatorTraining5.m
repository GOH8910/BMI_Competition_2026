function modelParameters = positionEstimatorTraining5(training_data)
    % v5 hybrid:
    % - PRO preprocessing (bin+sqrt, then Gaussian rate smoothing)
    % - PRO low-firing neuron filter criterion
    % - Soft direction probabilities (time-dependent LDA+kNN)
    % - Per-direction/time PCR-style delta-position regression mixed by soft probabilities

    nDirections = size(training_data, 2);
    nTrials = size(training_data, 1);
    nNeuronsRaw = size(training_data(1,1).spikes, 1);

    group = 20;      % bin size (ms)
    smoothWin = 50;  % PRO scale_window
    startTime = 320;
    endTime = 560;
    startBin = startTime / group;
    endBin = endTime / group;

    % Inference hyperparameters
    modelParameters.kNN_k = 5;
    modelParameters.temperatureEarly = 3.5;
    modelParameters.temperatureLate = 1.5;
    modelParameters.temperatureRampBins = 12;
    modelParameters.dirProbSmoothing = 0.75;
    modelParameters.maxStepMm = 5.0;
    modelParameters.stepWeightBase = 0.05;
    modelParameters.stepWeightConfGain = 0.35;
    modelParameters.stepRampBins = 10;
    modelParameters.alphaMin = 0.35;
    modelParameters.alphaMax = 0.85;
    modelParameters.rampSteps = 15;
    modelParameters.regWinBins = 10; % 200 ms causal window at 20 ms bins

    pcaDimReg = 20;  % PRO-like PCR dimensionality (kept moderate for speed/stability)
    lambdaReg = 1e-2;

    % PRO preprocessing on all training trials
    trialProcessed = bin_and_sqrt_set(training_data, group, 1);
    trialFinal = get_firing_rates_set(trialProcessed, group, smoothWin);

    % Build PRO low-firer matrix using first 560ms only
    firingData560 = zeros(nNeuronsRaw * endBin, nTrials * nDirections);
    for k = 1:nDirections
        for n = 1:nTrials
            col = nTrials * (k - 1) + n;
            rates = trialFinal(n,k).rates;
            for b = 1:endBin
                firingData560(nNeuronsRaw*(b-1)+1:nNeuronsRaw*b, col) = rates(:, b);
            end
        end
    end

    lowFirers = [];
    for x = 1:nNeuronsRaw
        checkRate = mean(mean(firingData560(x:nNeuronsRaw:end, :)));
        if checkRate < 0.5
            lowFirers = [lowFirers, x];
        end
    end
    keptNeurons = true(nNeuronsRaw, 1);
    keptNeurons(lowFirers) = false;

    % Keep only non-low-firing neurons and prepare trajectories
    trialRates = cell(nTrials, nDirections);
    trialRatesZ = cell(nTrials, nDirections);
    trialDisp = cell(nTrials, nDirections);
    allRates = [];
    minBinsAcrossTrials = inf;

    for k = 1:nDirections
        for n = 1:nTrials
            rates = trialFinal(n,k).rates(keptNeurons, :);
            trialRates{n,k} = rates;
            allRates = [allRates, rates];
            minBinsAcrossTrials = min(minBinsAcrossTrials, size(rates,2));

            hp = training_data(n,k).handPos(1:2,:);
            trialDisp{n,k} = hp - hp(:,1);
        end
    end

    maxTrainBin = min(minBinsAcrossTrials, endBin);
    trainBins = startBin:maxTrainBin;
    nTrainBins = length(trainBins);

    % Z-score per neuron before PCA/regression
    muRate = mean(allRates, 2);
    sdRate = std(allRates, 0, 2) + 1e-6;
    allRatesZ = (allRates - muRate) ./ sdRate;

    % PCA basis (classifier/regression features) via SVD
    centered = allRatesZ - mean(allRatesZ, 2);
    [U, S, ~] = svd(centered, 'econ');
    svals = diag(S);
    d = svals.^2;
    cumVar = cumsum(d) / sum(d);
    nPCs = find(cumVar >= 0.95, 1);
    nPCs = max(nPCs, 10);
    W_pca = U(:,1:nPCs);

    trialPC = cell(nTrials, nDirections);
    for k = 1:nDirections
        for n = 1:nTrials
            ratesZ = (trialRates{n,k} - muRate) ./ sdRate;
            trialRatesZ{n,k} = ratesZ;
            trialPC{n,k} = W_pca' * ratesZ;
        end
    end

    % Time-dependent soft classifier bank (same idea as v3)
    classBank = struct('bin', cell(1,nTrainBins), 'W_lda', cell(1,nTrainBins), ...
        'knnRef', cell(1,nTrainBins), 'knnLabels', cell(1,nTrainBins));

    for bi = 1:nTrainBins
        b = trainBins(bi);
        X = zeros(nTrials * nDirections, nPCs);
        labels = zeros(nTrials * nDirections, 1);
        si = 0;
        for k = 1:nDirections
            for n = 1:nTrials
                si = si + 1;
                b0 = max(1, b - modelParameters.regWinBins + 1);
                feat = mean(trialPC{n,k}(:,b0:b), 2);
                X(si,:) = feat';
                labels(si) = k;
            end
        end

        classMu = zeros(nDirections, nPCs);
        for k = 1:nDirections
            classMu(k,:) = mean(X(labels == k, :), 1);
        end
        globalMu = mean(X, 1);

        Sw = zeros(nPCs);
        Sb = zeros(nPCs);
        for k = 1:nDirections
            Xk = X(labels == k, :) - classMu(k, :);
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

        classBank(bi).bin = b;
        classBank(bi).W_lda = W_lda;
        classBank(bi).knnRef = X * W_lda;
        classBank(bi).knnLabels = labels;
    end

    % PRO-style per-direction per-time PCR regression models
    % Feature vector = stacked kept-neuron rates from bin 1..b
    regDpBank = cell(nDirections, nTrainBins);
    avgDisp = cell(nDirections, 1);

    for k = 1:nDirections
        % average displacement fallback by direction
        maxLen = 0;
        for n = 1:nTrials
            maxLen = max(maxLen, size(trialDisp{n,k},2));
        end
        avg = zeros(2, maxLen);
        cnt = zeros(1, maxLen);
        for n = 1:nTrials
            dxy = trialDisp{n,k};
            L = size(dxy,2);
            avg(:,1:L) = avg(:,1:L) + dxy;
            cnt(1:L) = cnt(1:L) + 1;
        end
        for t = 1:maxLen
            if cnt(t) > 0
                avg(:,t) = avg(:,t) / cnt(t);
            elseif t > 1
                avg(:,t) = avg(:,t-1);
            end
        end
        avgDisp{k} = avg;

        for bi = 1:nTrainBins
            b = trainBins(bi);
            nKeep = sum(keptNeurons);
            L = modelParameters.regWinBins;
            featDim = nKeep * L + nKeep;
            X = zeros(nTrials, featDim);
            Ydp = zeros(nTrials, 2);

            for n = 1:nTrials
                rates = trialRatesZ{n,k};
                b0 = max(1, b - L + 1);
                rwin = rates(:, b0:b);
                if size(rwin,2) < L
                    pad = zeros(nKeep, L - size(rwin,2));
                    rwin = [pad, rwin];
                end
                rmean = mean(rates(:,1:b), 2);
                X(n,:) = [rwin(:); rmean]';

                t = b * group;
                t1 = max(group, t - group);
                dxy = trialDisp{n,k};
                yNow = dxy(:, min(t, size(dxy,2)))';
                yPrev = dxy(:, min(t1, size(dxy,2)))';
                Ydp(n,:) = (yNow - yPrev); % mm per 20 ms
            end

            xMean = mean(X, 1);
            Xc = X - xMean;
            yMeanDp = mean(Ydp, 1);
            YcDp = Ydp - yMeanDp;

            % Fast PCR basis via economy SVD in trial-space (nTrials << featDim)
            [~, S, V] = svd(Xc, 'econ');
            svals = diag(S);
            pDim = min([pcaDimReg, sum(svals > 1e-8), size(V,2)]);
            if pDim < 1, pDim = 1; end
            Vsel = V(:, 1:pDim);

            Z = Xc * Vsel; % scores (nTrials x pDim)
            BzDp = (Z' * Z + lambdaReg * eye(pDim)) \ (Z' * YcDp); % pDim x 2
            regDpBank{k,bi}.xMean = xMean;
            regDpBank{k,bi}.V = Vsel;
            regDpBank{k,bi}.Bz = BzDp;
            regDpBank{k,bi}.yMean = yMeanDp;
        end
    end

    modelParameters.group = group;
    modelParameters.smoothWin = smoothWin;
    modelParameters.startBin = startBin;
    modelParameters.endBin = endBin;

    modelParameters.lowFirers = lowFirers;
    modelParameters.keptNeurons = keptNeurons;

    modelParameters.muRate = muRate;
    modelParameters.sdRate = sdRate;
    modelParameters.W_pca = W_pca;
    modelParameters.nPCs = nPCs;
    modelParameters.nDirections = nDirections;

    modelParameters.trainBins = trainBins;
    modelParameters.classBank = classBank;
    modelParameters.regDpBank = regDpBank;
    modelParameters.avgDisp = avgDisp;

    modelParameters.prevDirProb = [];
end

function trialProcessed = bin_and_sqrt_set(trial, group, to_sqrt)
    trialProcessed = struct;
    for i = 1:size(trial,2)
        for j = 1:size(trial,1)
            allSpikes = trial(j,i).spikes;
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
            trialProcessed(j,i).spikes = spikes;
        end
    end
end

function trialFinal = get_firing_rates_set(trialProcessed, group, scale_window)
    trialFinal = struct;
    win = 10 * (scale_window / group);
    normstd = scale_window / group;
    alpha = (win - 1) / (2 * normstd);
    temp1 = -(win-1)/2 : (win-1)/2;
    gausstemp = exp((-1/2) * (alpha * temp1 / ((win-1)/2)).^2)';
    gaussian_window = gausstemp / sum(gausstemp);

    for i = 1:size(trialProcessed,2)
        for j = 1:size(trialProcessed,1)
            hold_rates = zeros(size(trialProcessed(j,i).spikes,1), size(trialProcessed(j,i).spikes,2));
            for k = 1:size(trialProcessed(j,i).spikes,1)
                hold_rates(k,:) = conv(trialProcessed(j,i).spikes(k,:), gaussian_window, 'same') / (group/1000);
            end
            trialFinal(j,i).rates = hold_rates;
        end
    end
end
