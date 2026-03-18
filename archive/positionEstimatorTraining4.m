function modelParameters = positionEstimatorTraining4(training_data)
    % Hybrid v4:
    % - shared preprocessing + PCA
    % - time-dependent LDA+kNN direction classifier
    % - per-direction, per-time ridge regression to displacement (PCR-style idea)
    % - soft direction blending + avg-trajectory fallback in inference

    binSize = 20;
    gaussSigma = 25;
    lowFireThreshold = 0.02;
    classifyBins = 320 / binSize;
    lambda = 1.0;

    nDirections = size(training_data, 2);
    nTrials = size(training_data, 1);
    nNeuronsRaw = size(training_data(1,1).spikes, 1);

    % Inference hyperparameters
    modelParameters.kNN_k = 5;
    modelParameters.temperatureEarly = 2.5;
    modelParameters.temperatureLate = 1.4;
    modelParameters.temperatureRampBins = 15;
    modelParameters.dirProbSmoothing = 0.60;
    modelParameters.regBlendBase = 0.45;
    modelParameters.regBlendConfGain = 0.45;
    modelParameters.alphaMin = 0.35;
    modelParameters.alphaMax = 0.85;
    modelParameters.rampSteps = 15;

    % Gaussian kernel
    kernelHW = ceil(3 * gaussSigma);
    kernelX = -kernelHW:kernelHW;
    kernel = exp(-kernelX.^2 / (2 * gaussSigma^2));
    kernel = kernel / sum(kernel);

    % Build per-trial rates
    allRatesRaw = [];
    trialRatesRaw = cell(nTrials, nDirections);
    trialHP = cell(nTrials, nDirections);
    minBinsAcrossTrials = inf;

    for k = 1:nDirections
        for n = 1:nTrials
            spikes = training_data(n, k).spikes;
            T = size(spikes, 2);
            samplePts = binSize:binSize:T;
            nBins = length(samplePts);
            minBinsAcrossTrials = min(minBinsAcrossTrials, nBins);

            rates = zeros(nNeuronsRaw, nBins);
            for i = 1:nNeuronsRaw
                smoothed = conv(spikes(i,:), kernel, 'same');
                rates(i,:) = sqrt(smoothed(samplePts));
            end

            trialRatesRaw{n, k} = rates;
            trialHP{n, k} = training_data(n, k).handPos(1:2, :);
            allRatesRaw = [allRatesRaw, rates];
        end
    end

    % Low-firing neuron removal
    meanRatePerNeuron = mean(allRatesRaw, 2);
    keptNeurons = meanRatePerNeuron >= lowFireThreshold;
    allRates = allRatesRaw(keptNeurons, :);
    trialRates = cell(nTrials, nDirections);
    for k = 1:nDirections
        for n = 1:nTrials
            trialRates{n, k} = trialRatesRaw{n, k}(keptNeurons, :);
        end
    end

    % PCA
    mu = mean(allRates, 2);
    centered = allRates - mu;
    C = (centered * centered') / (size(centered, 2) - 1);
    [V, D] = eig(C);
    d = diag(D);
    [d, idx] = sort(d, 'descend');
    V = V(:, idx);
    cumVar = cumsum(d) / sum(d);
    nPCs = find(cumVar >= 0.95, 1);
    nPCs = max(nPCs, 10);
    W_pca = V(:, 1:nPCs);

    % Precompute per-trial PCA trajectories and displacement trajectories
    trialPC = cell(nTrials, nDirections);
    trajDisp = cell(nDirections, 1);
    avgDisp = cell(nDirections, 1);

    for k = 1:nDirections
        trajDisp{k} = cell(nTrials, 1);
        maxLen = 0;
        for n = 1:nTrials
            rates = trialRates{n, k};
            pc = W_pca' * (rates - mu);
            trialPC{n, k} = pc;

            hp = trialHP{n, k};
            trajDisp{k}{n} = hp - hp(:,1);
            maxLen = max(maxLen, size(hp, 2));
        end

        avg = zeros(2, maxLen);
        cnt = zeros(1, maxLen);
        for n = 1:nTrials
            dxy = trajDisp{k}{n};
            L = size(dxy, 2);
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
    end

    % Bins where we train both classifier and regressor
    timeBins = classifyBins:minBinsAcrossTrials;
    nTimeBins = length(timeBins);

    % Time-dependent classifier bank
    classBank = struct('bin', cell(1, nTimeBins), 'W_lda', cell(1, nTimeBins), ...
                       'knnRef', cell(1, nTimeBins), 'knnLabels', cell(1, nTimeBins));

    for bi = 1:nTimeBins
        b = timeBins(bi);
        X = zeros(nTrials * nDirections, nPCs);
        labels = zeros(nTrials * nDirections, 1);
        si = 0;

        for k = 1:nDirections
            for n = 1:nTrials
                si = si + 1;
                X(si, :) = mean(trialPC{n, k}(:,1:b), 2)';
                labels(si) = k;
            end
        end

        classMu = zeros(nDirections, nPCs);
        for k = 1:nDirections
            classMu(k, :) = mean(X(labels == k, :), 1);
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

    % Per-direction, per-time displacement regression bank
    % Feature = [meanPC(1:b); bias]
    regBank = cell(nDirections, nTimeBins);
    for k = 1:nDirections
        for bi = 1:nTimeBins
            b = timeBins(bi);
            Xreg = zeros(nTrials, nPCs + 1);
            Yreg = zeros(nTrials, 2);

            for n = 1:nTrials
                pc = trialPC{n, k};
                f = mean(pc(:,1:b), 2)';
                Xreg(n,:) = [f, 1];

                t = b * binSize;
                dxy = trajDisp{k}{n};
                if t <= size(dxy, 2)
                    Yreg(n,:) = dxy(:,t)';
                else
                    Yreg(n,:) = dxy(:,end)';
                end
            end

            XtX = Xreg' * Xreg;
            B = (XtX + lambda * eye(nPCs + 1)) \ (Xreg' * Yreg);
            regBank{k, bi} = B; % (nPCs+1) x 2
        end
    end

    % Store
    modelParameters.binSize = binSize;
    modelParameters.classifyBins = classifyBins;
    modelParameters.nDirections = nDirections;
    modelParameters.kernel = kernel;
    modelParameters.keptNeurons = keptNeurons;

    modelParameters.nPCs = nPCs;
    modelParameters.mu = mu;
    modelParameters.W_pca = W_pca;

    modelParameters.timeBins = timeBins;
    modelParameters.classBank = classBank;
    modelParameters.regBank = regBank;
    modelParameters.avgDisp = avgDisp;

    modelParameters.prevDirProb = [];
end
