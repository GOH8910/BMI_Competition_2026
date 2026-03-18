function modelParameters = positionEstimatorTraining6(training_data)
    % Improved version of positionEstimator2:
    % 1) time-dependent direction classifier (PCA + per-time LDA+kNN)
    % 2) progressive trajectory matching features by current available time
    % 3) soft-direction weighted kNN trajectory + average trajectory blend

    binSize = 20;
    gaussSigma = 25;
    lowFireThreshold = 0.02;
    classifyBins = 320 / binSize; % first decode bin

    nDirections = size(training_data, 2);
    nTrials = size(training_data, 1);
    nNeuronsRaw = size(training_data(1,1).spikes, 1);

    % Hyperparameters
    modelParameters.kNN_k = 5;            % classifier kNN
    modelParameters.knnTrajK = 10;        % trajectory kNN
    modelParameters.temperatureEarly = 2.5;
    modelParameters.temperatureLate = 1.5;
    modelParameters.temperatureRampBins = 15;
    modelParameters.hybridAlphaBase = 0.25;
    modelParameters.hybridAlphaConfGain = 0.55;
    modelParameters.dirProbSmoothing = 0.55;
    modelParameters.maxStepMm = 6.0;
    modelParameters.stepEma = 0.7;
    modelParameters.stepWeightBase = 0.25;
    modelParameters.stepWeightConfGain = 0.45;
    modelParameters.stepRampBins = 8;
    modelParameters.alphaMin = 0.35;
    modelParameters.alphaMax = 0.85;
    modelParameters.rampSteps = 15;

    % Gaussian kernel
    kernelHW = ceil(3 * gaussSigma);
    kernelX = -kernelHW:kernelHW;
    kernel = exp(-kernelX.^2 / (2 * gaussSigma^2));
    kernel = kernel / sum(kernel);

    % Build trial rates
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

    % Remove low-firing neurons
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

    % Precompute per-trial PC projections and trajectories
    trialPC = cell(nTrials, nDirections);
    trajFeatsByBin = cell(nDirections, minBinsAcrossTrials);
    trajDisp = cell(nDirections, 1);
    trajStep = cell(nDirections, 1);
    avgDisp = cell(nDirections, 1);
    avgStep = cell(nDirections, 1);

    for k = 1:nDirections
        trajDisp{k} = cell(nTrials, 1);
        trajStep{k} = cell(nTrials, 1);
        maxLen = 0;
        for n = 1:nTrials
            rates = trialRates{n, k};
            pc = W_pca' * (rates - mu);
            trialPC{n, k} = pc;

            hp = trialHP{n, k};
            dxy = hp - hp(:,1);
            step = zeros(2, size(dxy, 2));
            for t = 1:size(dxy, 2)
                tPrev = max(1, t - binSize);
                step(:, t) = dxy(:, t) - dxy(:, tPrev);
            end
            trajDisp{k}{n} = dxy;
            trajStep{k}{n} = step;
            maxLen = max(maxLen, size(hp, 2));
        end

        % Average displacement and 20ms-step trajectories for direction k
        avg = zeros(2, maxLen);
        avgS = zeros(2, maxLen);
        cnt = zeros(1, maxLen);
        for n = 1:nTrials
            dxy = trajDisp{k}{n};
            step = trajStep{k}{n};
            L = size(dxy, 2);
            avg(:,1:L) = avg(:,1:L) + dxy;
            avgS(:,1:L) = avgS(:,1:L) + step;
            cnt(1:L) = cnt(1:L) + 1;
        end
        for t = 1:maxLen
            if cnt(t) > 0
                avg(:,t) = avg(:,t) / cnt(t);
                avgS(:,t) = avgS(:,t) / cnt(t);
            elseif t > 1
                avg(:,t) = avg(:,t-1);
                avgS(:,t) = avgS(:,t-1);
            end
        end
        avgDisp{k} = avg;
        avgStep{k} = avgS;

        % Progressive trajectory features at each bin b: mean PC(:,1:b)
        for b = classifyBins:minBinsAcrossTrials
            F = zeros(nTrials, nPCs);
            for n = 1:nTrials
                pc = trialPC{n, k};
                F(n,:) = mean(pc(:,1:b), 2)';
            end
            trajFeatsByBin{k, b} = F;
        end
    end

    % Time-dependent classifier bank (per bin): LDA + kNN references
    classBins = classifyBins:minBinsAcrossTrials;
    nClassBins = length(classBins);
    classBank = struct('bin', cell(1, nClassBins), 'W_lda', cell(1, nClassBins), ...
                       'knnRef', cell(1, nClassBins), 'knnLabels', cell(1, nClassBins));

    for bi = 1:nClassBins
        b = classBins(bi);
        X = zeros(nTrials * nDirections, nPCs);
        labels = zeros(nTrials * nDirections, 1);
        si = 0;

        for k = 1:nDirections
            for n = 1:nTrials
                si = si + 1;
                pc = trialPC{n, k};
                X(si, :) = mean(pc(:,1:b), 2)';
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

    % Store
    modelParameters.binSize = binSize;
    modelParameters.classifyBins = classifyBins;
    modelParameters.nDirections = nDirections;
    modelParameters.kernel = kernel;
    modelParameters.keptNeurons = keptNeurons;
    modelParameters.nPCs = nPCs;
    modelParameters.mu = mu;
    modelParameters.W_pca = W_pca;

    modelParameters.classBins = classBins;
    modelParameters.classBank = classBank;
    modelParameters.trajFeatsByBin = trajFeatsByBin;
    modelParameters.maxFeatBins = minBinsAcrossTrials;
    modelParameters.trajDisp = trajDisp;
    modelParameters.trajStep = trajStep;
    modelParameters.avgDisp = avgDisp;
    modelParameters.avgStep = avgStep;

    modelParameters.prevDirProb = [];
    modelParameters.prevStep = [];
end
