function modelParameters = positionEstimatorTraining2(training_data)
    % Fig8-inspired alternative model:
    % 1) classify direction from PCA+LDA+kNN
    % 2) predict displacement with soft direction-weighted kNN trajectories
    % 3) blend with soft direction-weighted average trajectory

    binSize = 20;
    gaussSigma = 25;
    lowFireThreshold = 0.02;
    classifyBins = 320 / binSize;
    nDirections = size(training_data, 2);
    nTrials = size(training_data, 1);
    nNeuronsRaw = size(training_data(1,1).spikes, 1);

    % Trajectory hyperparameters
    modelParameters.knnTrajK = 10;
    modelParameters.temperature = 2.0;
    modelParameters.hybridAlpha = 0.3;
    modelParameters.alphaMin = 0.35;
    modelParameters.alphaMax = 0.85;
    modelParameters.rampSteps = 15;

    % Gaussian smoothing kernel
    kernelHW = ceil(3 * gaussSigma);
    kernelX = -kernelHW:kernelHW;
    kernel = exp(-kernelX.^2 / (2 * gaussSigma^2));
    kernel = kernel / sum(kernel);

    % Build per-trial rates
    allRatesRaw = [];
    trialRatesRaw = cell(nTrials, nDirections);
    trialHP = cell(nTrials, nDirections);
    for k = 1:nDirections
        for n = 1:nTrials
            spikes = training_data(n, k).spikes;
            T = size(spikes, 2);
            samplePts = binSize:binSize:T;
            nBins = length(samplePts);

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

    % LDA reference set for direction classification
    nSamples = nTrials * nDirections;
    X_lda = zeros(nSamples, nPCs);
    labels = zeros(nSamples, 1);
    si = 0;
    for k = 1:nDirections
        for n = 1:nTrials
            si = si + 1;
            rates = trialRates{n, k};
            nUseBins = min(classifyBins, size(rates, 2));
            meanR = mean(rates(:, 1:nUseBins), 2);
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

    % Direction-specific trajectory bank
    trajFeats = cell(nDirections, 1);
    trajDisp = cell(nDirections, 1);
    avgDisp = cell(nDirections, 1);
    for k = 1:nDirections
        featMat = zeros(nTrials, nPCs);
        dispCell = cell(nTrials, 1);
        maxLen = 0;
        for n = 1:nTrials
            rates = trialRates{n, k};
            pc = W_pca' * (rates - mu);
            nUseBins = min(classifyBins, size(pc, 2));
            featMat(n, :) = mean(pc(:, 1:nUseBins), 2)';

            hp = trialHP{n, k};
            dispCell{n} = hp - hp(:,1);
            maxLen = max(maxLen, size(hp, 2));
        end

        avg = zeros(2, maxLen);
        cnt = zeros(1, maxLen);
        for n = 1:nTrials
            dxy = dispCell{n};
            L = size(dxy, 2);
            avg(:, 1:L) = avg(:, 1:L) + dxy;
            cnt(1:L) = cnt(1:L) + 1;
        end
        for t = 1:maxLen
            if cnt(t) > 0
                avg(:, t) = avg(:, t) / cnt(t);
            elseif t > 1
                avg(:, t) = avg(:, t-1);
            end
        end

        trajFeats{k} = featMat;
        trajDisp{k} = dispCell;
        avgDisp{k} = avg;
    end

    % Store model parameters
    modelParameters.binSize = binSize;
    modelParameters.classifyBins = classifyBins;
    modelParameters.nDirections = nDirections;
    modelParameters.kernel = kernel;
    modelParameters.keptNeurons = keptNeurons;

    modelParameters.nPCs = nPCs;
    modelParameters.mu = mu;
    modelParameters.W_pca = W_pca;

    modelParameters.W_lda = W_lda;
    modelParameters.knnRef = knnRef;
    modelParameters.knnLabels = knnLabels;
    modelParameters.kNN_k = 5;

    modelParameters.trajFeats = trajFeats;
    modelParameters.trajDisp = trajDisp;
    modelParameters.avgDisp = avgDisp;
end
