function modelParameters = positionEstimatorTraining(training_data)
    binSize = 20;
    nDirections = size(training_data, 2);
    nTrials = size(training_data, 1);
    nNeurons = size(training_data(1,1).spikes, 1);
    classifyBins = 320 / binSize;
    nLags = 2;
    lambda = 0.01;

    %% Feature extraction: 20ms binning + sqrt transform
    allRates = [];
    trialRates = cell(nTrials, nDirections);
    trialHP = cell(nTrials, nDirections);

    for k = 1:nDirections
        for n = 1:nTrials
            spikes = training_data(n, k).spikes;
            T = size(spikes, 2);
            nBins = floor(T / binSize);

            rates = zeros(nNeurons, nBins);
            for b = 1:nBins
                rates(:, b) = sqrt(sum(spikes(:, (b-1)*binSize+1 : b*binSize), 2));
            end

            trialRates{n, k} = rates;
            trialHP{n, k} = training_data(n, k).handPos;
            allRates = [allRates, rates];
        end
    end

    %% PCA
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

    %% LDA + kNN reference set
    nSamples = nTrials * nDirections;
    X_lda = zeros(nSamples, nPCs);
    labels = zeros(nSamples, 1);
    si = 0;

    for k = 1:nDirections
        for n = 1:nTrials
            si = si + 1;
            rates = trialRates{n, k};
            meanR = mean(rates(:, 1:classifyBins), 2);
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

    %% Per-direction ridge regression (position + velocity targets)
    featureDim = nPCs * (nLags + 1) + 1;
    betaPos = cell(nDirections, 1);
    betaVel = cell(nDirections, 1);

    for k = 1:nDirections
        Xreg = [];
        Ypos = [];
        Yvel = [];

        for n = 1:nTrials
            rates = trialRates{n, k};
            hp = trialHP{n, k};
            nBins = size(rates, 2);

            pc = W_pca' * (rates - mu);

            startBin = max(classifyBins, nLags + 1);

            for b = startBin:nBins
                t = b * binSize;
                if t > size(hp, 2), break; end

                feat = zeros(featureDim, 1);
                for lag = 0:nLags
                    feat(lag*nPCs+1 : (lag+1)*nPCs) = pc(:, b - lag);
                end
                feat(end) = 1;

                posT = hp(1:2, t);

                tPrev = (b - 1) * binSize;
                if tPrev < 1, tPrev = 1; end
                velT = hp(1:2, t) - hp(1:2, tPrev);

                Xreg = [Xreg; feat'];
                Ypos = [Ypos; posT'];
                Yvel = [Yvel; velT'];
            end
        end

        XtX = Xreg' * Xreg;
        R = XtX + lambda * eye(featureDim);
        betaPos{k} = R \ (Xreg' * Ypos);
        betaVel{k} = R \ (Xreg' * Yvel);
    end

    %% Store all parameters
    modelParameters.binSize = binSize;
    modelParameters.nPCs = nPCs;
    modelParameters.nLags = nLags;
    modelParameters.mu = mu;
    modelParameters.W_pca = W_pca;
    modelParameters.W_lda = W_lda;
    modelParameters.knnRef = knnRef;
    modelParameters.knnLabels = knnLabels;
    modelParameters.betaPos = betaPos;
    modelParameters.betaVel = betaVel;
    modelParameters.classifyBins = classifyBins;
    modelParameters.nDirections = nDirections;
    modelParameters.alpha = 0.5;
    modelParameters.ensembleW = 0.5;
    modelParameters.kNN_k = 5;
    modelParameters.cachedWeights = [];
end
