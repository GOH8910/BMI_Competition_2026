function modelParameters = positionEstimatorTraining(training_data)
    binSize = 20;
    nDirections = size(training_data, 2);
    nTrials = size(training_data, 1);
    nNeuronsRaw = size(training_data(1,1).spikes, 1);
    classifyBins = 320 / binSize;
    nLags = 2;
    lambda = 0.01;
    gaussSigma = 25;
    phaseSplit = 0;
    lowFireThreshold = 0.02;

    %% Build Gaussian smoothing kernel
    kernelHW = ceil(3 * gaussSigma);
    kernelX = -kernelHW:kernelHW;
    kernel = exp(-kernelX.^2 / (2 * gaussSigma^2));
    kernel = kernel / sum(kernel);

    %% Feature extraction: Gaussian smooth + sqrt + resample at 20ms
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
            trialHP{n, k} = training_data(n, k).handPos;
            allRatesRaw = [allRatesRaw, rates];
        end
    end

    %% Remove low-firing neurons
    meanRatePerNeuron = mean(allRatesRaw, 2);
    keptNeurons = meanRatePerNeuron >= lowFireThreshold;
    allRates = allRatesRaw(keptNeurons, :);
    trialRates = cell(nTrials, nDirections);
    for k = 1:nDirections
        for n = 1:nTrials
            trialRates{n, k} = trialRatesRaw{n, k}(keptNeurons, :);
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

    %% LDA + kNN reference set (mean over all bins)
    nSamples = nTrials * nDirections;
    X_lda = zeros(nSamples, nPCs);
    labels = zeros(nSamples, 1);
    si = 0;

    for k = 1:nDirections
        for n = 1:nTrials
            si = si + 1;
            rates = trialRates{n, k};
            meanR = mean(rates, 2);
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

    %% Per-direction ridge regression
    % Feature: [pc_lag0; pc_lag1; pc_lag2; dpc_current; bias]
    featureDim = nPCs * (nLags + 1) + nPCs + 1;

    betaPosEarly = cell(nDirections, 1);
    betaVelEarly = cell(nDirections, 1);
    betaPosLate  = cell(nDirections, 1);
    betaVelLate  = cell(nDirections, 1);

    phaseBoundary = classifyBins + phaseSplit;

    for k = 1:nDirections
        Xreg_e = []; Ypos_e = []; Yvel_e = [];
        Xreg_l = []; Ypos_l = []; Yvel_l = [];

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
                dpcOffset = nPCs * (nLags + 1);
                if b >= 2
                    feat(dpcOffset+1 : dpcOffset+nPCs) = pc(:, b) - pc(:, b-1);
                end
                feat(end) = 1;

                posT = hp(1:2, t);
                tPrev = (b - 1) * binSize;
                if tPrev < 1, tPrev = 1; end
                velT = hp(1:2, t) - hp(1:2, tPrev);

                if phaseSplit == 0 || b <= phaseBoundary
                    Xreg_e = [Xreg_e; feat'];
                    Ypos_e = [Ypos_e; posT'];
                    Yvel_e = [Yvel_e; velT'];
                else
                    Xreg_l = [Xreg_l; feat'];
                    Ypos_l = [Ypos_l; posT'];
                    Yvel_l = [Yvel_l; velT'];
                end
            end
        end

        XtX = Xreg_e' * Xreg_e;
        R = XtX + lambda * eye(featureDim);
        betaPosEarly{k} = R \ (Xreg_e' * Ypos_e);
        betaVelEarly{k} = R \ (Xreg_e' * Yvel_e);

        if size(Xreg_l, 1) > 0
            XtX = Xreg_l' * Xreg_l;
            R = XtX + lambda * eye(featureDim);
            betaPosLate{k} = R \ (Xreg_l' * Ypos_l);
            betaVelLate{k} = R \ (Xreg_l' * Yvel_l);
        else
            betaPosLate{k} = betaPosEarly{k};
            betaVelLate{k} = betaVelEarly{k};
        end
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
    modelParameters.keptNeurons = keptNeurons;
    modelParameters.betaPosEarly = betaPosEarly;
    modelParameters.betaVelEarly = betaVelEarly;
    modelParameters.betaPosLate  = betaPosLate;
    modelParameters.betaVelLate  = betaVelLate;
    modelParameters.classifyBins = classifyBins;
    modelParameters.nDirections = nDirections;
    modelParameters.phaseSplit = phaseSplit;
    modelParameters.kernel = kernel;
    modelParameters.ensembleW = 0.5;
    modelParameters.kNN_k = 5;
    modelParameters.alphaMin = 0.4;
    modelParameters.alphaMax = 0.85;
    modelParameters.rampSteps = 15;
end
