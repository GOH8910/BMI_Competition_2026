% fig8.m - Diagnostics aligned with current positionEstimator pipeline
% Uses the same train/test split and online decoding loop as testFunction_for_students_MTb,
% then adds trajectory-based oracle analyses using compatible model fields.

load('monkeydata_training.mat');
rng(2013);
ix = randperm(length(trial));

trainingData = trial(ix(1:50), :);
testData = trial(ix(51:end), :);

fprintf('Training model with positionEstimatorTraining...\n');
modelParameters = positionEstimatorTraining(trainingData);

% ============================================================
% PART 1: Baseline online decoder (exact calling pattern)
% ============================================================
fprintf('\n=== BASELINE: positionEstimator online decode ===\n');
[rmseBase, nrmseBase, r2Base, allActual, allDecoded] = evaluateOnline(testData, modelParameters);
fprintf('Baseline RMSE:  %.4f\n', rmseBase);
fprintf('Baseline NRMSE: %.4f (%.2f%% of workspace range)\n', nrmseBase, 100 * nrmseBase);
fprintf('Baseline R^2:   %.4f\n', r2Base);

% Position range for trajectory diagnostics
posRange = max(max(allActual, [], 2) - min(allActual, [], 2));

% ============================================================
% PART 2: Build direction-specific trajectory references
% ============================================================
fprintf('\nBuilding trajectory references from training set...\n');
[trainFeats, trainDisp, avgDisp] = buildTrajectoryBank(trainingData, modelParameters);

% ============================================================
% PART 3: Precompute test features + soft direction probabilities
% ============================================================
fprintf('Precomputing test features and soft direction probabilities...\n');
results = buildTestDiagnostics(testData, modelParameters);

% ============================================================
% PART 4: Trajectory diagnostics compatible with current model
% ============================================================
fprintf('\n=== TRAJECTORY DIAGNOSTICS (compatible with current pipeline) ===\n');

rmse_oracle_avg = evalOracleAvg(results, avgDisp);
fprintf('Oracle avg trajectory:      RMSE=%.4f NRMSE=%.4f (%.2f%%)\n', ...
    rmse_oracle_avg, rmse_oracle_avg / posRange, 100 * rmse_oracle_avg / posRange);

for K = [1, 3, 5, 10, 25, 50]
    rmse_knn = evalKNNOracle(results, trainFeats, trainDisp, K);
    fprintf('Oracle k-NN (k=%2d):        RMSE=%.4f NRMSE=%.4f (%.2f%%)\n', ...
        K, rmse_knn, rmse_knn / posRange, 100 * rmse_knn / posRange);
end

for K = [3, 5, 10, 25, 50]
    for temp = [1.0, 2.0, 3.0]
        rmse_knn_soft = evalKNNSoft(results, trainFeats, trainDisp, K, temp);
        fprintf('Soft k-NN (k=%2d, temp=%.1f): RMSE=%.4f NRMSE=%.4f (%.2f%%)\n', ...
            K, temp, rmse_knn_soft, rmse_knn_soft / posRange, 100 * rmse_knn_soft / posRange);
    end
end

for K = [3, 5, 10]
    for alpha = [0.1, 0.2, 0.3, 0.5]
        rmse_hyb = evalHybridOracle(results, avgDisp, trainFeats, trainDisp, K, alpha);
        fprintf('Hybrid (k=%2d, alpha=%.1f):  RMSE=%.4f NRMSE=%.4f (%.2f%%)\n', ...
            K, alpha, rmse_hyb, rmse_hyb / posRange, 100 * rmse_hyb / posRange);
    end
end

% ============================================================
% Helpers
% ============================================================

function [rmse, nrmse, r2, allActual, allDecoded] = evaluateOnline(testData, modelParameters)
    meanSqError = 0;
    nPredictions = 0;
    allActual = [];
    allDecoded = [];

    for tr = 1:size(testData,1)
        for direc = randperm(size(testData,2))
            decodedHandPos = [];
            times = 320:20:size(testData(tr,direc).spikes,2);

            for t = times
                past_current_trial.trialId = testData(tr,direc).trialId;
                past_current_trial.spikes = testData(tr,direc).spikes(:,1:t);
                past_current_trial.decodedHandPos = decodedHandPos;
                past_current_trial.startHandPos = testData(tr,direc).handPos(1:2,1);

                [decodedPosX, decodedPosY, modelParameters] = positionEstimator(past_current_trial, modelParameters);
                decodedPos = [decodedPosX; decodedPosY];
                decodedHandPos = [decodedHandPos, decodedPos];

                actualPos = testData(tr,direc).handPos(1:2,t);
                allActual = [allActual, actualPos];
                allDecoded = [allDecoded, decodedPos];
                meanSqError = meanSqError + norm(actualPos - decodedPos)^2;
            end

            nPredictions = nPredictions + length(times);
        end
    end

    rmse = sqrt(meanSqError / nPredictions);
    posRange = max(max(allActual,[],2) - min(allActual,[],2));
    nrmse = rmse / posRange;
    ssRes = sum(sum((allActual - allDecoded).^2));
    ssTot = sum(sum((allActual - mean(allActual,2)).^2));
    r2 = 1 - ssRes / ssTot;
end

function [trainFeats, trainDisp, avgDisp] = buildTrajectoryBank(trainingData, mp)
    nDirs = size(trainingData,2);
    nTrials = size(trainingData,1);

    trainFeats = cell(1, nDirs);
    trainDisp = cell(1, nDirs);
    avgDisp = cell(1, nDirs);

    for k = 1:nDirs
        featMat = zeros(nTrials, mp.nPCs);
        dispCell = cell(1, nTrials);
        maxLen = 0;

        for j = 1:nTrials
            spikes = trainingData(j,k).spikes;
            hp = trainingData(j,k).handPos(1:2,:);

            rates = computeRates(spikes, mp.binSize, mp.kernel);
            rates = rates(mp.keptNeurons, :);
            pc = mp.W_pca' * (rates - mp.mu);

            featMat(j,:) = mean(pc, 2)';
            dispCell{j} = hp - hp(:,1);
            maxLen = max(maxLen, size(hp,2));
        end

        avg = zeros(2, maxLen);
        cnt = zeros(1, maxLen);
        for j = 1:nTrials
            d = dispCell{j};
            L = size(d,2);
            avg(:,1:L) = avg(:,1:L) + d;
            cnt(1:L) = cnt(1:L) + 1;
        end
        for t = 1:maxLen
            if cnt(t) > 0
                avg(:,t) = avg(:,t) / cnt(t);
            elseif t > 1
                avg(:,t) = avg(:,t-1);
            end
        end

        trainFeats{k} = featMat;
        trainDisp{k} = dispCell;
        avgDisp{k} = avg;
    end
end

function results = buildTestDiagnostics(testData, mp)
    nTr = size(testData,1);
    nDirs = size(testData,2);
    results = struct();

    for tr = 1:nTr
        for direc = 1:nDirs
            spikes = testData(tr,direc).spikes;
            hp = testData(tr,direc).handPos(1:2,:);

            rates = computeRates(spikes, mp.binSize, mp.kernel);
            rates = rates(mp.keptNeurons, :);
            pc = mp.W_pca' * (rates - mp.mu);
            feat = mean(pc, 2)';

            ldaFeat = (mp.W_lda' * mean(pc, 2))';
            probs = computeDirProb(ldaFeat, mp.knnRef, mp.knnLabels, mp.kNN_k, mp.nDirections);

            results(tr,direc).times = 320:20:size(spikes,2);
            results(tr,direc).hp = hp;
            results(tr,direc).startPos = hp(:,1);
            results(tr,direc).trueDir = direc;
            results(tr,direc).feat = feat;
            results(tr,direc).dirProb = probs;
        end
    end
end

function rates = computeRates(spikes, binSize, kernel)
    T = size(spikes,2);
    samplePts = binSize:binSize:T;
    nBins = length(samplePts);
    nNeurons = size(spikes,1);
    rates = zeros(nNeurons, nBins);

    for i = 1:nNeurons
        smoothed = conv(spikes(i,:), kernel, 'same');
        rates(i,:) = sqrt(smoothed(samplePts));
    end
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

function rmse = evalOracleAvg(results, avgDisp)
    mse = 0;
    np = 0;
    for tr = 1:size(results,1)
        for direc = 1:size(results,2)
            r = results(tr,direc);
            for ti = 1:length(r.times)
                t = r.times(ti);
                dAvg = avgDisp{direc};
                if t <= size(dAvg,2)
                    d = dAvg(:,t);
                else
                    d = dAvg(:,end);
                end
                pos = r.startPos + d;
                mse = mse + norm(r.hp(:,t) - pos)^2;
                np = np + 1;
            end
        end
    end
    rmse = sqrt(mse / np);
end

function rmse = evalKNNOracle(results, trainFeats, trainDisp, K)
    mse = 0;
    np = 0;
    nTrain = size(trainFeats{1}, 1);

    for tr = 1:size(results,1)
        for direc = 1:size(results,2)
            r = results(tr,direc);
            dists = sqrt(sum((trainFeats{direc} - r.feat).^2, 2));
            [~, idx] = sort(dists, 'ascend');
            kUse = min(K, nTrain);
            idx = idx(1:kUse);
            w = 1 ./ (dists(idx) + 1e-8);
            w = w / sum(w);

            for ti = 1:length(r.times)
                t = r.times(ti);
                dKnn = zeros(2,1);
                for j = 1:kUse
                    traj = trainDisp{direc}{idx(j)};
                    if t <= size(traj,2)
                        d = traj(:,t);
                    else
                        d = traj(:,end);
                    end
                    dKnn = dKnn + w(j) * d;
                end
                pos = r.startPos + dKnn;
                mse = mse + norm(r.hp(:,t) - pos)^2;
                np = np + 1;
            end
        end
    end
    rmse = sqrt(mse / np);
end

function rmse = evalKNNSoft(results, trainFeats, trainDisp, K, temperature)
    mse = 0;
    np = 0;
    nDirs = length(trainFeats);

    for tr = 1:size(results,1)
        for direc = 1:size(results,2)
            r = results(tr,direc);

            knnIdx = cell(1, nDirs);
            knnW = cell(1, nDirs);
            for k = 1:nDirs
                dists = sqrt(sum((trainFeats{k} - r.feat).^2, 2));
                [~, idx] = sort(dists, 'ascend');
                kUse = min(K, length(idx));
                idx = idx(1:kUse);
                w = 1 ./ (dists(idx) + 1e-8);
                w = w / sum(w);
                knnIdx{k} = idx;
                knnW{k} = w;
            end

            P = softmaxWithTemp(r.dirProb, temperature);

            for ti = 1:length(r.times)
                t = r.times(ti);
                dSoft = zeros(2,1);
                for k = 1:nDirs
                    dKnn = zeros(2,1);
                    idx = knnIdx{k};
                    w = knnW{k};
                    for j = 1:length(idx)
                        traj = trainDisp{k}{idx(j)};
                        if t <= size(traj,2)
                            d = traj(:,t);
                        else
                            d = traj(:,end);
                        end
                        dKnn = dKnn + w(j) * d;
                    end
                    dSoft = dSoft + P(k) * dKnn;
                end
                pos = r.startPos + dSoft;
                mse = mse + norm(r.hp(:,t) - pos)^2;
                np = np + 1;
            end
        end
    end

    rmse = sqrt(mse / np);
end

function rmse = evalHybridOracle(results, avgDisp, trainFeats, trainDisp, K, alpha)
    mse = 0;
    np = 0;
    nTrain = size(trainFeats{1}, 1);

    for tr = 1:size(results,1)
        for direc = 1:size(results,2)
            r = results(tr,direc);
            dists = sqrt(sum((trainFeats{direc} - r.feat).^2, 2));
            [~, idx] = sort(dists, 'ascend');
            kUse = min(K, nTrain);
            idx = idx(1:kUse);
            w = 1 ./ (dists(idx) + 1e-8);
            w = w / sum(w);

            for ti = 1:length(r.times)
                t = r.times(ti);

                dAvg = avgDisp{direc};
                if t <= size(dAvg,2)
                    da = dAvg(:,t);
                else
                    da = dAvg(:,end);
                end

                dKnn = zeros(2,1);
                for j = 1:kUse
                    traj = trainDisp{direc}{idx(j)};
                    if t <= size(traj,2)
                        d = traj(:,t);
                    else
                        d = traj(:,end);
                    end
                    dKnn = dKnn + w(j) * d;
                end

                dFinal = (1 - alpha) * da + alpha * dKnn;
                pos = r.startPos + dFinal;
                mse = mse + norm(r.hp(:,t) - pos)^2;
                np = np + 1;
            end
        end
    end

    rmse = sqrt(mse / np);
end

function P = softmaxWithTemp(baseProb, temperature)
    logP = log(baseProb + 1e-12) / temperature;
    logP = logP - max(logP);
    P = exp(logP);
    P = P / sum(P);
end
