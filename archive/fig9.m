% runDiag10.m - Test PCA regression with only 4-window features
% (always available since test starts at T=320ms)
% Also test zero-padded progressive features

load('monkeydata_training.mat');
rng(2013);
ix = randperm(length(trial));

trainingData = trial(ix(1:50),:);
testData = trial(ix(51:end),:);
modelParameters = positionEstimatorTraining3(trainingData);
nDirs = modelParameters.nDirections;
wLen = modelParameters.wLen;
maxNWin = modelParameters.nWin;
nN = modelParameters.nNeurons;

allPos = [];
for tr = 1:size(testData,1)
    for direc = 1:8
        times = 320:20:size(testData(tr,direc).spikes,2);
        allPos = [allPos, testData(tr,direc).handPos(1:2,times)];
    end
end
posRange = mean([max(allPos(1,:))-min(allPos(1,:)), max(allPos(2,:))-min(allPos(2,:))]);

nTrain = size(trainingData, 1);
nFeat4 = nN * 4;  % 4 windows
nFeat7 = nN * maxNWin;  % 7 windows

% ============================================================
% Store per-trial features (4-window and 7-window)
% ============================================================
trainFeats4 = cell(1, nDirs);
trainFeats7 = cell(1, nDirs);
trainDisp = cell(1, nDirs);

for k = 1:nDirs
    f4 = zeros(nTrain, nFeat4);
    f7 = zeros(nTrain, nFeat7);
    trajCell = cell(1, nTrain);
    for j = 1:nTrain
        sp = trainingData(j,k).spikes;
        hp = trainingData(j,k).handPos(1:2,:);
        trajCell{j} = hp - hp(:,1);
        for w = 1:4
            ts = (w-1)*wLen + 1; te = w * wLen;
            f4(j, (w-1)*nN+1:w*nN) = sqrt(sum(sp(:, ts:te), 2))';
        end
        for w = 1:maxNWin
            ts = (w-1)*wLen + 1; te = w * wLen;
            f7(j, (w-1)*nN+1:w*nN) = sqrt(sum(sp(:, ts:te), 2))';
        end
    end
    trainFeats4{k} = f4;
    trainFeats7{k} = f7;
    trainDisp{k} = trajCell;
end

% ============================================================
% Train PCA regression with 4-window features
% ============================================================
fprintf('Training PCA regression (4-window)...\n');
nPC = 10; lam = 1;
evalTimes = 320:20:800;

pca4Models = trainPCAModel(trainFeats4, trainDisp, nPC, lam, evalTimes, nTrain);
pca7Models = trainPCAModel(trainFeats7, trainDisp, nPC, lam, evalTimes, nTrain);

% ============================================================
% Test 1: PCA-4 vs PCA-7 oracle
% ============================================================
fprintf('\n=== ORACLE: PCA-4 vs PCA-7 ===\n');
[rmse4_oracle] = testPCA_oracle(testData, pca4Models, trainFeats4, nN, 4, modelParameters, posRange, evalTimes);
fprintf('PCA-4 oracle: RMSE=%.4f NRMSE=%.4f (%.2f%%)\n', rmse4_oracle, rmse4_oracle/posRange, 100*rmse4_oracle/posRange);
[rmse7_oracle] = testPCA_oracle(testData, pca7Models, trainFeats7, nN, maxNWin, modelParameters, posRange, evalTimes);
fprintf('PCA-7 oracle: RMSE=%.4f NRMSE=%.4f (%.2f%%)\n', rmse7_oracle, rmse7_oracle/posRange, 100*rmse7_oracle/posRange);

% ============================================================
% Test 2: PCA-4 + blend + soft classification
% ============================================================
fprintf('\n=== PCA-4 + BLEND + SOFT CLASSIFICATION ===\n');
for temp = [1.0, 2.0, 3.0, 5.0]
    for pcaW = [0.3, 0.5, 0.7]
        rmse = testPCA_soft(testData, pca4Models, trainFeats4, nN, 4, modelParameters, posRange, evalTimes, temp, pcaW);
        fprintf('PCA-4 temp=%.1f pca=%.1f: RMSE=%.4f NRMSE=%.4f (%.2f%%)\n', temp, pcaW, rmse, rmse/posRange, 100*rmse/posRange);
    end
end

% ============================================================
% Test 3: Progressive features (zero-pad missing windows)
% ============================================================
fprintf('\n=== PROGRESSIVE PCA-7 (zero-pad) + SOFT CLASSIFICATION ===\n');
for temp = [1.0, 2.0, 3.0, 5.0]
    for pcaW = [0.3, 0.5, 0.7]
        rmse = testPCA_progressive(testData, pca7Models, trainFeats7, nN, maxNWin, modelParameters, posRange, evalTimes, temp, pcaW);
        fprintf('Prog-7 temp=%.1f pca=%.1f: RMSE=%.4f NRMSE=%.4f (%.2f%%)\n', temp, pcaW, rmse, rmse/posRange, 100*rmse/posRange);
    end
end

% ============================================================
% Test 4: PCA-4 + finer grid
% ============================================================
fprintf('\n=== PCA-4 FINE GRID ===\n');
bestRMSE = inf; bestT = 0; bestP = 0;
for temp = [1.5, 2.0, 2.5, 3.0, 4.0]
    for pcaW = [0.3, 0.4, 0.5, 0.6, 0.7]
        rmse = testPCA_soft(testData, pca4Models, trainFeats4, nN, 4, modelParameters, posRange, evalTimes, temp, pcaW);
        if rmse < bestRMSE
            bestRMSE = rmse; bestT = temp; bestP = pcaW;
        end
    end
end
fprintf('Best PCA-4: temp=%.1f pca=%.1f RMSE=%.4f NRMSE=%.4f (%.2f%%)\n', bestT, bestP, bestRMSE, bestRMSE/posRange, 100*bestRMSE/posRange);

% ============================================================
% Test 5: PCA-4 with different nPC and lambda
% ============================================================
fprintf('\n=== PCA-4 PARAMETER SWEEP (temp=3.0, pca=0.5) ===\n');
for nPC2 = [5, 10, 15, 20, 30]
    for lam2 = [0.1, 1, 10, 50]
        pcaM = trainPCAModel(trainFeats4, trainDisp, nPC2, lam2, evalTimes, nTrain);
        rmse = testPCA_soft(testData, pcaM, trainFeats4, nN, 4, modelParameters, posRange, evalTimes, 3.0, 0.5);
        fprintf('nPC=%2d lam=%4.1f: RMSE=%.4f NRMSE=%.4f (%.2f%%)\n', nPC2, lam2, rmse, rmse/posRange, 100*rmse/posRange);
    end
end

% ============================================================
function models = trainPCAModel(trainFeats, trainDisp, nPC, lam, evalTimes, nTrain)
    nDirs = length(trainFeats);
    models = cell(1, nDirs);
    for k = 1:nDirs
        X = trainFeats{k};
        Xmean = mean(X, 1);
        Xc = X - Xmean;
        K_mat = Xc * Xc';
        [V, D] = eig(K_mat);
        [eigvals, idx] = sort(diag(D), 'descend');
        V = V(:, idx);
        nPC_use = min(nPC, sum(eigvals > 1e-6));
        scores = V(:, 1:nPC_use) .* sqrt(max(eigvals(1:nPC_use), 0))';

        betas = cell(1, length(evalTimes));
        ymeans = zeros(2, length(evalTimes));
        for ti = 1:length(evalTimes)
            t = evalTimes(ti);
            Y = zeros(nTrain, 2);
            for j = 1:nTrain
                traj = trainDisp{k}{j};
                if t <= size(traj,2); Y(j,:) = traj(:,t)'; else; Y(j,:) = traj(:,end)'; end
            end
            ym = mean(Y, 1);
            Yc = Y - ym;
            S2 = scores' * scores;
            beta = (S2 + lam * eye(nPC_use)) \ (scores' * Yc);
            betas{ti} = beta;
            ymeans(:,ti) = ym';
        end
        models{k}.Xmean = Xmean;
        models{k}.V = V(:, 1:nPC_use);
        models{k}.eigvals = eigvals(1:nPC_use);
        models{k}.Xc = Xc;
        models{k}.nPC = nPC_use;
        models{k}.betas = betas;
        models{k}.ymeans = ymeans;
        models{k}.evalTimes = evalTimes;
    end
end

function rmse = testPCA_oracle(testData, pcaModels, trainFeats, nN, nWin, mp, posRange, evalTimes)
    wLen = mp.wLen;
    mse = 0; np = 0;
    for tr = 1:size(testData,1)
        for direc = 1:8
            sp = testData(tr,direc).spikes;
            hp = testData(tr,direc).handPos(1:2,:);
            startPos = hp(:,1);
            Tmax = size(sp,2);
            times = 320:20:Tmax;

            testFeat = zeros(1, nN * nWin);
            for w = 1:nWin
                ts = (w-1)*wLen + 1; te = w * wLen;
                testFeat((w-1)*nN+1:w*nN) = sqrt(sum(sp(:, ts:te), 2))';
            end

            rm = pcaModels{direc};
            xc = testFeat - rm.Xmean;
            k_vec = rm.Xc * xc';
            test_score = (rm.V' * k_vec)' ./ sqrt(max(rm.eigvals', 1e-6));

            for ti = 1:length(times)
                t = times(ti);
                [~, eti] = min(abs(evalTimes - t));
                pred_disp = (test_score * rm.betas{eti})' + rm.ymeans(:,eti);
                pos = startPos + pred_disp;
                err = norm(hp(:,t) - pos)^2;
                mse = mse + err; np = np + 1;
            end
        end
    end
    rmse = sqrt(mse/np);
end

function rmse = testPCA_soft(testData, pcaModels, trainFeats, nN, nWin, mp, posRange, evalTimes, temperature, pcaWeight)
    wLen = mp.wLen; nDirs = mp.nDirections; maxNWin = mp.nWin;
    mse = 0; np = 0;
    for tr = 1:size(testData,1)
        for direc = 1:8
            sp = testData(tr,direc).spikes;
            hp = testData(tr,direc).handPos(1:2,:);
            startPos = hp(:,1);
            Tmax = size(sp,2);
            times = 320:20:Tmax;

            testFeat = zeros(1, nN * nWin);
            nAvail = min(nWin, floor(Tmax / wLen));
            for w = 1:nAvail
                ts = (w-1)*wLen + 1; te = w * wLen;
                testFeat((w-1)*nN+1:w*nN) = sqrt(sum(sp(:, ts:te), 2))';
            end

            % Pre-compute PCA projections for all directions
            pcaPreds = cell(1, nDirs);
            for k = 1:nDirs
                rm = pcaModels{k};
                xc = testFeat - rm.Xmean;
                k_vec = rm.Xc * xc';
                pcaPreds{k}.score = (rm.V' * k_vec)' ./ sqrt(max(rm.eigvals', 1e-6));
            end

            for ti = 1:length(times)
                t = times(ti);

                % LDA classification
                nAvailWin = min(floor(t / wLen), maxNWin);
                logP = zeros(1, nDirs);
                for w = 1:nAvailWin
                    ts = (w-1)*wLen + 1; te = w * wLen;
                    f = sqrt(sum(sp(:, ts:te), 2))';
                    for kk = 1:nDirs
                        d = f - mp.winLDA(w).dirMeans(kk,:);
                        logP(kk) = logP(kk) - 0.5 * (d * mp.winLDA(w).SwInv * d');
                    end
                end
                logP = logP / temperature;
                maxLP = max(logP);
                P = exp(logP - maxLP);
                P = P / sum(P);

                % Soft PCA + soft absolute trajectory
                softPCA = zeros(2,1);
                softAbs = zeros(2,1);
                for k = 1:nDirs
                    % PCA prediction
                    rm = pcaModels{k};
                    [~, eti] = min(abs(evalTimes - t));
                    pred_disp = (pcaPreds{k}.score * rm.betas{eti})' + rm.ymeans(:,eti);
                    softPCA = softPCA + P(k) * pred_disp;

                    % Absolute trajectory
                    avgT = mp.avgTraj{k};
                    aLen = mp.avgTrajLen(k);
                    if t <= aLen; softAbs = softAbs + P(k) * avgT(:,t);
                    else; softAbs = softAbs + P(k) * avgT(:,end); end
                end

                pcaPos = startPos + softPCA;
                absPos = softAbs;
                pos = pcaWeight * pcaPos + (1-pcaWeight) * absPos;

                err = norm(hp(:,t) - pos)^2;
                mse = mse + err; np = np + 1;
            end
        end
    end
    rmse = sqrt(mse/np);
end

function rmse = testPCA_progressive(testData, pcaModels, trainFeats, nN, maxNWin, mp, posRange, evalTimes, temperature, pcaWeight)
    % Progressive: zero-pad missing windows at each timepoint
    wLen = mp.wLen; nDirs = mp.nDirections;
    nFeat = nN * maxNWin;
    mse = 0; np = 0;
    for tr = 1:size(testData,1)
        for direc = 1:8
            sp = testData(tr,direc).spikes;
            hp = testData(tr,direc).handPos(1:2,:);
            startPos = hp(:,1);
            Tmax = size(sp,2);
            times = 320:20:Tmax;

            for ti = 1:length(times)
                t = times(ti);

                % Progressive feature: only available windows
                testFeat = zeros(1, nFeat);
                nAvailWin = min(floor(t / wLen), maxNWin);
                for w = 1:nAvailWin
                    ts = (w-1)*wLen + 1; te = w * wLen;
                    testFeat((w-1)*nN+1:w*nN) = sqrt(sum(sp(:, ts:te), 2))';
                end

                % LDA classification
                logP = zeros(1, nDirs);
                for w = 1:nAvailWin
                    ts = (w-1)*wLen + 1; te = w * wLen;
                    f = sqrt(sum(sp(:, ts:te), 2))';
                    for kk = 1:nDirs
                        d = f - mp.winLDA(w).dirMeans(kk,:);
                        logP(kk) = logP(kk) - 0.5 * (d * mp.winLDA(w).SwInv * d');
                    end
                end
                logP = logP / temperature;
                maxLP = max(logP);
                P = exp(logP - maxLP);
                P = P / sum(P);

                softPCA = zeros(2,1);
                softAbs = zeros(2,1);
                for k = 1:nDirs
                    rm = pcaModels{k};
                    xc = testFeat - rm.Xmean;
                    k_vec = rm.Xc * xc';
                    test_score = (rm.V' * k_vec)' ./ sqrt(max(rm.eigvals', 1e-6));
                    [~, eti] = min(abs(evalTimes - t));
                    pred_disp = (test_score * rm.betas{eti})' + rm.ymeans(:,eti);
                    softPCA = softPCA + P(k) * pred_disp;

                    avgT = mp.avgTraj{k};
                    aLen = mp.avgTrajLen(k);
                    if t <= aLen; softAbs = softAbs + P(k) * avgT(:,t);
                    else; softAbs = softAbs + P(k) * avgT(:,end); end
                end

                pcaPos = startPos + softPCA;
                absPos = softAbs;
                pos = pcaWeight * pcaPos + (1-pcaWeight) * absPos;

                err = norm(hp(:,t) - pos)^2;
                mse = mse + err; np = np + 1;
            end
        end
    end
    rmse = sqrt(mse/np);
end
