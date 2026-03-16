%%% Model 9: Population Vector Algorithm (PVA) Decoder
%%%   Preferred direction estimation via cosine tuning + average trajectory fallback
%%% BMI Spring 2025
%%% Team: Edward Goh

% function modelParameters = positionEstimatorTraining9(training_data)
%
% PVA pipeline:
%   1. Estimate preferred directions via cosine tuning (least squares fit)
%   2. Store preferred direction unit vectors, baseline rates, modulation depths
%   3. Average trajectories per direction (fallback)
%   4. Calibrate speed scaling factor from training data
%
% Arguments:
%   training_data(n,k)              n = trial index, k = reaching direction
%   training_data(n,k).spikes(i,t)  i = neuron, t = time (ms)
%   training_data(n,k).handPos(d,t) d = dimension [1-3], t = time (ms)
%
% Return Value:
%   modelParameters  struct with all learned parameters for positionEstimator9

function modelParameters = positionEstimatorTraining9(training_data)

    [nTrials, nDirs] = size(training_data);
    nNeurons = size(training_data(1,1).spikes, 1);

    modelParameters.nNeurons = nNeurons;
    modelParameters.nDirs    = nDirs;

    % Reaching angles in radians
    angles = [30, 70, 110, 150, 190, 230, 310, 350] * pi / 180;
    modelParameters.angles = angles;

    % ====================================================================
    % STEP 1: Compute average firing rates per neuron per direction
    % Use the entire trial duration for rate estimation during training
    % ====================================================================
    rates = zeros(nNeurons, nDirs);   % mean firing rate (spikes/sec)

    for k = 1:nDirs
        for n = 1:nTrials
            sp = training_data(n,k).spikes;
            T  = size(sp, 2);
            % Mean firing rate across trial (spikes/sec)
            rates(:,k) = rates(:,k) + sum(sp, 2) / (T / 1000);
        end
        rates(:,k) = rates(:,k) / nTrials;
    end

    % ====================================================================
    % STEP 2: Fit cosine tuning to estimate preferred directions
    %
    % Model: r_i(theta) = b_i + m_cos_i * cos(theta) + m_sin_i * sin(theta)
    %        => preferred direction: theta_i* = atan2(m_sin_i, m_cos_i)
    %        => modulation depth:    m_i = sqrt(m_cos_i^2 + m_sin_i^2)
    %
    % Design matrix A (nDirs x 3): [1, cos(angles'), sin(angles')]
    % Solve: A * [b; m_cos; m_sin]' = rates(i,:)'  for each neuron i
    % ====================================================================
    A = [ones(nDirs,1), cos(angles'), sin(angles')];   % nDirs x 3

    prefDir   = zeros(2, nNeurons);   % unit vectors [cos(theta*); sin(theta*)]
    baseline  = zeros(nNeurons, 1);
    modDepth  = zeros(nNeurons, 1);
    tuningCoeffs = zeros(nNeurons, 3);  % [b, m_cos, m_sin]

    for i = 1:nNeurons
        r = rates(i,:)';   % nDirs x 1
        % Least squares: coeffs = A \ r
        coeffs = A \ r;
        b_i     = coeffs(1);
        m_cos_i = coeffs(2);
        m_sin_i = coeffs(3);

        theta_star = atan2(m_sin_i, m_cos_i);
        m_i        = sqrt(m_cos_i^2 + m_sin_i^2);

        prefDir(:,i)     = [cos(theta_star); sin(theta_star)];
        baseline(i)      = b_i;
        modDepth(i)      = m_i;
        tuningCoeffs(i,:) = coeffs';
    end

    modelParameters.prefDir      = prefDir;       % 2 x nNeurons
    modelParameters.baseline     = baseline;      % nNeurons x 1
    modelParameters.modDepth     = modDepth;      % nNeurons x 1
    modelParameters.tuningCoeffs = tuningCoeffs;  % nNeurons x 3
    modelParameters.rates        = rates;         % nNeurons x nDirs (for diagnostics)

    % ====================================================================
    % STEP 3: Average trajectories per direction (fallback)
    % ====================================================================
    for k = 1:nDirs
        maxLen = 0;
        for n = 1:nTrials
            maxLen = max(maxLen, size(training_data(n,k).handPos, 2));
        end

        sumT   = zeros(2, maxLen);
        cntT   = zeros(1, maxLen);

        for n = 1:nTrials
            hp = training_data(n,k).handPos(1:2,:);
            L  = size(hp, 2);
            sumT(:, 1:L) = sumT(:, 1:L) + hp(:, 1:L);
            cntT(1:L)    = cntT(1:L) + 1;
        end

        avgT = zeros(2, maxLen);
        for t = 1:maxLen
            if cntT(t) > 0
                avgT(:,t) = sumT(:,t) / cntT(t);
            elseif t > 1
                avgT(:,t) = avgT(:,t-1);
            end
        end

        % Compute average displacement from start position per direction
        sumDisp = zeros(2, maxLen);
        for n = 1:nTrials
            hp     = training_data(n,k).handPos(1:2,:);
            L      = size(hp, 2);
            startP = hp(:,1);
            sumDisp(:, 1:L) = sumDisp(:, 1:L) + (hp(:, 1:L) - startP);
        end
        avgDisp = zeros(2, maxLen);
        for t = 1:maxLen
            if cntT(t) > 0
                avgDisp(:,t) = sumDisp(:,t) / cntT(t);
            elseif t > 1
                avgDisp(:,t) = avgDisp(:,t-1);
            end
        end

        modelParameters.avgTraj{k}    = avgT;
        modelParameters.avgDisp{k}    = avgDisp;
        modelParameters.avgTrajLen(k) = maxLen;
    end

    % ====================================================================
    % STEP 4: Calibrate PVA speed scaling factor
    %
    % At each eval time t (320:20:end), compute:
    %   PVA vector using 150ms window firing rates
    %   actual velocity from handPos (mm/ms)
    % Speed scale = mean(actual_speed / pva_magnitude) where pva_magnitude > 0
    % ====================================================================
    windowLen  = 200;   % ms for rate estimation
    evalTimes  = 320:20:800;

    pva_mags   = [];
    actual_speeds = [];

    for k = 1:nDirs
        for n = 1:nTrials
            sp = training_data(n,k).spikes;
            hp = training_data(n,k).handPos(1:2,:);
            Tmax = size(sp, 2);

            for ti = 1:length(evalTimes)
                t = evalTimes(ti);
                if t > Tmax || t < 21
                    continue;
                end

                % Causal firing rate in [t-windowLen+1, t]
                t_start = max(1, t - windowLen + 1);
                wL      = t - t_start + 1;
                r_win   = sum(sp(:, t_start:t), 2) / (wL / 1000);  % spikes/sec

                % PVA computation
                r_above = max(r_win - baseline, 0);
                popVec  = prefDir * (r_above .* modDepth);  % 2x1
                pva_mag = norm(popVec);

                % Actual velocity at time t (mm/ms)
                if t > 20
                    vel  = (hp(:,t) - hp(:,t-20)) / 20;
                    spd  = norm(vel);
                else
                    spd = 0;
                end

                if pva_mag > 1e-6
                    pva_mags      = [pva_mags;      pva_mag]; %#ok<AGROW>
                    actual_speeds = [actual_speeds; spd];     %#ok<AGROW>
                end
            end
        end
    end

    if ~isempty(pva_mags) && sum(pva_mags > 1e-6) > 10
        % Robust least-squares scale: scale = mean(actual_speed ./ pva_mag)
        speedScale = median(actual_speeds ./ max(pva_mags, 1e-6));
    else
        speedScale = 0.01;  % fallback default (mm/ms per unit PVA)
    end

    modelParameters.speedScale = speedScale;
    modelParameters.windowLen  = windowLen;

    % ====================================================================
    % STEP 5: LDA classification (for direction classification fallback)
    % Simple pooled LDA using 320ms of spike counts
    % ====================================================================
    wLen = 80;
    maxNWin = 4;
    alpha_reg = 0.3;

    nFeat = nNeurons * maxNWin;
    dirMeansLDA = zeros(nDirs, nFeat);
    allFeats    = zeros(nTrials * nDirs, nFeat);

    for k = 1:nDirs
        df = zeros(nTrials, nFeat);
        for n = 1:nTrials
            sp = training_data(n,k).spikes;
            for w = 1:maxNWin
                ts = (w-1)*wLen + 1;
                te =  w   *wLen;
                if te <= size(sp,2)
                    df(n, (w-1)*nNeurons+1 : w*nNeurons) = sqrt(sum(sp(:, ts:te), 2))';
                end
            end
        end
        dirMeansLDA(k,:) = mean(df, 1);
        allFeats((k-1)*nTrials+1 : k*nTrials, :) = df;
    end

    Sw = zeros(nFeat);
    for k = 1:nDirs
        C  = allFeats((k-1)*nTrials+1 : k*nTrials, :) - dirMeansLDA(k,:);
        Sw = Sw + C' * C;
    end
    Sw = Sw / (nTrials*nDirs - nDirs);
    Sw = (1 - alpha_reg) * Sw + alpha_reg * (trace(Sw) / nFeat) * eye(nFeat);

    modelParameters.ldaDirMeans = dirMeansLDA;
    modelParameters.ldaSwInv    = inv(Sw);
    modelParameters.wLen        = wLen;
    modelParameters.maxNWin     = maxNWin;

end
