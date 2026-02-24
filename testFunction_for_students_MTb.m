% Test Script to give to the students, March 2015
%% Continuous Position Estimator Test Script
% This function first calls the function "positionEstimatorTraining" to get
% the relevant modelParameters, and then calls the function
% "positionEstimator" to decode the trajectory. 
close all; clear all
% testFunction_for_students_MTb('FunnyBMI')

% teamName = 'FunnyBMI';

% function RMSE = testFunction_for_students_MTb(teamName)

load monkeydata_training.mat

% Set random number generator
rng(2013);
ix = randperm(length(trial));

% addpath(teamName);

% Select training and testing data (you can choose to split your data in a different way if you wish)
trainingData = trial(ix(1:50),:);
testData = trial(ix(51:end),:);

fprintf('Testing the continuous position estimator...')

meanSqError = 0;
n_predictions = 0;  
allActual = [];
allDecoded = [];

figure
hold on
axis square
grid

% Train Model
modelParameters = positionEstimatorTraining(trainingData)

for tr=1:size(testData,1)
    display(['Decoding block ',num2str(tr),' out of ',num2str(size(testData,1))]);
    pause(0.001)
    for direc=randperm(8) 
        decodedHandPos = [];

        times=320:20:size(testData(tr,direc).spikes,2);
        
        for t=times
            past_current_trial.trialId = testData(tr,direc).trialId;
            past_current_trial.spikes = testData(tr,direc).spikes(:,1:t); 
            past_current_trial.decodedHandPos = decodedHandPos;

            past_current_trial.startHandPos = testData(tr,direc).handPos(1:2,1); 
            if nargout('positionEstimator') == 3
                [decodedPosX, decodedPosY, newParameters] = positionEstimator(past_current_trial, modelParameters);
                modelParameters = newParameters;
            elseif nargout('positionEstimator') == 2
                [decodedPosX, decodedPosY] = positionEstimator(past_current_trial, modelParameters{direc});
            end
%             display(decodedPosX)
%             display(decodedPosY)
            decodedPos = [decodedPosX; decodedPosY];
            decodedHandPos = [decodedHandPos decodedPos];
            
            actualPos = testData(tr,direc).handPos(1:2,t);
            allActual = [allActual actualPos];
            allDecoded = [allDecoded decodedPos];
            
            meanSqError = meanSqError + norm(actualPos - decodedPos)^2;
            
        end
        n_predictions = n_predictions+length(times);
        hold on
        plot(decodedHandPos(1,:),decodedHandPos(2,:), 'r');
        plot(testData(tr,direc).handPos(1,times),testData(tr,direc).handPos(2,times),'b')
    end
end

legend('Decoded Position', 'Actual Position')

RMSE = sqrt(meanSqError/n_predictions)

% Normalized RMSE (by range of actual positions)
posRange = max(max(allActual,[],2) - min(allActual,[],2));
NRMSE = RMSE / posRange

% R-squared (coefficient of determination)
SS_res = sum(sum((allActual - allDecoded).^2));
SS_tot = sum(sum((allActual - mean(allActual,2)).^2));
R2 = 1 - SS_res / SS_tot

fprintf('\n--- Results ---\n')
fprintf('RMSE:  %.4f\n', RMSE)
fprintf('NRMSE: %.4f (%.1f%% of workspace range)\n', NRMSE, NRMSE*100)
fprintf('R²:    %.4f\n', R2)

% rmpath(genpath(teamName))

% end
