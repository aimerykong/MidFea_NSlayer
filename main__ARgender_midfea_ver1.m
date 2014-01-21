%% This code tries my mechanism which models neuron selectivity over
% mid-level features.
%
% author: Shu Kong
% date: Sep. 6, 2013
% email: aimerykong@gmail.com

clear all;
close all;
clc;

addpath(genpath('./'));
addpath(genpath('../libsvm-3.17'));
addpath(genpath('../ksvdbox13'));

%{
load('AR66x48');

rDim = 300;
U = RandProj(size(trainSet, 1), rDim )';
trainSetActv1 = mapstd(trainSet', 0,1)';
testSetActv1 = mapstd(testSet', 0,1)';


trainSetActv1 = U'*trainSetActv1;
testSetActv1 = U'*testSetActv1;
clear testSet ttrainSet
testSetActv1  =  testSetActv1./( repmat(sqrt(sum(testSetActv1.^2)), [size(testSetActv1,1),1]) );
trainSetActv1  =  trainSetActv1./( repmat(sqrt(sum(trainSetActv1.^2)), [size(trainSetActv1,1),1]) );

%}
%load('AR_300dim');
load('midFeaActv1_300_3DPoolRandProjALL');

labelIndex = unique(trainLabel);
classNum = length(labelIndex);

visibleSize = size(trainSetActv1,1);

%% parameters
noDisplay = true;

alpha= 1;
lambdaH = 0.5;
lambdaH2 = 0.001;
beta = 0.5;
gamma = 2.00;

lambdaD = 0.001;
lambdaW = 0.0001;

hiddenSize = classNum * 3;

MAXITER = 15;

% parameters for initializing W and b
iniMax = 30; % for updating Hc

Dstep = 2;
stepGradW = 0.1;
Hstep = 0.001;

%% initialize D and H
data = trainSetActv1;
labels = trainLabel;

fprintf('initialize D...\n');
D = zeros(visibleSize, hiddenSize);
D = rand(visibleSize, hiddenSize);
D = D./ repmat(sqrt(sum(D.^2, 1)), size(D,1), 1);
H = zeros(hiddenSize, size(data,2));
r = sqrt(6) / sqrt(hiddenSize+size(data,2)+1); 
H = rand(hiddenSize, size(data,2)) * r * 0.001;

%{
for c = 1:classNum
    fprintf('.%d.', c);
    a = find(labels == labelIndex(c));
    [IDX, C] = kmeans(data(:, a)', floor(hiddenSize/classNum), 'MaxIter', iniMax*5);
    C = C';
    C = C ./ repmat( sqrt(sum(C.^2,1)), size(C,1), 1);
    D(:, 1+(c-1)*floor(hiddenSize/classNum):floor(hiddenSize/classNum)*c) = C;
    
    Htmp = ones( size(C,2), length(a) );% / size(C,2);
    %{
    Htmp = (C'*C) \ (C'*data(:,a));    
    Htmp(find(Htmp<0)) = 0;
    %}
    %%{
    for i = 1:length(a)
        htmp = sum( (C - repmat(data(:,a(i)), 1, size(C,2))).^2, 1 );
        htmp = exp(-htmp/0.5);
        htmp = htmp ./ sum(htmp);
        htmp = htmp';
      %  htmp(find(htmp < 0.5*mean(htmp))) = 0;
        htmp = htmp/norm(htmp,'fro');
        Htmp(:,i) = htmp;
    end
        
    H(1+(c-1)*floor(hiddenSize/classNum):floor(hiddenSize/classNum)*c, a) = Htmp;
    clear Htmp C IDX htmp;

end
%}

%H = D'*data;
%H( find(H < mean(H(:)) ) ) = 0;
%{
H = omp(D'*data, D'*D, 10);
H = full(H);
H(find(H < 0)) = 0;
%H = abs(H);
H = H ./ repmat(sqrt(sum(H.^2,1)), size(H,1), 1);
%}
% figure('name', 'initialized dictionary');
% title('initialized dictionary');
% display_network(D);

%% initialize W and b
fprintf('\ninitialization of W and b...\n');
r = sqrt(4) / sqrt(hiddenSize+visibleSize+1);
W = rand(hiddenSize, visibleSize) * 2 * r - r;
%W = D';
b = zeros(hiddenSize, 1);
%[W, b, cost] = initializeWb(W, b, data, H, 5, 1000);
clear r;

%figure;
%plot(1:length(cost), cost, 'r-');
% figure('name', 'initialized filters');
% title('initialized filters');
% display_network(W');

%% intermediate classification results
fprintf('\nlinearSVM acc on the original mid-level features...');
model = svmtrain(trainLabel', data', '-c 1 -t 0 -m 5000 -q');
[predicted_label, accuracy, classprobilities] = svmpredict(testLabel', testSetActv1', model);
accMidFea = accuracy(1);

fprintf('\nlinearSVM acc on the initialized filters...');
trainSetH = sigmoid(W*data+repmat(b, 1, size(data,2)));
testSetH = sigmoid(W*testSetActv1+repmat(b, 1, size(testSetActv1,2) ));
model = svmtrain(trainLabel', trainSetH', '-c 1 -t 0 -m 5000 -q');
[predicted_label, accuracy, classprobilities] = svmpredict(testLabel', testSetH', model);
accList = zeros(1,MAXITER+1);
accList(1) = accuracy(1);

%{
fprintf('one iteration to update filter W and intercept b...\n');
[W, b, cost] = updateWb(W, b, data, H, 3, iniMax*400);
I = displayDictionaryElementsAsImage(W', 4, 10, 33, 24);
imshow(I);
figure('name', ['convergence check']);
plot(1:length(cost), cost, 'r-');

trainSetH = sigmoid(W*data+repmat(b, 1, size(data,2)));
testSetH = sigmoid(W*testSetActv1+repmat(b, 1, size(testSetActv1,2) ));
model = svmtrain(trainLabel', trainSetH', '-c 1 -t 0 -m 5000 -q');
[predicted_label, accuracy, classprobilities] = svmpredict(testLabel', testSetH', model);
accList(2) = accuracy(1);
%}

%% optimization - loop through each variables
fprintf('\n\nbegin optimization (maximal iteration: %d)...\n', MAXITER);
iterCount = 0;
objCostList = zeros(1,MAXITER);
while iterCount < MAXITER
    iterCount = iterCount + 1;
    fprintf('Iter-%d ...\n', iterCount);
    
    %% update W and b --- gradient descent
    fprintf('%d/%d\tupdate filter W and intercept b...\n', iterCount, MAXITER);
    [W, b, cost] = updateWb(W, b, data, H, 3, iniMax*400);
  %  figure('name', ['convergence check at iteration-' num2str(iterCount)]);
   % plot(1:length(cost), cost, 'r-');
    %     figure('name', ['updated filters at iteration-' num2str(iterCount)]);
    % I = displayDictionaryElementsAsImage(W', 5, 8, 33, 24); figure, imshow(I);
    
    [junk, predicted_label] = max(testSetH,[],1);
    a = predicted_label - testLabel;
    a = length(find(a==0))/length(a);
    fprintf('as a strong classifier with only one neuron for each class: %.4f\n', a);
    
    
    tmp = sigmoid(W*data+repmat(b,1,size(data,2)));
    trainSetH = sigmoid(W*data+repmat(b, 1, size(data,2)));
    testSetH = sigmoid(W*testSetActv1+repmat(b, 1, size(testSetActv1,2) ));
    model = svmtrain(trainLabel', trainSetH', '-c 1 -t 0 -m 5000 -q');
    [predicted_label, accuracy, classprobilities] = svmpredict(testLabel', testSetH', model);
    accList(iterCount+1) = accuracy(1);
    
    %% update D with normalization
  %  if mod(iterCount,2) ~= 0
        fprintf('%d/%d\tupdate dictionary D...\n', iterCount, MAXITER);
        
        %{
        gramH = H*H';
        dataH = data*H';
        
        for x = 1:iniMax*10
            D = D + 2*Dstep/size(data,2)*(dataH-D*gramH);
        end
         %}
        D = (data*H') / (H*H'+lambdaD*eye(size(H,1)));
        
        dd = sqrt( sum(D.^2, 1) );
       % D = D ./ repmat(dd, size(D,1), 1);
  %  end
  %  figure('name', ['updated dictionary at iteration-' num2str(iterCount)]);
  %  display_network(D);
    
    %% update Hc
    fprintf('%d/%d\tupdate selective neurons H...\n', iterCount, MAXITER);
    
    HmeanErr = zeros(1,classNum);
    HdiffErr = zeros(1,classNum);
    H12Err = zeros(1,classNum);
    for c = 1:classNum
        a = find(labels == labelIndex(c));
        aa = 1:length(labels);
        aa = setdiff(aa, a);
        fprintf('%d/%d\t\tfor class-%d (#%d)\n', iterCount, MAXITER, c, length(a));
        
        Xc = data(:, a);
        Hc = H(:,a);
        Hcdiff = H(:,aa);
        
        for x = 1:iniMax
            G = [Xc;
                sqrt(alpha)*sigmoid( W*Xc+repmat(b,1,size(Xc,2)) );
                sqrt(beta)*repmat( mean(Hc, 2), 1, size(Hc,2) );
                zeros( size(data,2)-size(Xc,2), size(Hc,2) ) ];
            
            Q = [D;
                sqrt(alpha)*eye(size(Hc,1));
                sqrt(beta)*eye(size(Hc,1));
                sqrt(gamma)*Hcdiff';];
                        
            C = (2*sqrt(sum(Hc.^2, 2)));
            C(find(C < 0.00001)) = 0.00001;
            
            C = diag(  C.^(-1) + lambdaH2*ones(size(Hc,1),1) );
            Hc = (Q'*Q + lambdaH*C) \ (Q'*G);
            Hc(find(Hc<0.00001)) = 0;
            
            %{            
            C = diag( (2*sqrt(sum(Hc.^2, 2))).^(-1) + lambdaH2*ones(size(Hc,1),1) );
            Hc = (Q'*Q + lambdaH*C) \ (Q'*G);
%             for y = 1:iniMax
%                 Hc = Hc - Hstep*(-2*Q'*G + 2*(Q'*Q)*Hc + gamma*C*Hc);
%             end
            %}

        end
        
        HmeanErr(c) = sum(sum( (Hc - repmat(mean(Hc,2), 1, size(Hc,2))).^2 ));
        H12Err(c) = sum( sqrt(sum( Hc.^2, 2)) );
        HdiffErr(c) = sum(sum((Hc'*Hcdiff).^2));
        Hc = Hc ./ repmat( sqrt(sum(Hc.^2, 1)), size(Hc,1), 1 );
        H(:, a) = Hc;
    end
    clear a aa Xc Hc;
        
    %% value of objective function
    objCostList(iterCount) = sum(sum( (data-D*H).^2 )) ...
        + alpha*sum(sum( (H-sigmoid(W*data+repmat(b,1,size(data,2)))).^2 ))...
        + lambdaH*sum(H12Err)...
        + gamma*sum(HdiffErr)...
        + beta*sum(HmeanErr)...
        + lambdaH*lambdaH2*sum(sum(H.^2))...
        + lambdaW*sum(sum(W.^2))...
        + lambdaD*sum(sum(D.^2));
    
end
fprintf('\nEND.\n\n');


%% visualize the results
trainSetH = sigmoid(W*data+repmat(b, 1, size(data,2)));
H = full(H);

if ~noDisplay
    
    figure('name', 'Firing Neurons');
    subplot(2,1,1);
    imshow(H);
    title('H');
    
    subplot(2,1,2);    
    imshow(trainSetH);
    %display_network(TMPgen(:,1:end), 2);
    title('fast predicted H');
    
    figure('name', 'trainSet raw image');
    I = displayDictionaryElementsAsImage(trainSetActv1, 20, 35, 33, 24);
    imshow(I);    
    title('raw image');
    
    figure('name', 'rec by SC');
    I = displayDictionaryElementsAsImage(D*H, 20, 35, 33, 24);
    imshow(I);
    title('rec by SC');
    
    figure('name', 'rec by fast prediction');
    I = displayDictionaryElementsAsImage(D*trainSetH, 20, 35, 33, 24);
    imshow(I);
    title('rec by fast prediction');
    
    clear tmp
    figure('name', 'learned filters');
    I = displayDictionaryElementsAsImage(W', 5, 8, 33, 24);
    imshow(I);
    title('learned filters');
    
    figure('name', 'learned dictionaries');
    I = displayDictionaryElementsAsImage(D, 5, 8, 33, 24);
    imshow(I);
    title('learned dictionaries');
    
    figure('name', 'check convergence');
    plot(1:length(objCostList), objCostList, '-');
    xlabel('#steps');
    ylabel('#obj val');
    title('convergence');
end

%% fire the neurons given the test set (optional visualization)
testSetH = sigmoid(W*testSetActv1+repmat(b, 1, size(testSetActv1,2) ));

if ~noDisplay
    figure('name', 'prediction');    
    for c = 1:classNum
        subplot(classNum, 1, c);        
        a = find(testLabel==c);
        imshow(testSetH(:,a));
        title(['predicted features on test set of class-' num2str(c)]);
    end
    
    % %{
    I = displayDictionaryElementsAsImage(testSetActv1, 21, 33, 33, 24);
    figure('name', 'test set - raw image');    
    imshow(I)
    title('test set - raw image');
    
    I = displayDictionaryElementsAsImage(D*testSetH, 21, 33, 33, 24);
    figure('name', 'rec TestSet');
    imshow(I);
    title('rec TestSet');
    %%}
end

%% classification results
model = svmtrain(trainLabel', trainSetH', '-c 1 -t 0 -m 5000 -q');
[predicted_label, accuracy, classprobilities] = svmpredict(testLabel', testSetH', model);

errRate = zeros(classNum,3);
for c = 1:classNum
    a = find(testLabel==c);
    errRate(c, 1) = c;
    errRate(c, 2) = length(a);
    errRate(c, 3) = length(find(predicted_label(a)~=c)) / length(a);
end
disp(errRate);

a = testLabel-predicted_label';
a = length(find(a~=0))/length(testLabel);
fprintf('overall acc:%4f, error rate: %4f\n\n', 1-a, a);


figure('name', 'gender classification accuracy on AR');
plot(1:length(accList), accList, 'r*-')
hold on;
plot(1:length(accList), accMidFea*ones(1,length(accList)), 'b.-');
legend('linear SVM', 'Ours with linear regression', 'Location', 'SouthEast');
xlabel('#iteration');
ylabel('accuracy (%)');
title('gender classification accuracy on AR');

%% store the whole results
T = datestr(now,'yyyymmddHHMMSS');
%save(['../result/results_' datasetName '_' T '.mat']);
