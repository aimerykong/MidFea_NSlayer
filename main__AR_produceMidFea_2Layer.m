%% This code tries my mechanism which models neuron selectivity over
% mid-level features.
%
% author: Shu Kong
% date: Sep. 6, 2013
% email: aimerykong@gmail.com
%
clear all;
close all;
clc;

addpath(genpath('./'));
addpath(genpath('../libsvm-3.17'));
addpath(genpath('../ksvdbox13'));

%% parameters
reLearnDict1 = true;
rDim = 500;

imgSize = [66, 48];

parSet.K1 = 9;
parSet.filterSize1 = 7;
parSet.visibleSize1 = parSet.filterSize1.^2;

parSet.downRatio2 = 2;
parSet.K2 = 36;

cmbNum = 8;


parSet.iniMax = 50;
%{
selectionMat2 = zeros(parSet.K2, cmbNum);
for i = 1:parSet.K2
    a = randperm(parSet.K1);
    selectionMat2(i,:) = a(1:cmbNum);
end
%}
selectionMat2 = nchoosek(1:parSet.K1, 2);
%selectionMat2 = selectionMat2(a(1:parSet.K2), :);


%% load database
load('AR66x48.mat');
trainSet = reshape(trainSet(:), [imgSize size(trainSet,2)]);
trainSet = trainSet ./ 255;
testSet = reshape(testSet(:), [imgSize size(testSet,2)]);
testSet = testSet ./ 255;

%% learn the 1st-layer filters in a divide-and-conquer manner
if reLearnDict1
    numSubDict = parSet.K1;
    numDivide = 1;
    Dict1 = zeros(parSet.visibleSize1, numSubDict*numDivide);
    
    fprintf('learning filters by k-mean in a divide-and-conque manner...\n');
    for i = 1:numDivide
        fprintf('\t%d-round...\n', i);
        a = randperm(size(trainSet,3));
        tmpDAT = trainSet(:,:,a(1:8));
        patches = ExtractPatchesFromPool(tmpDAT, parSet.filterSize1);
        clear tmpDAT;
        a = find((sum(patches.^2,1))~=0);
        patches = patches(:,a);
        
        [ind, tmpDict] = kmeans(patches', numSubDict, 'MaxIter', parSet.iniMax);
        tmpDict = tmpDict';
        tmpDict = tmpDict ./ repmat( sqrt(sum(tmpDict.^2, 1)), size(tmpDict,1), 1);
        
        Dict1(:,(i-1)*numSubDict+1:i*numSubDict) = tmpDict;
    end
    
    I = displayDictionaryElementsAsImage(Dict1, numDivide, numSubDict, parSet.filterSize1, parSet.filterSize1);
    figure('name', 'learned filters');
    imshow(I);
    title('learned filters');
    
    %% select the most informative filters
    fprintf('select the most informative atoms...\n');
    score = sum(abs(Dict1'*Dict1), 1);
    [junk idx] = sort(abs(score), 'ascend');
    Dict1 = Dict1(:, idx);
    save('Dict1', 'Dict1');
    clear i numSubDict numDivide tmpDict
else
    load('Dict1');
end

Dict1 = Dict1(:, 1:parSet.K1);
Dict1 = Dict1-repmat( min(Dict1, [], 1), size(Dict1,1), 1);
Dict1 = Dict1 ./ repmat( sqrt(sum(Dict1.^2,1)), size(Dict1,1), 1);
%%{
I = displayDictionaryElementsAsImage(Dict1, floor(sqrt(parSet.K1)), floor(sqrt(parSet.K1)), parSet.filterSize1, parSet.filterSize1);
figure('name', 'selective filters');
imshow(I);
title('selective filters');
%}

%% soft convolution with 3D-max-pooling on trainSet
trainSetActv1 = zeros( ...
    floor((imgSize(1)-parSet.filterSize1+1)/parSet.downRatio2), ... %
    floor((imgSize(2)-parSet.filterSize1+1)/parSet.downRatio2),... %
    size(selectionMat2,1), size(trainSet,3)  );

for m = 1:size(trainSet,3)
    im = trainSet(:,:,m);
    if mod(m, 20) == 0
        fprintf('%d.',m);
    end
    Actv1tmp = zeros( size(trainSet,1)+1-parSet.filterSize1, size(trainSet,2)+1-parSet.filterSize1, size(Dict1, 2));
    %{
    for ind = 1:size(Dict1, 2)
        kernel = reshape( Dict1(:, ind), parSet.filterSize1, parSet.filterSize1);
    %    kernel = kernel ./ max(kernel(:));
        tmp = conv2(im, flipud(fliplr(kernel)), 'valid');
        tmp(find(abs(tmp) < mean(abs(tmp(:))))) = 0;
      %  tmp  = poolFeaMap(tmp, parSet.downRatio2);
        Actv1tmp(:,:,ind) = tmp;
    end
    %}
    for i = 1:size(Dict1,2)
        kernel = reshape( Dict1(:,i), sqrt(size(Dict1,1)), sqrt(size(Dict1,1)));
        tmp = conv2(double(im), flipud(fliplr(kernel)), 'valid');
        %  tmp = tmp - mean(tmp(:));
        %  tmp = tmp./max(abs(tmp(:)));
        %  tau = 0.3*(max(abs(tmp(:))) + mean(abs(tmp(:))));
        %  tmp(find(abs(tmp) < tau)) = 0;
        Actv1tmp(:,:,i) = abs(tmp);
    end
    tau = 1.00;
    
    mat2 = sort(Actv1tmp,3);
    mat2 = mat2(:,:,6); %floor(size(mat,3)/2)
    mat2 = Actv1tmp - repmat(tau*mat2, [1, 1, size(Actv1tmp,3)]);
    mat2(find(mat2<0)) = 0;
    
    
    maskmat = sqrt( sum(mat2.^2, 3) );
    maskmat(find(maskmat==0)) = 1;
    maskmat = repmat(maskmat, [1 1 size(mat2,3)]);
    mat2 = mat2./ maskmat;
    
    Actv1tmp = mat2;
    
    %mat = mat - repmat(mean(mean(mat,1),2), [size(mat,1) size(mat,2) 1]);
    %Actv1tmp = mat2 ./ repmat(sqrt(sum(sum(mat2.^2,1),2)), [size(Actv1tmp,1) size(Actv1tmp,2) 1]);
    
    %{
    figure('name', ['image-' num2str(m) ' activations']);
    tmp = reshape(Actv1tmp(:), numel(Actv1tmp)/size(Actv1tmp,3), size(Actv1tmp,3));
    I = displayDictionaryElementsAsImage(tmp, sqrt(parSet.K1), sqrt(parSet.K1), size(Actv1tmp,1), size(Actv1tmp,2));
    imshow(I);
    title(['image-' num2str(m) ' activations']);
    %}
    
    Actv1tmp = pooling3D(Actv1tmp, selectionMat2, parSet.downRatio2);
    
    %{
    figure('name', ['image-' num2str(m) ' activations']);
    tmp = reshape(Actv1tmp(:), numel(Actv1tmp)/size(Actv1tmp,3), size(Actv1tmp,3));
    I = displayDictionaryElementsAsImage(tmp, sqrt(parSet.K2), sqrt(parSet.K2), size(Actv1tmp,1), size(Actv1tmp,2));
    imshow(I);
    title(['image-' num2str(m) ' activations']);
    %}
    
    trainSetActv1(:,:,:,m) = Actv1tmp;
end
fprintf('\nfeature extraction finished on train set...\n');

trainSetActv1 = reshape(trainSetActv1(:), numel(trainSetActv1)/size(trainSetActv1,4), size(trainSetActv1,4));
flag = true;
st = 1;
while flag
    ed = st+100;
    if ed >= size(trainSetActv1, 2)
        ed = size(trainSetActv1, 2);
        flag = false;
    end
    trainSetActv1(:,st:ed) = mapstd(trainSetActv1(:,st:ed)', 0,1)';
    st = ed+1;
end

[V, Ssq] = svd(trainSetActv1'*trainSetActv1);
V = V(:,1:rDim);
Ssq = Ssq(1:rDim,1:rDim);
U = trainSetActv1*V*diag(diag(Ssq).^(-0.5));

% U = RandProj(size(trainSetActv1, 1), rDim);
% U = U';

trainSetActv1  =  U'*trainSetActv1;
trainSetActv1  =  trainSetActv1./( repmat(sqrt(sum(trainSetActv1.^2)), [size(trainSetActv1,1),1]) );

%save('Actv1', 'Actv1', 'idLabel', 'genderLabel');
%}


%% soft convolution with spatial pooling on testSet
testSetActv1 = zeros( ...
    floor((imgSize(1)-parSet.filterSize1+1)/parSet.downRatio2), ... %
    floor((imgSize(2)-parSet.filterSize1+1)/parSet.downRatio2),... %
    size(selectionMat2,1), size(testSet,3)  );

for m = 1:size(testSet,3)
    im = testSet(:,:,m);
    if mod(m, 20) == 0
        fprintf('%d.',m);
    end
    Actv1tmp = zeros( size(testSet,1)+1-parSet.filterSize1, size(testSet,2)+1-parSet.filterSize1, size(Dict1, 2));
    %{
    for ind = 1:size(Dict1, 2)
        kernel = reshape( Dict1(:, ind), parSet.filterSize1, parSet.filterSize1);
    %    kernel = kernel ./ max(kernel(:));
        tmp = conv2(im, flipud(fliplr(kernel)), 'valid');
        tmp(find(abs(tmp) < mean(abs(tmp(:))))) = 0;
      %  tmp  = poolFeaMap(tmp, parSet.downRatio2);
        Actv1tmp(:,:,ind) = tmp;
    end
    %}
    for i = 1:size(Dict1,2)
        kernel = reshape( Dict1(:,i), sqrt(size(Dict1,1)), sqrt(size(Dict1,1)));
        tmp = conv2(double(im), flipud(fliplr(kernel)), 'valid');
        %  tmp = tmp - mean(tmp(:));
        %  tmp = tmp./max(abs(tmp(:)));
        %  tau = 0.3*(max(abs(tmp(:))) + mean(abs(tmp(:))));
        %  tmp(find(abs(tmp) < tau)) = 0;
        Actv1tmp(:,:,i) = abs(tmp);
    end
    tau = 1.00;
    
    mat2 = sort(Actv1tmp,3);
    mat2 = mat2(:,:,5); %floor(size(mat,3)/2)
    mat2 = Actv1tmp - repmat(tau*mat2, [1, 1, size(Actv1tmp,3)]);
    mat2(find(mat2<0)) = 0;
    
    
    maskmat = sqrt( sum(mat2.^2, 3) );
    maskmat(find(maskmat==0)) = 1;
    maskmat = repmat(maskmat, [1 1 size(mat2,3)]);
    mat2 = mat2./ maskmat;
    
    Actv1tmp = mat2;
    %mat = mat - repmat(mean(mean(mat,1),2), [size(mat,1) size(mat,2) 1]);
    %Actv1tmp = mat2 ./ repmat(sqrt(sum(sum(mat2.^2,1),2)), [size(Actv1tmp,1) size(Actv1tmp,2) 1]);
    
    %{
    figure('name', ['image-' num2str(m) ' activations']);
    tmp = reshape(Actv1tmp(:), numel(Actv1tmp)/size(Actv1tmp,3), size(Actv1tmp,3));
    I = displayDictionaryElementsAsImage(tmp, sqrt(parSet.K1), sqrt(parSet.K1), size(Actv1tmp,1), size(Actv1tmp,2));
    imshow(I);
    title(['image-' num2str(m) ' activations']);
    %}
    
    Actv1tmp = pooling3D(Actv1tmp, selectionMat2, parSet.downRatio2);
    
    %{
    figure('name', ['image-' num2str(m) ' activations']);
    tmp = reshape(Actv1tmp(:), numel(Actv1tmp)/size(Actv1tmp,3), size(Actv1tmp,3));
    I = displayDictionaryElementsAsImage(tmp, sqrt(parSet.K2), sqrt(parSet.K2), size(Actv1tmp,1), size(Actv1tmp,2));
    imshow(I);
    title(['image-' num2str(m) ' activations']);
    %}
        
    testSetActv1(:,:,:,m) = Actv1tmp;
end
fprintf('\nfeature extraction finished on train set...\n');

testSetActv1 = reshape(testSetActv1(:), numel(testSetActv1)/size(testSetActv1,4), size(testSetActv1,4));

flag = true;
st = 1;
while flag
    ed = st+100;
    if ed >= size(testSetActv1, 2)
        ed = size(testSetActv1, 2);
        flag = false;
    end
    testSetActv1(:,st:ed) = mapstd(testSetActv1(:,st:ed)', 0,1)';
    st = ed+1;
end

%testSetActv1 = mapstd(testSetActv1', 0,1)';
testSetActv1  =  U'*testSetActv1;
testSetActv1  =  testSetActv1./( repmat(sqrt(sum(testSetActv1.^2)), [size(testSetActv1,1),1]) );

%%
%%{
save(['midFeaActv1_' num2str(rDim) '_3DPoolPCAProjALL'], 'testSetActv1','trainSetActv1', 'selectionMat2', 'parSet', 'U', 'Dict1', 'testLabel','trainLabel');
%}

%{
train = U*trainSetActv1;
train2 = reshape( train(:), [30 21 36 700] );
tmp = train2(:, :, :, 1);
tmp = reshape(tmp(:), [numel(tmp)/size(tmp,3) size(tmp,3)]);
I = displayDictionaryElementsAsImage(tmp, 6, 6, 30, 21 );
imshow(I);
title('learned dictionary');
%}
