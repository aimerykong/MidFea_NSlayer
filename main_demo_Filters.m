close all;
%clc

%% project dictionary back to input space;
D2 = U*((U'*U)\D);
imSize = [30 21];
D2 = reshape(D2(:), prod(imSize), numel(D2)/prod(imSize));

D3 = zeros(size(D2,1), size(D,2) );
for i = 1:size(D3,2)
    st = (i-1)*36+1;
    ed = st+35;
    D3(:,i) = mean(D2(:,st:ed), 2);  % max(D2(:,st:ed), [], 2);
end
figure('name', 'learned filters');
subplot(2,1,1);
I = displayDictionaryElementsAsImage(D2, 2, numel(D2)/prod(imSize)/2, imSize(1), imSize(2) );
imshow(I);
title('learned filters');

subplot(2,1,2);
I = displayDictionaryElementsAsImage(D3, 1, 2, imSize(1), imSize(2) );
imshow(I);
title('the mean of learned filters');


%% for a query image

imName = 'M-001-18.bmp'; % 'W-025-01.bmp';
im = imread(['../dataset/AR_database/' imName]);
if imName(1) == 'M'
    t = 1;
else
    t = 2;
end

if length(size(im)) == 3
    im = rgb2gray(im);
end
im = imresize(double(im), [66 48]);
im = double(im)./255;
% normalization
Actv1tmp = zeros( size(im,1)+1-parSet.filterSize1, size(im,2)+1-parSet.filterSize1, size(Dict1, 2));


fprintf('\n\n\nfed a query image with size %dx%d...\n', size(im,1), size(im,2) );

tic;
for i = 1:size(Dict1,2)
    kernel = reshape( Dict1(:,i), sqrt(size(Dict1,1)), sqrt(size(Dict1,1)));
    tmp = conv2(double(im), flipud(fliplr(kernel)), 'valid');
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
a(1) = toc;
fprintf('soft convolution... %.4f seconds\n', a(1));
Actv1tmp = mat2;


figure('name', 'activations');
subplot(1,4,1);
imshow(im); title('original image');

subplot(1,4,2);
display_network(Dict1);
title('low-level feature extractors');

subplot(1,4,3);
tmp = reshape(Actv1tmp(:), numel(Actv1tmp)/size(Actv1tmp,3), size(Actv1tmp,3));
I = displayDictionaryElementsAsImage(tmp, sqrt(parSet.K1), sqrt(parSet.K1), size(Actv1tmp,1), size(Actv1tmp,2));
imshow(I);
title('activations/featureMaps');

selectionMat2 = nchoosek(1:parSet.K1, 2);
tic;
Actv1tmp = pooling3D_noSwitch(Actv1tmp, selectionMat2, parSet.downRatio2);
a(2) = toc;
fprintf('3D pooling... %.4f seconds\n', a(2));


Actv1tmp = reshape( Actv1tmp(:), [30*21 36 ] ); 
subplot(1,4,4);
I = displayDictionaryElementsAsImage(Actv1tmp, 6, 6, 30, 21 );
imshow(I);
title('3D pooling featureMaps');

%%
tic;
Actv1tmp = U'*Actv1tmp(:);
a(3) = toc;
fprintf('dimensionality reduction by random projection/PCA... %.4f seconds\n', a(3));

tic;
[predicted_label, accuracy, classprobilities] = svmpredict(t', sigmoid(W*Actv1tmp+b)', model);
a(4) = toc;
fprintf('classification... %.4f seconds\n', a(4));

fprintf('\n\ntotal inference time... %.4f seconds\n\n', sum(a));

%% project back to mid-level feature space
tmp = W'*((W*W')\W)*Actv1tmp;
tmp = U*((U'*U)\tmp);

figure;
subplot(1,5,1);
imshow(im);
title('original image');

Actv1tmp = U*((U'*U)\Actv1tmp);
Actv1tmp = reshape( Actv1tmp(:), [30 21 36 ] ); 
Actv1tmp = reshape(Actv1tmp(:), [numel(Actv1tmp)/size(Actv1tmp,3) size(Actv1tmp,3)]);
subplot(1,5,2);
I = displayDictionaryElementsAsImage(Actv1tmp, 6, 6, 30, 21 );
imshow(I);
title('mid-feature');

subplot(1,5,3);
I = displayDictionaryElementsAsImage(sum(Actv1tmp,2), 1, 1, 30, 21 );
imshow(I);
title('sum-up');

subplot(1,5,4);
tmp = reshape( tmp(:), [30 21 36 ] ); 
tmp = reshape(tmp(:), [numel(tmp)/size(tmp,3) size(tmp,3)]);
I = displayDictionaryElementsAsImage(tmp, 6, 6, 30, 21 );
imshow(I);
title('response projected back to input space');

subplot(1,5,5);
I = displayDictionaryElementsAsImage(mean(tmp,2), 1, 1, 30, 21 );
imshow(I);
title('sum-up');
