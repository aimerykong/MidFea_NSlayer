function pooledActv1 = pooling3D_noSwitch(Actv1, selectionMat, poolRatio)

pooledActv1 = zeros( floor(size(Actv1,1)/poolRatio), floor(size(Actv1,2)/poolRatio), size(selectionMat,1), size(Actv1, 4) );

tmp = zeros( size(Actv1,1),  size(Actv1,2), size(selectionMat,1) );

for i = 1:size(selectionMat, 1)
    tmp(:,:,i) = max( abs(Actv1(:, :, selectionMat(i,:))), [], 3 );    
end

for y = 1:size(pooledActv1,2)
    for x = 1:size(pooledActv1,1)
        cubic = tmp( ((x-1)*poolRatio+1):(x*poolRatio), ((y-1)*poolRatio+1):(y*poolRatio), : );
        cubic = reshape( abs(cubic(:)), [poolRatio^2, size(cubic,3)] );
        pooledActv1(x,y,:) = max(cubic, [], 1);        
    end
end
