clc
clear
close all

%% Set query and dataset folder
imgSet = imageSet([cd '/dataset_small']);
query = randi(imgSet.Count);

%% Load trained model
net = vgg16();
% net = densenet201();
% net = googlenet();

% disp(net.Layers);

imgSize = net.Layers(1).InputSize;

%% Get the feature vector for each image in dataset
h = waitbar(0,'Get the feature vector for each image in dataset');
featureVectors = zeros(imgSet.Count, 25088);
for k = 1:imgSet.Count
    % Load the image
    img = read(imgSet, k);

    % Adjust size of the image to network predefined size
    img = imresize(img, imgSize(1:2));

    % Get the feature vector
    featuremap = activations(net, img, 'pool5');
    featureVectors(k,:) = reshape(featuremap, 1, []);

    waitbar(k/imgSet.Count,h);
end
close(h);

%% retrived 5 best match images from dataset

% Calculate norm2 distances for dataset
queryFeature = featureVectors(query, :);
norm_dist = zeros(imgSet.Count, 1);
for k = 1:imgSet.Count
    norm_dist(k) = norm(featureVectors(k,:) - queryFeature);
end

% Get sort indexes of norm2 distances
[~, idx] = sort(norm_dist);

% Load the query image
queryImg = read(imgSet, query);

% Adjust size of the image to network predefined size
queryImg = imresize(queryImg, imgSize(1:2));

% Get queryImg label from network
label = classify(net, queryImg);
label = cellstr(label);

% Show the query image
subplot(2, 3, 1); imshow(queryImg); title(['query: ' label{1}]);

% Retrived first five images
for k = 1:5
    img = read(imgSet, idx(k));
    subplot(2, 3, k+1); imshow(img); title(sprintf('distance: %.2f', norm_dist(idx(k))));
end
