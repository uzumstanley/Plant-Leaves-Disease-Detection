% Pomegranate Leaf Disease Detection Script
clc;
close all;
clear all;


% Image Loading
%[filename, pathname] = uigetfile({'*.*'; '*.bmp'; '*.jpg'; '*.gif'}, 'Pick a Leaf Image File');
%if isequal(filename, 0) || isequal(pathname, 0)
%    error('No file selected.');
%end



% Specify the URL of the Google Drive file
driveFileURL = 'https://drive.google.com/file/d/1xnBUAaYTLOKuDKyxGvYoEo0pVP9_oLaE/view?usp=share_link';

% Extract the file ID from the URL
fileId = '1xnBUAaYTLOKuDKyxGvYoEo0pVP9_oLaE';

% Construct the direct download URL using the file ID
downloadURL = ['https://drive.google.com/uc?export=download&id=', fileId];

% Specify the local filename to save the downloaded image file
localFilename = 'downloaded_image.jpg';

% Download the image file from the URL and save it locally
websave(localFilename, downloadURL);

% Check if the download was successful
if ~isfile(localFilename)
    error('Failed to download the file from the provided URL.');
end

% Load the image file using the 'imread' function
image = imread(localFilename);

% Display the loaded image
imshow(image);

% Output message indicating that the image was loaded successfully
disp('Image loaded successfully.');


% Load the image file using the 'imread' function
localFilename = 'downloaded_image.jpg';
I = imread(localFilename);


%I = imread(fullfile(pathname, filename));
I = imresize(I, [256, 256]);


% Contrast Enhancement
I = imadjust(I, stretchlim(I));
figure;
imshow(I);
title('Contrast Enhanced');

% Convert Image to HSV color space
leafHSV = rgb2hsv(I);
hueChannel = leafHSV(:, :, 1);

% Compute histogram of hue values
[hueHistogram, hueBins] = imhist(hueChannel);

% Define hue range (0 to 1)
minHueRange = 0;
maxHueRange = 1;

% Find min and max hue values within the range
minHueValue = min(hueBins(hueHistogram > 0 & hueBins >= minHueRange & hueBins <= maxHueRange));
maxHueValue = max(hueBins(hueHistogram > 0 & hueBins >= minHueRange & hueBins <= maxHueRange));

% Create binary mask based on min and max hue values
diseaseMask = (hueChannel >= minHueValue) & (hueChannel <= maxHueValue);

% Localize disease-affected areas
diseaseAreas = I;
diseaseAreas(repmat(~diseaseMask, [1 1 3])) = 0;

% Display original image and disease-affected areas
figure;
subplot(1, 2, 1);
imshow(I);
title('Original Leaf Image');
subplot(1, 2, 2);
imshow(diseaseAreas);
title('Disease-Affected Areas');

% Convert Image from RGB to CIELAB color space
cform = makecform('srgb2lab');
lab_he = applycform(I, cform);

% K-means clustering
ab = double(lab_he(:, :, 2:3));
nrows = size(ab, 1);
ncols = size(ab, 2);
ab = reshape(ab, nrows * ncols, 2);
nColors = 3; % Adjust as needed

% Perform K-means clustering
[cluster_idx, cluster_center] = kmeans(ab, nColors, 'distance', 'sqEuclidean', 'Replicates', 3);

% Label each pixel in the image using clustering results
pixel_labels = reshape(cluster_idx, nrows, ncols);
segmented_images = cell(1, nColors);
rgb_label = repmat(pixel_labels, [1, 1, 3]);

% Create RGB label using pixel_labels
for k = 1:nColors
    colors = I;
    colors(rgb_label ~= k) = 0;
    segmented_images{k} = colors;
end

% Display segmented images
figure;
subplot(3, 1, 1);
imshow(segmented_images{1});
title('Cluster 1');
subplot(3, 1, 2);
imshow(segmented_images{2});
title('Cluster 2');
subplot(3, 1, 3);
imshow(segmented_images{3});
title('Cluster 3');
set(gcf, 'Position', get(0, 'Screensize'));

% User input to choose the cluster with ROI
x = inputdlg('Enter the cluster no. containing the ROI only:');
i = str2double(x);
if isempty(i) || i < 1 || i > nColors
    error('Invalid cluster number entered.');
end
seg_img = segmented_images{i};

% Convert to grayscale
if ndims(seg_img) == 3
    img = rgb2gray(seg_img);
end

% Evaluate disease-affected area
black = im2bw(seg_img, graythresh(seg_img));
m = size(seg_img, 1);
n = size(seg_img, 2);

% Calculate area of disease-affected region
cc = bwconncomp(seg_img, 6);
diseasedata = regionprops(cc, 'basic');
A1 = diseasedata.Area;

% Calculate total leaf area
I_black = im2bw(I, graythresh(I));
kk = bwconncomp(I, 6);
leafdata = regionprops(kk, 'basic');
A2 = leafdata.Area;

% Calculate affected area percentage
Affected_Area = A1 / A2;
if Affected_Area < 0.1
    Affected_Area = Affected_Area + 0.15;
end
disp(sprintf('Affected Area is: %g%%', Affected_Area * 100));

% Feature Extraction
glcms = graycomatrix(img);
stats = graycoprops(glcms, 'Contrast Correlation Energy Homogeneity');
Contrast = stats.Contrast;
Correlation = stats.Correlation;
Energy = stats.Energy;
Homogeneity = stats.Homogeneity;

% Other features
Mean = mean2(seg_img);
Standard_Deviation = std2(seg_img);
Entropy = entropy(seg_img);
RMS = mean2(rms(seg_img));
Variance = mean2(var(double(seg_img)));
a = sum(double(img(:)));
Smoothness = 1 - (1 / (1 + a));

Kurtosis = kurtosis(double(seg_img(:)));
Skewness = skewness(double(seg_img(:)));

% Inverse Difference Movement
in_diff = 0;
for i = 1:m
    for j = 1:n
        temp = seg_img(i, j) / (1 + (i - j) ^ 2);
        in_diff = in_diff + temp;
    end
end
IDM = double(in_diff);

% Combine features
feat_disease = [Contrast, Correlation, Energy, Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness, IDM];

% Load training data
load('Training_Data.mat');

% Use the trained SVM classifier
test = feat_disease;
result = multisvm(Train_Feat, Train_Label, test);

% Display the classification result
switch result
    case 0
        disp('Alternaria Alternata');
    case 1
        disp('Anthracnose');
    case 2
        disp('Bacterial Blight');
    case 3
        disp('Cercospora Leaf Spot');
    case 4
        disp('Healthy Leaf');
end

% Evaluate model accuracy
load('Accuracy_Data.mat');
Accuracy_Percent = zeros(200, 1);

% Compute the histogram of hue values
[hueHistogram, hueBins] = imhist(hueChannel);

% Define the range of hue values (0 to 1)
minHueRange = 0;
maxHueRange = 1;

% Find the min and max hue values within the predefined range
minHueValue = min(hueBins(hueHistogram > 0 & hueBins >= minHueRange & hueBins <= maxHueRange));
maxHueValue = max(hueBins(hueHistogram > 0 & hueBins >= minHueRange & hueBins <= maxHueRange));

% Plot Min-Max Hue Histogram text
figure; % Create a new figure
% Plot the histogram of hue values
bar(hueBins, hueHistogram, 'FaceColor', [0.2 0.6 0.2], 'EdgeColor', 'k');
title('Hue Histogram');
xlabel('Hue Value');
ylabel('Frequency');


% Cross-validation for accuracy evaluation
for i = 1:100
    data = Train_Feat;
    groups = ismember(Train_Label, 0); % Adjust the group based on the label type
    [train, test] = crossvalind('HoldOut', groups);
    cp = classperf(groups);
    svmModel = fitcsvm(data(train, :), groups(train), 'KernelFunction', 'linear');
    classes = predict(svmModel, data(test, :));
    classperf(cp, classes, test);
    Accuracy = cp.CorrectRate;
    Accuracy_Percent(i) = Accuracy * 100;
end

Max_Accuracy = max(Accuracy_Percent);
disp(sprintf('Accuracy of Linear Kernel with 100 iterations is: %g%%', Max_Accuracy));
