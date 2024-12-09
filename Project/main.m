
%domino_chain_solver("CSCI432-Computer-Vision/Project/IMG_5402_DIRT_EASY.JPG")
%test('CSCI432-Computer-Vision/Project/IMG_5402_DIRT_EASY.JPG"')
image_path = "CSCI432-Computer-Vision/Project/IMG_5402_DIRT_EASY.JPG";
img = imread(image_path);
gray_img = rgb2gray(img);
enhanced_img = imadjust(gray_img);


%domino_chain_solver(image_path);
%dominoes = extract_dominoes_best_rotation(enhanced_img);

solver(image_path)

function solver(image_path)
    orig_img = imread(image_path);
    imagesc(orig_img);
    
    % Convert to grayscale and enhance contrast
    gray_img = rgb2gray(orig_img);
    enhanced_img = imadjust(gray_img);

    dominoes = extract_dominoes_best_rotation(enhanced_img);
    
    % Count spots on the dominoes
    dominoes = count_domino_spots(dominoes);

    % Visualize the results
    visualize_dominoes(orig_img, dominoes);
end

function dominoes = count_domino_spots(dominoes)
    for i = 1:length(dominoes)
        % Get the domino image
        domino_img = dominoes(i).image;
        
        % Count spots on the left and right halves
        [left_count, right_count] = count_spots_with_circles(domino_img);
        
        % Add spot counts to the domino object
        dominoes(i).leftSpots = left_count;
        dominoes(i).rightSpots = right_count;
    end
end

function [left_spots, right_spots] = count_spots_with_circles(domino_img)
    % Resize the image to a consistent size (optional)
    domino_img = imresize(domino_img, [100, NaN]); 
    
    % Apply adaptive histogram equalization for contrast enhancement
    enhanced_img = adapthisteq(domino_img);
    
    % Convert to binary image
    binary_img = imbinarize(enhanced_img, 'adaptive', 'Sensitivity', 0.4);
    
    % Invert the image to detect dark spots
    binary_img = imcomplement(binary_img);
    
    % Perform morphological operations to clean up the binary image
    cleaned_img = imopen(binary_img, strel('disk', 2));
    
    % Divide the domino into left and right halves
    [height, width] = size(cleaned_img);
    left_half = cleaned_img(:, 1:floor(width / 2));
    right_half = cleaned_img(:, ceil(width / 2):end);
    
    % Use imfindcircles to find circles (spots) in the left and right halves
    [left_centers, left_radii] = imfindcircles(left_half, [10 30], 'ObjectPolarity', 'dark', 'Sensitivity', 0.9);
    [right_centers, right_radii] = imfindcircles(right_half, [10 30], 'ObjectPolarity', 'dark', 'Sensitivity', 0.9);
    
    % Count the number of detected circles (spots)
    left_spots = length(left_centers);
    right_spots = length(right_centers);
    
    % Debugging: Visualize the detected circles
    figure;
    subplot(1, 2, 1);
    imshow(left_half);
    title('Left Half - Detected Circles');
    hold on;
    viscircles(left_centers, left_radii, 'EdgeColor', 'r'); % Plot detected circles
    
    subplot(1, 2, 2);
    imshow(right_half);
    title('Right Half - Detected Circles');
    hold on;
    viscircles(right_centers, right_radii, 'EdgeColor', 'r'); % Plot detected circles
end



function [left_spots, right_spots] = count_spots_refined(domino_img)
    % Resize image to normalize scale
    domino_img = imresize(domino_img, [100, NaN]); 
    
    % Apply adaptive histogram equalization for contrast enhancement
    enhanced_img = adapthisteq(domino_img);
    
    % Apply a median filter for noise reduction while preserving edges
    filtered_img = medfilt2(enhanced_img, [3, 3]); % 3x3 filter window

    % Convert to binary image with a lower threshold to capture all spots
    binary_img = imbinarize(filtered_img, 'adaptive', 'Sensitivity', 0.4);
    
    % Invert the image to detect dark spots
    binary_img = imcomplement(binary_img);
    
    % Perform morphological opening to remove small noise
    clean_img = imopen(binary_img, strel('disk', 2));
    
    % Divide the domino into left and right halves
    [height, width] = size(clean_img);
    left_half = clean_img(:, 1:floor(width / 2));
    right_half = clean_img(:, ceil(width / 2):end);

    % Label and count connected components in each half
    left_labeled = bwlabel(left_half);
    right_labeled = bwlabel(right_half);

    % Extract region properties for filtering noise
    left_stats = regionprops(left_labeled, 'Area');
    right_stats = regionprops(right_labeled, 'Area');
    
    % Filter out small regions
    min_area = 10; % Adjust based on expected spot size
    left_spots = sum([left_stats.Area] > min_area);
    right_spots = sum([right_stats.Area] > min_area);

    % Debugging: Visualize steps
    figure;
    subplot(2, 3, 1), imshow(domino_img), title('Original Domino');
    subplot(2, 3, 2), imshow(enhanced_img), title('Enhanced Contrast');
    subplot(2, 3, 3), imshow(filtered_img), title('Filtered Image');
    subplot(2, 3, 4), imshow(binary_img), title('Binarized Image');
    subplot(2, 3, 5), imshow(clean_img), title('Cleaned Binary');
    %subplot(2, 3, 6), imshow(left_half , right_half), title('Left and Right Halves');
end


function [left_count, right_count] = count_spots_lab(domino_img)
    % Resize to standard size for better consistency (optional)
    max_height = 300;
    [rows, ~, channels] = size(domino_img);
    if rows > max_height
        domino_img = imresize(domino_img, [max_height, NaN]);
    end

        % Ensure the image is RGB
        if channels == 1
            % Convert grayscale to RGB by replicating the single channel
            domino_img = repmat(domino_img, [1, 1, 3]);
        end

    % Convert to Lab color space
    lab_img = rgb2lab(domino_img);

    % Extract the Luminance (L) channel
    L_channel = lab_img(:, :, 1);

    % Normalize L channel (range 0 to 1)
    L_channel = mat2gray(L_channel);

    % Enhance contrast
    enhanced_L = imadjust(L_channel, stretchlim(L_channel, [0.02, 0.98]), []);

    % Binarize the enhanced L channel
    binary_img = imbinarize(enhanced_L, 'adaptive', 'Sensitivity', 0.4);
    binary_img = imcomplement(binary_img); % Invert to detect dark spots

    % Split into left and right halves
    mid_col = floor(size(binary_img, 2) / 2);
    left_half = binary_img(:, 1:mid_col);
    right_half = binary_img(:, mid_col+1:end);

    % Use connected components or circle detection to count dots
    left_cc = bwconncomp(left_half);
    right_cc = bwconncomp(right_half);

    % Count the number of detected dots
    left_count = left_cc.NumObjects;
    right_count = right_cc.NumObjects;

% Visualize for debugging
figure;
subplot(1, 3, 1);
imshow(L_channel);
title('Luminance Channel (L)');

subplot(1, 3, 2);
imshow(binary_img);
title('Binary Image');

subplot(1, 3, 3);
imshow(domino_img);
title('Original Image');

% Optional: Visualize centroids
left_props = regionprops(left_cc, 'Centroid');
left_centroids = vertcat(left_props.Centroid);
hold on;
if ~isempty(left_centroids)
    plot(left_centroids(:, 1), left_centroids(:, 2), 'ro', 'MarkerSize', 5);
end

end


%{
function [left_spots, right_spots] = count_spots_via_gaussian(domino_img)
    % Resize image to normalize scale
    domino_img = imresize(domino_img, [100, NaN]); 
    
    % Apply Gaussian smoothing to reduce noise
    smoothed_img = imgaussfilt(domino_img, 0.4); % Adjust sigma as needed

    % Apply adaptive histogram equalization for contrast enhancement
    enhanced_img = adapthisteq(smoothed_img);
    
    % Convert to binary image with a lowered threshold
    binary_img = imbinarize(enhanced_img, 'adaptive', 'Sensitivity', 0.4);
    
    % Invert the image to detect dark spots
    binary_img = imcomplement(binary_img);
    figure;imagesc(binary_img);
    title("here")

    % Shrink the image further to reduce noise in large background areas
    shrunken_img = imresize(binary_img, 0.5); 
    
    % Perform morphological opening to remove small noise
    clean_img = imopen(shrunken_img, strel('disk', 2)); 
    
    % Divide the domino into left and right halves
    [height, width] = size(clean_img);
    left_half = clean_img(:, 1:floor(width / 2));
    right_half = clean_img(:, ceil(width / 2):end);

    % Label and count connected components in each half
    left_labeled = bwlabel(left_half);
    right_labeled = bwlabel(right_half);

    % Extract region properties for filtering noise
    left_stats = regionprops(left_labeled, 'Area');
    right_stats = regionprops(right_labeled, 'Area');
    
    % Filter out small regions
    min_area = 10; % Adjust this threshold based on spot size
    left_spots = sum([left_stats.Area] > min_area);
    right_spots = sum([right_stats.Area] > min_area);

    % Debugging: Visualize steps
    %figure;
    %subplot(1, 3, 1), imshow(domino_img), title('Original Domino');
    %subplot(1, 3, 2), imshow(binary_img), title('Binarized Domino');
    %subplot(1, 3, 3), imshow(clean_img), title('Cleaned Binary');
end


function [left_count, right_count] = count_spots(domino_img)
    % Resize to standard size for better consistency (optional)
    domino_img = imresize(domino_img, [100, NaN]);
    
    % Convert to binary for spot detection
    binary_img = imbinarize(domino_img, 'adaptive'); % Adaptive thresholding
    binary_img = imcomplement(binary_img); % Invert to detect dark spots
    
    % Split into left and right halves
    mid_col = floor(size(binary_img, 2) / 2);
    left_half = binary_img(:, 1:mid_col);
    right_half = binary_img(:, mid_col+1:end);
    
    % Count connected components in each half
    left_cc = bwconncomp(left_half);
    right_cc = bwconncomp(right_half);
    left_count = left_cc.NumObjects;
    right_count = right_cc.NumObjects;
end
%}

function visualize_dominoes(orig_img, dominoes)
    % Display original image with annotations
    figure; imshow(orig_img); hold on;
    for i = 1:length(dominoes)
        % Draw the rotated bounding box
        bb = dominoes(i).boundingBox;
        plot([bb(1,:), bb(1,1)], [bb(2,:), bb(2,1)], 'r-', 'LineWidth', 2);
        
        % Display orientation angle
        text(dominoes(i).centroid(1), dominoes(i).centroid(2), ...
             sprintf('L:%d R:%d', dominoes(i).leftSpots, dominoes(i).rightSpots), ...
             'Color', 'blue', 'FontSize', 15);
    end
    title('Detected Dominoes with Spot Counts');
end


function dominoes = extract_dominoes_best_rotation(input_img)
    % Use edge detection to highlight boundaries
    edges = edge(input_img, 'canny');
    
    % Dilate the edges to connect broken parts
    dilated_edges = imdilate(edges, strel('disk', 5));
    
    % Fill enclosed regions
    filled_img = imfill(dilated_edges, 'holes');
    
    % Filter out small regions that are not domino-sized
    binary_img = bwareaopen(filled_img, 7000); % Keep only large regions
    
    % Label connected components
    labeled_img = bwlabel(binary_img);
    stats = regionprops(labeled_img, 'Area', 'Orientation', 'Centroid', 'PixelIdxList', 'Image');
    
    dominoes = struct([]);
    for i = 1:length(stats)
        % Extract pixels of the current domino
        mask = false(size(binary_img));
        mask(stats(i).PixelIdxList) = true;
        
        % Find the convex hull of the domino
        [row, col] = find(mask);
        points = [col, row]; % (x, y) coordinates
        hull_indices = convhull(points,'Simplify',true); % Convex hull indices
        hull_points = points(hull_indices, :); % Convex hull points
        
        % Fit a minimum bounding rectangle to the convex hull
        [rotated_bb, rotation_angle] = minBoundingBox(hull_points);
        
        % Crop the rotated region from the original image
        rotated_img = cropRotatedRegion(input_img, rotated_bb, rotation_angle);
        
        % Save the domino's properties
        dominoes(end+1).image = rotated_img; 
        dominoes(end).boundingBox = rotated_bb;
        dominoes(end).orientation = rotation_angle;
        dominoes(end).centroid = stats(i).Centroid;
    end
    
    % Debugging: Visualize rotated bounding boxes
    %{
    figure; imshow(input_img); hold on;
    for i = 1:length(dominoes)
        % Draw the rotated bounding box
        bb = dominoes(i).boundingBox;
        plot([bb(1,:), bb(1,1)], [bb(2,:), bb(2,1)], 'r-', 'LineWidth', 2);
        
        % Display orientation angle
        text(dominoes(i).centroid(1), dominoes(i).centroid(2), ...
             sprintf('%.1f°', dominoes(i).orientation), ...
             'Color', 'yellow', 'FontSize', 8);
    end
    title('Detected Dominoes with Oriented Bounding Boxes');
    %}
end

function [rotated_bb, rotation_angle] = minBoundingBox(points)
    % Fit a minimum bounding box around a set of points
    % Input: points - Nx2 matrix of (x, y) coordinates
    % Output: rotated_bb - 2x4 matrix (x, y) of the bounding box corners
    %         rotation_angle - angle of rotation in degrees
    
    % Perform Principal Component Analysis (PCA)
    coeff = pca(points);
    rotated_points = points * coeff; % Rotate points to align with PCA axes
    
    % Find the bounding box in the PCA space
    x_min = min(rotated_points(:, 1));
    x_max = max(rotated_points(:, 1));
    y_min = min(rotated_points(:, 2));
    y_max = max(rotated_points(:, 2));
    
    % Construct the bounding box in PCA space
    box_pca = [x_min, y_min; x_max, y_min; x_max, y_max; x_min, y_max];
    
    % Rotate the bounding box back to original space
    rotated_bb = (box_pca * coeff')';
    rotation_angle = atan2d(coeff(2, 1), coeff(1, 1)); % Angle of the principal axis
end

function cropped_img = cropRotatedRegion(img, bb, angle)
    % Crop a rotated region from the image
    % Input: img - the original image
    %        bb - 2x4 matrix (x, y) of bounding box corners
    %        angle - rotation angle in degrees
    
    % Define the bounding polygon
    mask = poly2mask(bb(1, :), bb(2, :), size(img, 1), size(img, 2));
    
    % Rotate the entire image for easy cropping
    rotated_img = imrotate(img, -angle, 'crop');
    
    % Crop the rotated image using the mask
    cropped_img = bsxfun(@times, rotated_img, cast(mask, 'like', img));
end
    
