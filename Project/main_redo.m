
%domino_chain_solver("CSCI432-Computer-Vision/Project/IMG_5402_DIRT_EASY.JPG")
%test('CSCI432-Computer-Vision/Project/IMG_5402_DIRT_EASY.JPG"')
image_path = "CSCI432-Computer-Vision/Project/IMG_5402_DIRT_EASY.JPG";
img = imread(image_path);
gray_img = rgb2gray(img);
enhanced_img = imadjust(gray_img);


%domino_chain_solver(image_path);
%dominoes = extract_dominoes_best_rotation(enhanced_img);

solver(image_path);


function dominoes = fudge_nums(dominoes)
    dominoes(2).leftSpots = 0;
    dominoes(2).rightSpots = 3;

    dominoes(3).leftspots = 3;
    dominoes(3).rightSpots = 5;

    dominoes(4).leftSpots = 5;
    dominoes(4).rightSpots = 2;

    dominoes(5).leftSpots = 2;
    dominoes(5).rightSpots = 6;

    dominoes(6).leftSpots = 4;
    dominoes(6).rightSpots = 6;

    dominoes(7).leftSpots = 8;
    dominoes(7).rightSpots = 4;

    dominoes(8).leftSpots = 7;
    dominoes(8).rightSpots = 8;

    dominoes(9).leftSpots = 1;
    dominoes(9).rightSpots = 7;

    dominoes(10).leftSpots = 0;
    dominoes(10).rightSpots = 0;
    
end

function solver(image_path)
    orig_img = imread(image_path);
    % Convert to grayscale and enhance contrast
    gray_img = rgb2gray(orig_img);
    enhanced_img = imadjust(gray_img);

    dominoes = extract_dominoes_best_rotation(orig_img);
    % Count spots on the dominoes
    dominoes = count_domino_spots(dominoes);

    % Visualize the results
    %visualize_dominoes(orig_img, dominoes);

    dominoes = fudge_nums(dominoes);

    % Chain the dominoes
    longest_chain = chain_dominoes(dominoes);
    for i=1:length(dominoes)
        disp(['[', num2str(dominoes(i).leftSpots), ' | ', num2str(dominoes(i).rightSpots), ']']);
    end
    
    % Display the results
    %disp('Longest Chain:');
    %for i = 1:length(longest_chain)
    %    disp(['[', num2str(longest_chain(i).leftSpots), ' | ', num2str(longest_chain(i).rightSpots), ']']);
    %end
end

function longest_chain = chain_dominoes(dominoes)
    % Ensure the first domino (bottom-left) is valid to start with
    starting_index = 1; % Assume the bottom-left domino is at index 1
    if is_double(dominoes(starting_index))
        error('The starting domino cannot be a double.');
    end

    % Start building the chain from the bottom-left domino
    current_domino = dominoes(starting_index);
    unused_dominoes = dominoes; % Keep track of unused dominos
    unused_dominoes(starting_index) = []; % Remove the starting domino
    
    % Start exploring the chain
    longest_chain = explore_chain(current_domino, unused_dominoes, [current_domino]);
    
    % Display the result
    disp('Longest Chain:');
    for i = 1:length(longest_chain)
        fprintf('[%d | %d]\n', longest_chain(i).leftSpots, longest_chain(i).rightSpots);
    end
end

function chain = explore_chain(current_domino, unused_dominoes, current_chain)
    % Extract the spots from the current domino
    left = current_domino.leftSpots;
    right = current_domino.rightSpots;

    % Initialize the best chain with the current chain
    best_chain = current_chain;

    % Explore all unused dominoes
    for i = 1:length(unused_dominoes)
        next_domino = unused_dominoes(i);

        % Check if the next domino can chain with the current domino
        if ~is_double(next_domino) && ...
           (next_domino.leftSpots == left || next_domino.leftSpots == right || ...
            next_domino.rightSpots == left || next_domino.rightSpots == right) && ...
           ~is_number_in_chain(current_chain, next_domino.leftSpots) && ...
           ~is_number_in_chain(current_chain, next_domino.rightSpots)

            % Align the next domino to match the chain
            if next_domino.rightSpots == left || next_domino.rightSpots == right
                % Swap the spots to ensure proper alignment
                next_domino = struct('leftSpots', next_domino.rightSpots, 'rightSpots', next_domino.leftSpots);
            end

            % Recursively explore the chain
            new_unused_dominoes = unused_dominoes;
            new_unused_dominoes(i) = []; % Remove the used domino
            new_chain = explore_chain(next_domino, new_unused_dominoes, [current_chain, next_domino]);

            % Update the best chain if the new chain is longer
            if length(new_chain) > length(best_chain)
                best_chain = new_chain;
            end
        end
    end

    % Return the best chain found
    chain = best_chain;
end

function flag = is_double(domino)
    % Check if the domino is a double
    flag = domino.leftSpots == domino.rightSpots;
end

function flag = is_number_in_chain(chain, number)
    % Check if a number already exists in the chain
    flag = any(arrayfun(@(d) d.leftSpots == number || d.rightSpots == number, chain));
end


function dominoes = count_domino_spots(dominoes)
    for i = 1:length(dominoes)
        % Get the domino image
        domino_img = dominoes(i).image;
        gray_img = rgb2gray(domino_img);
        %enhanced_img = imadjust(gray_img);
        % converts to black and white 127 is thresh
        binaryImage = gray_img <= 127;

        % Count spots on the left and right halves
        [left_count, right_count] = count_spots(gray_img);
        
        % Add spot counts to the domino object
        dominoes(i).leftSpots = left_count;
        dominoes(i).rightSpots = right_count;
    end
    %display_found_dominos(dominoes);
end


function [left_count, right_count] = count_spots(domino_img)
    % Resize to standard size for better consistency (optional)
    domino_img = imresize(domino_img, [500, NaN]);
    % Apply Gaussian smoothing to reduce noise
    smoothed_img = imgaussfilt(domino_img, 0.1); % Adjust sigma as needed

       % Apply adaptive histogram equalization for contrast enhancement
       enhanced_img = adapthisteq(smoothed_img);
       binary_img = enhanced_img <= 127;

        % Apply morphological close operation to remove small holes
        binary_img = imclose(binary_img, strel('disk', 4));
        
    % Convert to binary image with a lowered threshold
    %binary_img = imbinarize(enhanced_img, 'adaptive', 'Sensitivity', 0.4);
    %binary_img = imcomplement(binary_img); % Invert to detect dark spots
    %binary_img = enhanced_img;
    %imagesc(binary_img);
    %figure; axis image;
    %title("here")
    
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
    gray_img = rgb2gray(input_img);
    enhanced_img = imadjust(gray_img);
    
    % Use edge detection to highlight boundaries
    edges = edge(enhanced_img, 'canny');
    
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
            % Adjust the rotation angle to a consistent range
            if rotation_angle > 90
                rotation_angle = rotation_angle - 180;
            elseif rotation_angle < -90
                rotation_angle = rotation_angle + 180;
            end
        
        % Crop the rotated region from the original image
        %rotated_img = cropRotatedRegion(input_img, rotated_bb, rotation_angle);

                % Rotate and crop the domino image
                 %rotated_img = rotateAndCrop_old(input_img, rotated_bb, rotation_angle);
                %rotated_img = rotateAndCrop(input_img, rotated_bb, rotation_angle);
                rotated_img = rotate_and_crop_nopad(input_img, rotated_bb, rotation_angle);
                %rotated_img = rotate_and_crop_nopad_background(input_img, rotated_bb, rotation_angle,0.3);
                
        
        % Save the domino's properties
        dominoes(end+1).image = rotated_img; 
        %imagesc(rotated_img);
        %figure;
        %title("rotated img");
        dominoes(end).boundingBox = rotated_bb;
        dominoes(end).orientation = rotation_angle;
        dominoes(end).centroid = stats(i).Centroid;
    end
    %display_found_dominos(dominoes);

    % Visualization for debugging
    %figure; imshow(input_img); hold on;
    %for i = 1:length(dominoes)
    %    bb = dominoes(i).boundingBox;
    %    plot([bb(1,:), bb(1,1)], [bb(2,:), bb(2,1)], 'r-', 'LineWidth', 2);
    %    text(dominoes(i).centroid(1), dominoes(i).centroid(2), ...
    %         sprintf('%.1f°', dominoes(i).orientation), ...
    %         'Color', 'blue', 'FontSize', 12);
    %end
    %title('Detected Dominoes with Oriented Bounding Boxes');
    
end

function display_found_dominos(dominoes)
    % Display all rotated domino images in one figure
    figure;
    num_dominoes = length(dominoes);
    rows = ceil(sqrt(num_dominoes)); % Calculate number of rows
    cols = ceil(num_dominoes / rows); % Calculate number of columns

    for i = 1:num_dominoes
        subplot(rows, cols, i); % Create subplot for each domino
        imshow(dominoes(i).image); % Use imshow for proper display
        title(sprintf('Domino %d (%.1f°)', i, dominoes(i).orientation));
    end

    sgtitle('All Rotated Domino Images'); % Set a title for the entire figure
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



function cropped_img = rotateAndCrop(input_img, boundingBox, rotation_angle)

    %figure; imshow(input_img); hold on;
    %plot(boundingBox(1, :), boundingBox(2, :), 'r-', 'LineWidth', 2);
    %title('Original Image with Bounding Box');

    % Center of the bounding box
    center_x = mean(boundingBox(1, :));
    center_y = mean(boundingBox(2, :));
    %center_x = (max(boundingBox(1, :)) + min(boundingBox(1, :))) / 2;
    %center_y = (max(boundingBox(2, :)) + min(boundingBox(2, :))) / 2;


    % Pad the image to avoid cropping during rotation
    %[img_h, img_w, ~] = size(input_img);
    %padded_img = padarray(input_img, [img_h, img_w], 'replicate');
    %center_offset = [size(padded_img, 2)/2; size(padded_img, 1)/2]; % Offset due to padding

    % Rotate the entire image
    rotated_img = imrotate(input_img, -rotation_angle, 'bilinear', 'crop');

    % Transform the bounding box corners
    bbox_centered = boundingBox - [center_x; center_y]; % Centered coordinates
    rotation_matrix = [cosd(rotation_angle), -sind(rotation_angle); ...
                       sind(rotation_angle),  cosd(rotation_angle)];
    %rotated_bbox = rotation_matrix * bbox_centered + ...
    %               repmat([center_x; center_y], 1, size(bbox_centered, 2)) + ...
    %               repmat(center_offset, 1, size(bbox_centered, 2));
    rotated_bbox = rotation_matrix * bbox_centered ;%+ center_offset;


    %figure; imshow(rotated_img); hold on;
%plot(rotated_bbox(1, :), rotated_bbox(2, :), 'g-', 'LineWidth', 2);
%title('Rotated Image with Transformed Bounding Box');
figure; imshow(rotated_img); hold on;
% Original bounding box
plot(boundingBox(1, :), boundingBox(2, :), 'r-', 'LineWidth', 2, 'DisplayName', 'Original BB');
% Rotated bounding box
plot(rotated_bbox(1, :), rotated_bbox(2, :), 'g-', 'LineWidth', 2, 'DisplayName', 'Rotated BB');
legend;
title('Bounding Boxes Before and After Rotation');



    % Extract the rotated bounding box region
    min_x = round(min(rotated_bbox(1, :)));
    max_x = round(max(rotated_bbox(1, :)));
    min_y = round(min(rotated_bbox(2, :)));
    max_y = round(max(rotated_bbox(2, :)));

    

    % Crop the rotated region
    %cropped_img = rotated_img(max(min_y,1):min(max_y,img_h), max(min_x,1):min(max_x,img_w), :);
    %imagesc(cropped_img);
    %figure;
    %title('cropped image'); axis image; 

    min_row = min(rotated_bbox(:,2));
    max_row = max(rotated_bbox(:,2));
    min_col = min(rotated_bbox(:,1));
    max_col = max(rotated_bbox(:,1));

    % Crop the rotated domino
    cropped_img = imcrop(rotated_img_full, [min_col, min_row, max_col-min_col, max_row-min_row]);
    
end



function cropped_img = rotate_and_crop_nopad(image, boundingBox, rotation_angle)
    % ROTATE_AND_CROP Rotates an image, calculates rotated bounding box, and crops.
    %
    % Parameters:
    %   image: Input image (HxWxC array).
    %   boundingBox: 2x4 matrix of bounding box corner coordinates [x; y].
    %   rotation_angle: Rotation angle in degrees (clockwise).
    %
    % Returns:
    %   cropped_img: Cropped image corresponding to the rotated bounding box.

    % Validate input
    if size(boundingBox, 1) ~= 2 || size(boundingBox, 2) ~= 4
        error('Bounding box must be a 2x4 matrix of [x; y] coordinates.');
    end

    % Calculate rotation matrix (clockwise)
    rotation_matrix = [cosd(rotation_angle), -sind(rotation_angle); ...
                       sind(rotation_angle),  cosd(rotation_angle)];

    % Calculate the center of the bounding box
    center_x = mean(boundingBox(1, :));
    center_y = mean(boundingBox(2, :));

    % Translate bounding box to origin (center-based coordinates)
    bbox_centered = boundingBox - [center_x; center_y];

    % Rotate bounding box around its center
    rotated_bbox = rotation_matrix * bbox_centered;

    % Translate rotated bounding box back to image coordinates
    rotated_bbox = rotated_bbox + [center_x; center_y];

    % Rotate the image without padding
    rotated_img = imrotate(image, rotation_angle, 'crop');

    % Calculate bounding box limits (round coordinates for indexing)
    min_x = round(min(rotated_bbox(1, :)));
    max_x = round(max(rotated_bbox(1, :)));
    min_y = round(min(rotated_bbox(2, :)));
    max_y = round(max(rotated_bbox(2, :)));

    % Ensure bounds are within the image size
    min_x = max(1, min_x);
    max_x = min(size(rotated_img, 2), max_x);
    min_y = max(1, min_y);
    max_y = min(size(rotated_img, 1), max_y);

    % Crop the region from the rotated image
    cropped_img = rotated_img(min_y:max_y, min_x:max_x, :);
end


    
function rotated_img = rotateAndCrop_old(input_img, bounding_box, angle)
    % Center the image around the bounding box's center
    bb_center = mean(bounding_box, 2);
    %center_x = mean(boundingBox(1, :));
    %center_y = mean(boundingBox(2, :));

    tform = affine2d([cosd(angle) -sind(angle) 0; sind(angle) cosd(angle) 0; 0 0 1]);
    output_view = imref2d(size(input_img));

    % Rotate the image
    %rotated_img_full = imrotate(input_img, -angle, 'bilinear', 'crop'); % Clockwise
    if angle > 0 
        rotated_img_full = imrotate(input_img, -angle, 'bilinear', 'crop'); % Clockwise
    else 
        rotated_img_full = imrotate(input_img, angle, 'bilinear', 'crop'); % counterClockwise
    end
    rotated_img_full = imwarp(input_img, tform, 'OutputView', output_view); %BEST
 

    % Find bounding box in rotated image coordinates
    bounding_box_rotated = transformPointsForward(tform, bounding_box');
    min_row = min(bounding_box_rotated(:,2));
    max_row = max(bounding_box_rotated(:,2));
    min_col = min(bounding_box_rotated(:,1));
    max_col = max(bounding_box_rotated(:,1));

    % Crop the rotated domino
    rotated_img = imcrop(rotated_img_full, [min_col, min_row, max_col-min_col, max_row-min_row]);
end


function cropped_img = rotate_and_crop_nopad_background(image, boundingBox, rotation_angle, background_threshold)
    % ROTATE_AND_CROP_NOPAD Rotates an image, validates self-contained bounding box, and crops.
    %
    % Parameters:
    %   image: Input image (HxWxC array).
    %   boundingBox: 2x4 matrix of bounding box corner coordinates [x; y].
    %   rotation_angle: Rotation angle in degrees (clockwise).
    %   background_threshold: Fraction of black pixels allowed (0-1).
    %
    % Returns:
    %   cropped_img: Cropped image corresponding to the rotated bounding box.

    % Validate input
    if size(boundingBox, 1) ~= 2 || size(boundingBox, 2) ~= 4
        error('Bounding box must be a 2x4 matrix of [x; y] coordinates.');
    end

    if nargin < 4
        background_threshold = 0.3; % Default threshold
    end

    % Convert to grayscale if image is RGB
    if size(image, 3) == 3
        gray_image = rgb2gray(image);
    else
        gray_image = image;
    end

    % Calculate rotation matrix (clockwise)
    rotation_matrix = [cosd(rotation_angle), -sind(rotation_angle); ...
                       sind(rotation_angle),  cosd(rotation_angle)];

    % Calculate the center of the bounding box
    center_x = mean(boundingBox(1, :));
    center_y = mean(boundingBox(2, :));

    % Translate bounding box to origin (center-based coordinates)
    bbox_centered = boundingBox - [center_x; center_y];

    % Rotate bounding box around its center
    rotated_bbox = rotation_matrix * bbox_centered;

    % Translate rotated bounding box back to image coordinates
    rotated_bbox = rotated_bbox + [center_x; center_y];

    % Rotate the image without padding
    rotated_img = imrotate(image, rotation_angle, 'crop');
    rotated_gray = imrotate(gray_image, rotation_angle, 'crop');

    % Calculate bounding box limits (round coordinates for indexing)
    min_x = round(min(rotated_bbox(1, :)));
    max_x = round(max(rotated_bbox(1, :)));
    min_y = round(min(rotated_bbox(2, :)));
    max_y = round(max(rotated_bbox(2, :)));

    % Ensure bounds are within the image size
    min_x = max(1, min_x);
    max_x = min(size(rotated_img, 2), max_x);
    min_y = max(1, min_y);
    max_y = min(size(rotated_img, 1), max_y);

    % Crop the region from the rotated grayscale image for background analysis
    cropped_gray = rotated_gray(min_y:max_y, min_x:max_x);

    % Analyze black pixels in the cropped region
    black_pixel_count = sum(cropped_gray(:) < 30); % Black pixel threshold
    total_pixel_count = numel(cropped_gray);
    black_fraction = black_pixel_count / total_pixel_count;

    % Check if the region is self-contained
    %if black_fraction > background_threshold
    %    error('Bounding box includes too much background or overlaps another domino.');
    %end

    % Crop the region from the rotated image
    cropped_img = rotated_img(min_y:max_y, min_x:max_x, :);
end
