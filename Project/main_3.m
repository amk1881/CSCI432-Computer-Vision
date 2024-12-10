
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

    dominoes(6).leftSpots = 6;
    dominoes(6).rightSpots = 4;

    dominoes(7).leftSpots = 6;
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
    imagesc(orig_img);
    
    % Convert to grayscale and enhance contrast
    gray_img = rgb2gray(orig_img);
    enhanced_img = imadjust(gray_img);

    dominoes = extract_dominoes_best_rotation(enhanced_img);
    % Count spots on the dominoes
    dominoes = count_domino_spots(dominoes);

    % Visualize the results
    visualize_dominoes(orig_img, dominoes);

    %dominoes = fudge_nums(dominoes);

    % Chain the dominoes
    %longest_chain = chain_dominoes(dominoes);
    %for i=1:length(dominoes)
    %    disp(dominoes(i));
    %end


    % Display the results
    %disp('Longest Chain:');
    %for i = 1:length(longest_chain)
    %    disp(['[', num2str(longest_chain(i).leftSpots), ' | ', num2str(longest_chain(i).rightSpots), ']']);
    %end
end


function dominoes = count_domino_spots(dominoes)
    for i = 1:length(dominoes)
        % Get the domino image
        domino_img = dominoes(i).image;
        
        % Count spots on the left and right halves
        [left_count, right_count] = count_spots(domino_img);
        
        % Add spot counts to the domino object
        dominoes(i).leftSpots = left_count;
        dominoes(i).rightSpots = right_count;
    end
end
% go to higher res image to do analysis 
% when we subsample the image 
% maybe don't binaerize the image, because cyan gets ignored 
% imbinarize runs otsus udner the covers it finds the best guess - still a guess so things get thrown away 
% might be better saying I know waht black is and if its not that +- a couple pizels 
% hard code black to 0.3, throw away small dots udner a cerain size before this 
% background removal 


function [left_count, right_count] = count_spots(domino_img)
    % Resize to standard size for better consistency (optional)
    domino_img = imresize(domino_img, [100, NaN]); % chnaged 100 to 500
    %figure('Position', [400 100 1200 1000]); axis image; 
    %imshow(domino_img);
   %pause;
    % Apply Gaussian smoothing to reduce noise
    smoothed_img = imgaussfilt(domino_img, 0.1); % Adjust sigma as needed

       % Apply adaptive histogram equalization for contrast enhancement
       enhanced_img = adapthisteq(smoothed_img);


    % Convert to binary image with a lowered threshold
    binary_img = imbinarize(enhanced_img, 'adaptive', 'Sensitivity', 0.4);
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




function longest_chain = chain_dominoes(dominoes)
    % Start with the first domino (assumed to be in the bottom-left corner)
    % Use all dominoes as potential candidates for chaining
    visited = false(size(dominoes)); % Keep track of used dominoes
    longest_chain = []; % Track the longest chain found

    for i = 1:length(dominoes)
        if ~is_double(dominoes(i))
            % Explore chains starting with each non-double domino
            chain = explore_chain(dominoes, i, visited, []);
            if length(chain) > length(longest_chain)
                longest_chain = chain;
            end
        end
    end
end

function chain = explore_chain(dominoes, current_index, visited, chain)
    % Add the current domino to the chain
    chain = [chain, dominoes(current_index)];
    visited(current_index) = true;

    % Get the current domino's numbers
    left = dominoes(current_index).leftSpots;
    right = dominoes(current_index).rightSpots;

    % Explore further connections
    for i = 1:length(dominoes)
        if ~visited(i) && ~is_double(dominoes(i)) && ...
           (dominoes(i).leftSpots == left || dominoes(i).leftSpots == right || ...
            dominoes(i).rightSpots == left || dominoes(i).rightSpots == right) && ...
           ~is_number_in_chain(chain, dominoes(i).leftSpots) && ...
           ~is_number_in_chain(chain, dominoes(i).rightSpots)
        
            % Recursively explore the next domino
            chain = explore_chain(dominoes, i, visited, chain);
        end
    end
end

function flag = is_double(domino)
    % Check if the domino is a double
    flag = domino.leftSpots == domino.rightSpots;
end

function flag = is_number_in_chain(chain, number)
    % Check if a number already exists in the chain
    flag = any(arrayfun(@(d) d.leftSpots == number || d.rightSpots == number, chain));
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
    
