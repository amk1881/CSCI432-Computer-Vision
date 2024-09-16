% 1 

img = imread('/Users/Milana/IdeaProjects/CSCI431_ComputerVision/CSCI432-Computer-Vision/HW_01_Anna_Kurchenko_DIR/Jairaj_for_HW_01_handout.jpg');
imshow(img);

[x, y] = ginput(1); % Click on the tip of the nose and press Enter


fprintf('Column = %.2f\n', x);
fprintf('Row = %.2f\n', y);

%2 
im_gray = rgb2gray(img); 
imagesc( im_gray );
colormap( gray(256) );
axis image;  


%3 
b_low_values = im_gray < 180; 
imagesc( b_low_values );


%4 
red_channel = img(:, :, 1);   
green_channel = img(:, :, 2);
blue_channel = img(:, :, 3); 

figure;
subplot(2, 2, 1); % 2x2 grid, position 1
imshow(im_gray);
title('Original Grayscale');

subplot(2, 2, 2); % 2x2 grid, position 2
imshow(red_channel, []);
title('Red Channel');

subplot(2, 2, 3); % 2x2 grid, position 3
imshow(green_channel, []);
title('Green Channel');

subplot(2, 2, 4); % 2x2 grid, position 4
imshow(blue_channel, []);
title('Blue Channel');


% 5 
img_anna = imread('/Users/Milana/IdeaProjects/CSCI431_ComputerVision/CSCI432-Computer-Vision/HW_01_Anna_Kurchenko_DIR/Anna.png');
imagesc(img_anna)

green_channel = img_anna(:, :, 2);
gray_anna = green_channel;

rotation_angle = -30;
rotated_image = imrotate(gray_anna, rotation_angle, 'crop');

chunk_size = 650; 

[height, width] = size(rotated_image);
center_y = round(height / 2);
center_x = round(width / 2);

crop_rect = [center_x - chunk_size / 2, center_y - chunk_size / 2, chunk_size, chunk_size];
cropped_chunk = imcrop(rotated_image, crop_rect);

output_filename = 'HW01_Dynamic_Anna_Kurchenko.jpg'; 
imwrite(cropped_chunk, output_filename);

figure;
imshow(cropped_chunk);
title('Cropped Square Chunk');

% 6 
im_in_rgb = im2double(imread('HW_01_Anna_Kurchenko_DIR/Kitchen_Kolors_4670_ss.jpg')); 
[im_red, im_grn, im_blu] = imsplit(im_in_rgb);

im_red_bin = im_red >= 0.5;
im_grn_bin = im_grn >= 0.5;
im_blu_bin = im_blu >= 0.5;

im_red_bin = double(im_red_bin);
im_grn_bin = double(im_grn_bin);
im_blu_bin = double(im_blu_bin);
im_new_part_a = cat(3, im_red_bin, im_grn_bin, im_blu_bin);

figure;
imshow(im_new_part_a);
title('Image with two Levels per Channel');


% b. 
im_quantized = round(im_in_rgb * 4) / 4.0;
figure;
imshow(im_quantized);
title('Image with Four Levels per Channel');


%c. 
num_levels = 6;
im_quantized = round(im_in_rgb * (num_levels - 1)) / (num_levels - 1);
figure;
imshow(im_quantized);
title('Image with 6 Levels per Channel');


%d. 
[im_palette, my_palette] = rgb2ind(im_quantized, 256, 'nodither');
figure;
imshow(im_palette, my_palette);
title('Indexed Image with 256 Colors');

% Display the number of unique colors in the colormap
num_unique_colors = size(my_palette, 1);
fprintf('Number of unique colors found in the palette: %d\n', num_unique_colors)


% 7. 
x = 0:1200; % Degrees
y = sind(x); % Base sine wave in degrees

% Add sine waves for odd values up to 101
for odd_idx = 3:2:101
    y = y + (1 / odd_idx) * sind(odd_idx * x);
end

figure;
plot(x, y, 'b'); 
axis tight; 
xlabel('Degrees', 'FontSize', 18); 
ylabel('Sum of Sine Waves', 'FontSize', 18); 

saveas(gcf, 'Mixture_of_Sine_Waves.png'); 