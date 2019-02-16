clc; clear all;


path = ['data/original/images/'];
gt_path = 'data/original/ground_truth/';

num_images = 25;

for idx = 1:num_images
    figure;
    index = num2str(idx);
    img_name = ['IMG_' index];
    img = strcat(path,img_name,'.jpg');
    imshow(img);
    [x,y] = getpts;
    
    location = [x y];
    number = size(location,1);
    
    lo_num.location = location;
    lo_num.number = number;
    image_info = {lo_num};
    
    file_name = strcat(gt_path, 'GT_',img_name,'.mat');
    save(file_name, 'image_info');
    close all;
end
close all;
