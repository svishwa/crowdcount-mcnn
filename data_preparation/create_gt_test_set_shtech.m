%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File to create grount truth density map for test set%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clc; clear all;
dataset = 'A';
dataset_name = ['shanghaitech_part_' dataset ];
path = ['../data/original/shanghaitech/part_' dataset '_final/test_data/images/'];
gt_path = ['../data/original/shanghaitech/part_' dataset '_final/test_data/ground_truth/'];
gt_path_csv = ['../data/original/shanghaitech/part_' dataset '_final/test_data/ground_truth_csv/'];

mkdir(gt_path_csv )
if (dataset == 'A')
    num_images = 182;
else
    num_images = 316;
end

for i = 1:num_images    
    if (mod(i,10)==0)
        fprintf(1,'Processing %3d/%d files\n', i, num_images);
    end
    load(strcat(gt_path, 'GT_IMG_',num2str(i),'.mat')) ;
    input_img_name = strcat(path,'IMG_',num2str(i),'.jpg');
    im = imread(input_img_name);
    [h, w, c] = size(im);
    if (c == 3)
        im = rgb2gray(im);
    end     
    annPoints =  image_info{1}.location;   
    [h, w, c] = size(im);
    im_density = get_density_map_gaussian(im,annPoints);    
    csvwrite([gt_path_csv ,'IMG_',num2str(i) '.csv'], im_density);       
end

