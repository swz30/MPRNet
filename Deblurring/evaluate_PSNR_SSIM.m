close all;clear all;

datasets = {'GoPro'};
% datasets = {'GoPro', 'HIDE', 'Realblur_J', 'Realblur_R'};
num_set = length(datasets);

for idx_set = 1:num_set
    file_path = strcat('./results/', datasets{idx_set}, '/');
    gt_path = strcat('./Datasets/' datasets{idx_set}, '/test/target/');
    path_list = [dir(strcat(file_path,'*.jpg')); dir(strcat(file_path,'*.png'))];
    gt_list = [dir(strcat(gt_path,'*.jpg')); dir(strcat(gt_path,'*.png'))];
    img_num = length(path_list);

    total_psnr = 0;
    total_ssim = 0;
    if img_num > 0 
        for j = 1:img_num 
           image_name = path_list(j).name;
           gt_name = gt_list(j).name;
           input = imread(strcat(file_path,image_name));
           gt = imread(strcat(gt_path, gt_name));
           ssim_val = ssim(input, gt);
           psnr_val = psnr(input, gt);
           total_ssim = total_ssim + ssim_val;
           total_psnr = total_psnr + psnr_val;
       end
    end
    qm_psnr = total_psnr / img_num;
    qm_ssim = total_ssim / img_num;

end
