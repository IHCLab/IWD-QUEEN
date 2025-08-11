clc,clear,close all;

%% Setting - Path 

cygnss_data_path = './cygnss_data';
gt_data_path = "./Merit_Hydro_Amazon.mat";
collect_result_save_path = './collect_result';

%% Setting - region

region = [-69 -67 -5 -3]; 

%% Start

month_folders = dir(cygnss_data_path);
month_folders = month_folders([month_folders.isdir]);
month_folders = month_folders(~ismember({month_folders.name}, {'.', '..'}));

for i = 1:length(month_folders)
    month_folder = fullfile(cygnss_data_path,month_folders(i).name);
    day_folder = dir(month_folder);
    day_folder = day_folder([day_folder.isdir]);
    day_folder = day_folder(~ismember({day_folder.name}, {'.', '..'}));
    for j = 1:length(day_folder)
        try
            folder_path = fullfile(month_folder, day_folder(j).name);
            file_name = strcat(day_folder(j).name,'.mat');
            save_path = fullfile(collect_result_save_path,month_folders(i).name,file_name);

            if ~exist(fullfile(collect_result_save_path,month_folders(i).name),'dir')
            mkdir(fullfile(collect_result_save_path,month_folders(i).name))
            end

            fprintf('------------------------------------------------------------------------------- \n')
            fprintf("Start finding... '%s' \n",folder_path);
            fprintf("\n");

            collect_data(folder_path,save_path,region,gt_data_path);

            fprintf('\n');
            fprintf("Save at... '%s' \n",save_path)
            fprintf('------------------------------------------------------------------------------- \n')
            fprintf("\n\n")
        catch ME
            fprintf('Error encountered in: %s. Skipping...\n', folder_path);
            fprintf('Error message: %s\n', ME.message);
            fprintf('------------------------------------------------------------------------------- \n');
        end
    end
end

%% Remove error data
month_folders = dir(collect_result_save_path);
month_folders = month_folders([month_folders.isdir]);
month_folders = month_folders(~ismember({month_folders.name}, {'.', '..'}));

land_num = 0;
surfaceWater_num = 0;

for i = 1:length(month_folders)
    month_folder = fullfile(collect_result_save_path,month_folders(i).name);
    day_datas = dir(month_folder);
    day_datas = day_datas(~ismember({day_datas.name}, {'.', '..'}));
    for j = 1:length(day_datas)
        file_path = fullfile(month_folder, day_datas(j).name);
        load(file_path);
        indices_to_remove = [];
        for k = 1:size(ddm, 3)
            ddm_has_nan = any(isnan(ddm(:,:,k)), 'all');
            ddm_all_zero = all(ddm(:,:,k) == 0, 'all');

            if ddm_has_nan || ddm_all_zero
                fprintf('Problem detected at %s, index: %d\n', file_path, k);
                indices_to_remove = [indices_to_remove, k]; 
            end
        end

        if ~isempty(indices_to_remove)
            ddm(:,:,indices_to_remove) = [];
            index(indices_to_remove, :) = [];
            location(indices_to_remove, :) = [];
            fprintf('Remove data from: %s, indices: %s\n', file_path, mat2str(indices_to_remove));
            save(file_path, 'ddm', 'index', 'location');
            fprintf('Saved updated data to: %s\n', file_path);
            fprintf('----------------------------------------------------------- \n')
        end
    end
end
fprintf('Finish \n')