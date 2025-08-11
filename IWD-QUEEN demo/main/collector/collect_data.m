%% collect_data
% Read CYGNSS .nc data files, crop by specified geographic region,
% apply quality control, match with ground truth (GT) labels,
% and save processed DDMs (Delay-Doppler Maps) with location and labels.
%
% INPUT:
%   base_path  - Path to directory containing .nc files
%   save_name  - Name of output .mat file to save
%   region     - [lon_min, lon_max, lat_min, lat_max] bounding box
%   gt_data    - Path to GT data file (must contain variable 'data' as [lon, lat, label])
%
% OUTPUT:
%   Saves .mat file containing:
%     ddm       - DDM cube (rows × cols × samples)
%     location  - [lon, lat] of each sample
%     index     - Label for each sample (0 = land, 1 = surface water)

function collect_data(base_path,save_name,region,gt_data)

% Read the first file to initialize variables
files = dir(fullfile(base_path, '*.nc'));

% Read variables from the first .nc file
path = fullfile(base_path,files(1).name);  
fprintf('read... %s\n', path);
pra_all = ncread(path,'power_analog');
lon_all = ncread(path,'sp_lon');
lat_all = ncread(path,'sp_lat');
gr_all = ncread(path,'sp_rx_gain');
sp_inc_all = ncread(path,'sp_inc_angle');
snr_all = ncread(path,'ddm_snr');
qf1_all =  ncread(path,'quality_flags');
qf2_all =  ncread(path,'quality_flags_2');

% Loop through remaining files and concatenate data
for i =2:size(files,1)
    path = fullfile(base_path,files(i).name); 
    fprintf('read... %s\n', path);

    pra = ncread(path,'power_analog');
    pra_all = cat(4,pra_all,pra);

    lon = ncread(path,'sp_lon');
    lon_all = cat(2,lon_all,lon);

    lat = ncread(path,'sp_lat');
    lat_all = cat(2,lat_all,lat);

    gr = ncread(path,'sp_rx_gain');
    gr_all = cat(2,gr_all,gr);

    sp_inc = ncread(path,'sp_inc_angle');
    sp_inc_all = cat(2,sp_inc_all,sp_inc);

    snr = ncread(path,'ddm_snr');
    snr_all = cat(2,snr_all,snr);

    qf1 =  ncread(path,'quality_flags');
    qf1_all = cat(2,qf1_all,qf1);
    qf2 =  ncread(path,'quality_flags_2');
    qf2_all = cat(2,qf2_all,qf2);
end

% Crop data by specified region for each channel
for i=1:4
    lon = lon_all(i,:);
    lon(lon > 180) = lon(lon > 180) - 360;
    valid_indices_lon(i,:) = lon >= region(1) & lon <= region(2);
    valid_indices_lat(i,:) = lat_all(i,:) >= region(3) & lat_all(i,:) <= region(4);
    valid_indices(i,:) = valid_indices_lat(i,:) & valid_indices_lon(i,:);
end

pra_channels = pra_all(:, :, 1, valid_indices(1,:));
lon_channels = lon_all(1,valid_indices(1,:));
lat_channels = lat_all(1,valid_indices(1,:));
gr_channels = gr_all(1,valid_indices(1,:), :);
sp_inc_channels = sp_inc_all(1,valid_indices(1,:));
snr_channels = snr_all(1, valid_indices(1,:));
qf1_channels = qf1_all(1, valid_indices(1,:));
qf2_channels = qf2_all(1, valid_indices(1,:));


for i = 2:4
    pra_channels = cat(4,pra_channels,pra_all(:, :, i, valid_indices(i,:)));
    lon_channels = cat(2,lon_channels,lon_all(i,valid_indices(i,:)));
    lat_channels = cat(2,lat_channels,lat_all(i,valid_indices(i,:)));
    gr_channels = cat(2,gr_channels,gr_all(i,valid_indices(i,:)));
    sp_inc_channels = cat(2,sp_inc_channels,sp_inc_all(i,valid_indices(i,:)));
    snr_channels = cat(2,snr_channels,snr_all(i,valid_indices(i,:)));
    qf1_channels = cat(2,qf1_channels,qf1_all(i,valid_indices(i,:)));
    qf2_channels = cat(2,qf2_channels,qf2_all(i,valid_indices(i,:)));

end

pra_channels = squeeze(pra_channels);

pra=pra_channels;
lon=lon_channels';
lat=lat_channels';
gr=gr_channels';
sp_inc=sp_inc_channels';
snr=snr_channels';
qf1 = qf1_channels';
qf2 = qf2_channels';


% Apply quality control filtering
lon(lon(:,1)>180,1) = lon(lon(:,1)>180,1)-360; 
location = [lon lat];

index = true(size(snr));
for i = 1:size(pra,3)
    qf1_temp = dec2bin(qf1(i));
    powers1 = fliplr(log2(2.^(0:numel(qf1_temp)-1)));
    selectedPowers1 = powers1(qf1_temp == '1');
    qf2_temp = dec2bin(qf2(i));
    powers1 = fliplr(log2(2.^(0:numel(qf2_temp)-1)));
    selectedPowers2 = powers1(qf2_temp == '1');

    if sp_inc(i) > 65 ||...
            gr(i) < 0 ||...
            snr(i) < 2 ||...
            any(ismember(selectedPowers1, [1,3,4,7,15,16])) ||...
            any(ismember(selectedPowers2, [0,2,6])) ||...
            any(isnan(pra(:,:,i)),'all') ||...
            any(max(max(pra(:,:,i)))==0) ||...
            all(pra(:,:,i) == 0, 'all')
        index(i) = false; % NaN for elimination
    end
end

pra = pra(:,:,index);
location = location(index,:);

[h,w] = size(flipud(pra(:,:,1)'));
ddm = zeros([h,w,size(pra,3)]);
for i=1:size(pra,3)
    ddm(:,:,i) = flipud(pra(:,:,i)');
end
%% Find nearest point
fprintf('Start finding nearest point... \n');

gt_data = load(gt_data);
data = gt_data.data;

region = [min(location(:,1)) max(location(:,1)) min(location(:,2)) max(location(:,2))];

data = data(find(data(:,2)<=region(4)),:);  %find latitude < max(location(:,2))
data = data(find(data(:,2)>=region(3)),:);  %find latitude > max(location(:,2))
data = data(find(data(:,1)<=region(2)),:);  %find longitude < max(location(:,1))
data = data(find(data(:,1)>=region(1)),:);  %find longitude < min(location(:,1))

fprintf("Region of data: [%f ~ %f ,%f ~ %f] \n",min(location(:,1)),max(location(:,1)),min(location(:,2)),max(location(:,2)))
fprintf("Region of crop: [%f ~ %f ,%f ~ %f] \n",min(data(:,1)),max(data(:,1)),min(data(:,2)),max(data(:,2)))

% Match samples to ground truth labels

fprintf('Start knnsearch... \n');
idx = knnsearch(data(:,1:2),location);
index = data(idx,3);

fprintf("Finish!! \n")

%% Extract land、surface water、ocean
fprintf("Start seperating the data... \n")
land.ddm = ddm(:,:,find(index==0));
land.location = location(find(index==0),:);

surfaceWater.ddm = ddm(:,:,find(index==1));
surfaceWater.location = location(find(index== 1),:);

%% Extract land and surfaceWater

land_num = size(land.ddm,3);
surfaceWater_num = size(surfaceWater.ddm,3);

%different land and water
fprintf("Number of  surface water: %d, Number of land: %d \n",surfaceWater_num,land_num);

ddm = cat(3,land.ddm,surfaceWater.ddm);
location = cat(1,land.location,surfaceWater.location);
index = [zeros(land_num, 1); ones(surfaceWater_num, 1)];

num_samples = size(ddm, 3);
random_indices = randperm(num_samples);

ddm = ddm(:, :, random_indices);
index = index(random_indices, :);
location = location(random_indices,:);

clearvars -except ddm ddm_grad index save_name location

save(save_name)
end