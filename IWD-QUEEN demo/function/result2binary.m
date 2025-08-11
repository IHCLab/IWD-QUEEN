function binary_img = result2binary(inference_save_path)

% Load inference result file containing all_location and all_output
data = load(inference_save_path);

% Extract location and output arrays from loaded data
all_location = data.all_location;
all_output = data.all_output;

% Set binary classification threshold
th = 0.5;

% Extract longitude, latitude, and classification values
lon = all_location(:,1);
lat = all_location(:,2);
val = (all_output(1,:) == 1)';

% Determine minimum and maximum longitude and latitude (rounded)
lon_min = round(min(all_location(:,1)));
lon_max = round(max(all_location(:,1)));
lat_min = round(min(all_location(:,2)));
lat_max = round(max(all_location(:,2)));

% Create longitude and latitude grid edges with 0.01° resolution
lon_edges = lon_min:0.01:lon_max;
lat_edges = lat_min:0.01:lat_max;

% Map each point to corresponding grid cell index
lon_idx = discretize(lon, lon_edges);
lat_idx = discretize(lat, lat_edges);

% Aggregate classification values per grid cell using mean
Z = accumarray([lat_idx, lon_idx], val, ...
    [length(lat_edges)-1, length(lon_edges)-1], ...
    @mean, NaN);

% Flip image vertically
img_flipped = flipud(Z);

% Apply threshold to create binary image
binary_img = img_flipped > th;

end