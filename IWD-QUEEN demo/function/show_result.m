function show_result(inference_save_path,binary_img, model_name)

% Load ground truth (GT) and optical reference images
gt = imread("./main/img/gt.png");
optical = imread("./main/img/optical.png");

% Load inference result data
data = load(inference_save_path);
all_location = data.all_location;
all_output = data.all_output;
time = data.inference_time;

% Extract coordinates of detected positive outputs
output_location = all_location(all_output==1,:);

% Load binary reference map for evaluation
gt_bi = load('./main/reference_binary_map/reference_binary_map.mat');

% Compute quantitative evaluation metrics
[Recall, Precision, F1, OA, Kappa] = quantitative_analysis(gt_bi.binary_img,binary_img);

figure('Position', [100 100 1000 700]),

tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

% Display ground truth (GT) image
nexttile
imshow(gt);
title("            Merit Hydro (GT)",FontName='Times');

% Display geospatial scatter plot of positive detection results
nexttile
geoscatter(output_location(:,2),output_location(:,1),1.5,'red','filled');
geobasemap("none");

ax = gca;            
ax.FontSize = 11;    
ax.FontName = 'Times';

switch model_name
    case "IWD_QUEEN"
        title("Yearly detection result (IWD-QUEEN)",FontName='Times');
    case "IWD_Transformer"
        title("Yearly detection result (IWD-Transformer)",FontName='Times');
end

% Display optical reference image
nexttile
imshow(optical);
title("        SRTM",FontName='Times');

% Display binary detection map
ax4 = nexttile;
imshow(binary_img)
switch model_name
    case "IWD_QUEEN"
        title("Binary grid map (IWD-QUEEN)",FontName='Times');
    case "IWD_Transformer"
        title("Binary grid map (IWD-Transformer)",FontName='Times');
end

% Display performance metrics text below the binary map
metrics_str = sprintf(...
    'Recall: %.2f | Precision: %.2f | F1: %.2f | OA: %.2f | Kappa: %.2f | Time: %.2fs', ...
    Recall, Precision, F1, OA, Kappa, time);
text(ax4, 0.5, -0.05, metrics_str, ...
    'Units', 'normalized', ...  
    'HorizontalAlignment', 'center', ...
    'FontName', 'Times', ...
    'FontSize', 11, ...
    'Color', 'k');

end