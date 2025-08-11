clc,clear,close all;

% Add function folder to MATLAB search path
addpath("function/");

% Define inference data path
inference_data_path = "./main/collector/demo_ddm";

% Select model name ("IWD_QUEEN" or "IWD_Transformer")
model_name = "IWD_QUEEN"; %IWD_Transformer

% Set corresponding model and output paths based on model selection
switch model_name
    case "IWD_QUEEN"
        inference_model_path = "./main/model/model_IWD_QUEEN.pt";
        inference_save_path = "./main/inference_result/IWD_QUEEN.mat";
    case "IWD_Transformer"
        inference_model_path = "./main/model/model_IWD_Transformer.pt";
        inference_save_path = "./main/inference_result/IWD_Transformer.mat";
end

fprintf("Start inferencing ... \n")

% Build command to call Python and run inference.py with arguments
cmd = sprintf([...
    'conda run -n cygnss python ./main/inference.py ' ...
    '--model_name %s ' ...
    '--model_path "%s" ' ...
    '--data_path "%s" ' ...
    '--save_path "%s" ' ], ...
    model_name, inference_model_path, inference_data_path, inference_save_path);

% Execute system command for inference
system(cmd);

% Convert inference results into binary grid image
binary_img = result2binary(inference_save_path);

% Display results
show_result(inference_save_path,binary_img, model_name);
