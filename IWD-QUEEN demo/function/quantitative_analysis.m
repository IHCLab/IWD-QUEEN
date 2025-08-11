function [Recall, Precision, F1, OA, Kappa] = quantitative_analysis(gt,pred)

% Flatten ground truth (gt) and prediction (pred) into column vectors
gt_flat = gt(:);
pred_flat = pred(:);

% Create mask to exclude NaN values from both gt and pred
mask = ~isnan(gt_flat) & ~isnan(pred_flat);
pred_flat = pred_flat(mask);
gt_flat = gt_flat(mask);

% Calculate true positives (TP), true negatives (TN),
% false positives (FP), and false negatives (FN)
TP = sum(gt_flat == 1 & pred_flat == 1);
TN = sum(gt_flat == 0 & pred_flat == 0);
FP = sum(gt_flat == 0 & pred_flat == 1);
FN = sum(gt_flat == 1 & pred_flat == 0);

% Calculate total number of valid samples
total = TP + TN + FP + FN;

OA = (TP + TN) / total;
Recall = TP / (TP + FN + eps);
Precision = TP / (TP + FP + eps);
F1 = 2 * (Precision * Recall) / (Precision + Recall + eps);
po = OA;
pe = ((TP + FP)*(TP + FN) + (FN + TN)*(FP + TN)) / (total^2);
Kappa = (po - pe) / (1 - pe + eps);

% Round all metrics to two decimal places
OA = round(OA, 2);
Recall = round(Recall, 2);
Precision = round(Precision, 2);
F1 = round(F1, 2);
Kappa = round(Kappa, 2);
end
