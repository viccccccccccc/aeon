% Load the original .mat file
load('psyllid.mat');

% Extract the first 1000 entries from the 'data' matrix
reduced_data = data(1:1000);

% Specify the path where the new .mat file should be saved
save_path = 'C:\Users\Victor\Desktop\bachelorarbet\aeon\aeon\datasets\data\segmentation\psyllid_reduced.mat';

% Save the reduced data (with the name 'data') and the 'name' variable to the specified path
save(save_path, 'reduced_data', 'name');

% Rename the variable 'reduced_data' to 'data' in the saved .mat file
load(save_path, 'reduced_data', 'name');
data = reduced_data;
save(save_path, 'data', 'name');
