%{
Training code to generate inference models for benchmarking
%}

clear
clc

% Extract csv file names
files = dir('BLISS_training/*.csv');
filenames = string({files.name});  % Convert to string array

% Extract subject codes
codes = extractBetween(filenames(strlength(filenames) >= 5), 3, 5);

% Get unique patterns as string array
uniqueCodes = unique(codes);

% Train on folds
for sub_code = uniqueCodes
    PHRASE_Train("BLISS_training", sub_code, 'W', 10);
end