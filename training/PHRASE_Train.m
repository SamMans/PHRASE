function [] = PHRASE_Train(ds, unseen_code, activity, seq_sz)
    %{

    Trains PHRASE method for gait phase recognition on an offline dataset. 

    The dataset must contain subject csv files with numerical codes
    in the format (ABXXX), followed by any trial identifier.

    Each CSV file must contain the following headers: 
        - Right_Shank_Ax (Ax: accelerometer in x-direction)
        - Right_Shank_Az (Az: accelerometer in z-direction)
        - Right_Shank_Gy (Gy: Gyroscope about y-axis)
        - Left_Shank_Ax  (Ax: accelerometer in x-direction)
        - Left_Shank_Az  (Az: accelerometer in z-direction)
        - Left_Shank_Gy  (Gy: Gyroscope about y-axis)
        - Mode           (Gait activity)
        - phase          (Walking gait phase)

    (Axis conventions are shown in the paper referenced below)

    The "Mode" header represents the actual activity: 
    - 'W': walking
    - 'RA': ramp ascent
    - 'RD': ramp descent 
    - 'SA': stair ascent
    - 'SD': stair descent 

    The phase header respresents the actual gait phase:
    * Walking: 
        - 0>>LR
        - 1>>MST
        - 2>>TS
        - 3>>PSW
        - 4>>ISW
        - 5>>MSW
        - 6>>TSW

    * Stair/ramp ascent: 
        - 0>>WA
        - 1>>PU
        - 2>>FCO
        - 3>>FCL
        - 4>>FP 

    * Stair/ramp descent: 
        - 0>>WA
        - 1>>FCO
        - 2>>CL
        - 3>>LP
        - 4>>FP

    Each CSV file must have a sufficient number of points for one gait bout
    of at least 1.5 gait cycles (preferrably more). 

    The explanation of header data and how to collect them can be found in:
    S. A. Mohamed and U. Martinez-Hernandez, "Wearable Interface for 
    Real-time Gait Phase Recognition using Sensor Networks," 
    Applied Soft Computing, 2026.
    
    Parameters
    ----------
    ds : string
        Dataset folder name
    seq_sz : int
        Redundant parameter for comparison against benchmarks
    act : string
        Activity of interest 
            - 'W': walking
            - 'RA': ramp ascent 
            - 'RD': ramp descent
            - 'SA': stair ascent
            - 'SD': stair descent)
    unseen_code : string
        Code number of unseen subject
    activity : string
        Activity of interest
    
    Returns
    -------
    N/A

    %}

    % Segment the dataset
    warning('off', 'signal:findpeaks:largeMinPeakHeight');
    modalities = ["Right_Shank_Ax","Right_Shank_Az","Right_Shank_Gy",...
        "Left_Shank_Ax","Left_Shank_Az","Left_Shank_Gy"]; %--> Sensor modalities
    disp('Segmenting dataset...')
    jsonText = fileread(ds + "/metadata.json"); metadata = jsondecode(jsonText);
    fsample = metadata.signal_parameters.sampling_frequency; % Training dataset's sampling frequency
    win_size = round(fsample*0.03);     % Window size is a function of sampling frequency
    multiplier = 7;                     % A window multiplier used for heuristic detection of some transitions
    if ~exist(ds + "/" + unseen_code + ".mat", 'file')
        % Segmenting the dataset into train, validation and test bouts
        dataseg = DS_Segment(ds, unseen_code, activity);
        
        % Extract raw prior biomechanical anchored features from the subsets
        disp('Extracting raw features from training set (60% of seen subject data)...')
        [F.train] = Prior_Raw_Extract(dataseg.train, fsample, win_size, multiplier, activity, true);
        disp('Extracting raw features from validation set (40% of seen subject data)...')
        F.val = Prior_Raw_Extract(dataseg.val, fsample, win_size, multiplier, activity, false);
        disp('Extracting raw features from test set (useen subject data)...')
        F.test = Prior_Raw_Extract(dataseg.test, fsample, win_size, multiplier, activity, false);

        % Save data segments and extracted features to the dataset directory
        save(ds + "/" + unseen_code + ".mat", 'F', 'dataseg');
    else
        % Load data segments if extraction was executed before from the dataset directory
        load(ds + "/" + unseen_code + ".mat", 'F', 'dataseg');
    end
    
    % Extract gaussian multivariate distribution parameters from training subset prior features
    D = Prior_Train(F.train);
    
    % Evaluate prior gaussian knowledge performance on anchored features 
    sets = fieldnames(F);   % Set names (train, val, test sets)
    for st = 1:numel(sets)
        % Looping over sets
        disp(sets{st}+" set prior knowledge contribution per transition: ")
        TP = 0; FP = 0; TN = 0; FN = 0; % Overall true/false positives and negatives for the set
        non_anchor = string(fieldnames(D)); % Non-anchor phases (anchored by anchor phases)
        for na = 1:numel(non_anchor)
            % Event priors (50/50 chance if prior anchored features not given)
            prob_true = 0.5; prob_false = 0.5; % P(T) and P(F)

            % Prior anchored features with 'true' ground truth label
            true_features = F.(sets{st}).(non_anchor(na)).True;
            X_true = Prior_Preprocess(true_features, D.(non_anchor(na)).mu_raw, D.(non_anchor(na)).sigma_raw, ...
                D.(non_anchor(na)).proj, D.(non_anchor(na)).num_components); % Normalized ground true features

            % Likelihoods given true and false conditions for true ground prior features
            likelihood_true_true = diagonal_mvnpdf_vectorized(X_true, D.(non_anchor(na)).true.mu, ...
                D.(non_anchor(na)).true.cov); % P(X_true|T)
            likelihood_true_false = diagonal_mvnpdf_vectorized(X_true, D.(non_anchor(na)).false.mu, ...
                D.(non_anchor(na)).false.cov); % P(X_true|F)
            Evidence = likelihood_true_true * prob_true + ...
                likelihood_true_false * prob_false; % P(X_true)

            % Probabilities for ground truth positive
            prob_true_true = (likelihood_true_true * prob_true) ./ ...
                Evidence; % P(T|X_true)
            prob_true_false = (likelihood_true_false * prob_false) ./ ...
                Evidence; % P(F|X_true)

            % Prior anchored features with false ground truth label
            if F.(sets{st}).(non_anchor(na)).False
                % Features with false ground truth label
                false_features = F.(sets{st}).(non_anchor(na)).False;
                X_false = Prior_Preprocess(false_features, ...
                    D.(non_anchor(na)).mu_raw, D.(non_anchor(na)).sigma_raw, ...
                    D.(non_anchor(na)).proj, D.(non_anchor(na)).num_components); % Normalized ground false features
                
                % Likelihood given true and false transitions
                likelihood_false_false = diagonal_mvnpdf_vectorized(X_false, D.(non_anchor(na)).false.mu, ...
                    D.(non_anchor(na)).false.cov); % P(X_false|F)
                likelihood_false_true = diagonal_mvnpdf_vectorized(X_false, D.(non_anchor(na)).true.mu, ...
                    D.(non_anchor(na)).true.cov); % P(X_false|T)
                Evidence = likelihood_false_false * prob_false + ...
                    likelihood_false_true * prob_true; % P(X_false)
                
                % Probabilities
                prob_false_false = (likelihood_false_false * prob_false) ./ ...
                    Evidence; % P(F|X_false)
                prob_false_true = (likelihood_false_true * prob_true) ./ ...
                    Evidence; % P(T|X_false)
                
                % Performance metrics per phase
                true_positives_idx = prob_true_true>prob_true_false;
                false_positives_idx = prob_true_true<=prob_true_false;
                true_negatives_idx = prob_false_false>prob_false_true;
                false_negatives_idx = prob_false_false<=prob_false_true;
                acc = (sum(true_positives_idx) + ...
                    sum(true_negatives_idx)) / ...
                    (size(X_true, 1)+size(X_false, 1)) * 100; % accuracy for current phase
                TP = TP + sum(true_positives_idx); FP = FP + sum(false_positives_idx); % Increment overall dataset counts
                TN = TN + sum(true_negatives_idx); FN = FN + sum(false_negatives_idx); % Increment overall dataset counts

                % Performance visuals
                %{
                figure
                if any(true_positives_idx)
                    plot(0:size(true_features(true_positives_idx, :),2)-1, ...
                        true_features(true_positives_idx, :))
                end
                title(sets{st} + " " + non_anchor(na) + " True Positives")
                figure
                if any(false_positives_idx)
                    plot(0:size(true_features(false_positives_idx, :),2)-1, ...
                        true_features(false_positives_idx, :))
                end
                title(sets{st} + " " + non_anchor(na) + " False Positives")
                figure
                if any(true_negatives_idx)
                    plot(0:size(false_features(true_negatives_idx, :),2)-1, ...
                        false_features(true_negatives_idx, :))
                end
                title(sets{st} + " " + non_anchor(na) + " True Negatives")
                figure
                if any(false_negatives_idx)
                    plot(0:size(false_features(false_negatives_idx, :),2)-1, ...
                        false_features(false_negatives_idx, :))
                end
                title(sets{st} + " " + non_anchor(na) + " False Negatives")
                %}
            else
                % Performance metrics per phase (if no false features)
                true_positives_idx = prob_true_true>prob_true_false; % Increment overall dataset counts
                false_positives_idx = prob_true_true<=prob_true_false; % Increment overall dataset counts
                TP = TP + sum(true_positives_idx); FP = FP + sum(false_positives_idx);
                acc = sum(true_positives_idx) / ...
                    size(X_true, 1) * 100; % accuracy for current phase

                % Performance visuals
                %{
                figure
                if any(true_positives_idx)
                    plot(0:size(true_features(true_positives_idx, :),2)-1, ...
                        true_features(true_positives_idx, :))
                end
                title(sets{st} + " " + non_anchor(na) + " True Positives")
                figure
                if any(false_positives_idx)
                    plot(0:size(true_features(false_positives_idx, :),2)-1, ...
                        true_features(false_positives_idx, :))
                end
                title(sets{st} + " " + non_anchor(na) + " False Positives")
                %}
            end
            disp("  Prior accuracy for "+non_anchor(na)+": "+num2str(acc)+"%")
        end
        % Overall performance metrics for the current dataset split
        disp("  Overall Prior accuracy: "+num2str((TP+TN)/(TP+TN+FP+FN)*100)+"%")
        disp("  Overall Recall/Sensitivity: "+num2str(TP/(TP+FN)))
        disp("  Overall Precision: "+num2str(TP/(TP+FP)))
        disp("  Overall Specificity: "+num2str(TN/(TN+FP)))
    end
    
    % Extract post-processed time-domain feature window dataset for phase NN
    [PNN_dsTrain] = NN_Pprocess(dataseg.train,win_size); % Training NN dataset
    [PNN_dsCv] = NN_Pprocess(dataseg.val,win_size); % Val. NN dataset
    [PNN_dsTest] = NN_Pprocess(dataseg.test,win_size); % Test NN dataset
    
    % Train the phase NN
    W = PNN_train(PNN_dsTrain,PNN_dsCv,PNN_dsTest);
   
    % Run analystics on NN
    if(activity=="W")
        phases = ["LR","MST","TS","PSW","ISW","MSW","TSW"]; %--> Walking phases
    elseif(activity=="RA"||activity=="SA")
        phases = ["WAA","PUA","FCA","FClA","FPA"]; %--> Ascent phases
    else
        phases = ["WAD","FCD","CLD","LPTD","FPD"]; %--> Descent phases
    end
    %--> Prediction for training set
    p = Predict(cell2mat({PNN_dsTrain(:).X}.'), W); 
    figure
    cm = confusionchart(phases(cell2mat({PNN_dsTrain(:).y}.')), phases(p));
    cm.Title = 'ANN training confusion matrix'; set(gcf,'color','w'); %--> Confusion matrix plot
    %--> Prediction for validation set
    pcv = Predict(cell2mat({PNN_dsCv(:).X}.'), W); 
    figure
    cm = confusionchart(phases(cell2mat({PNN_dsCv(:).y}.')), phases(pcv));
    cm.Title = 'ANN validation confusion matrix'; set(gcf,'color','w'); %--> Confusion matrix plot
    cm = confusionmat(cell2mat({PNN_dsCv(:).y}.'), pcv); cmt = cm.'; diagonal = diag(cmt); row_sum = sum(cmt, 2);
    col_sum = sum(cmt,1);
    precision = diagonal./row_sum; recall = diagonal./col_sum';
    specificity = (sum(cmt(:)) - sum(cmt,2) - sum(cmt,1)' + diag(cmt)) ./ ...
                   (sum(cmt(:)) - sum(cmt,1)');
    F1_cv = 2*((precision.*recall)./(precision+recall));
    disp("ANN validation F1 score: "); %--> Validation F1 score 
    disp(phases)
    disp(F1_cv.');
    disp("ANN validation precision: "); %--> Validation precision score 
    disp(phases)
    disp(precision.');
    disp("ANN validation recall: "); %--> Validation recall score 
    disp(phases)
    disp(recall.');
    disp("ANN validation specificity: "); %--> Validation specificity score 
    disp(phases)
    disp(specificity.');
    %--> Prediction for test set
    ptest = Predict(cell2mat({PNN_dsTest(:).X}.'), W); 
    figure
    cm = confusionchart(phases(cell2mat({PNN_dsTest(:).y}.')),phases(ptest));
    cm.Title = 'ANN testing confusion matrix'; set(gcf,'color','w'); %--> Confusion matrix plot
    cm = confusionmat(cell2mat({PNN_dsTest(:).y}.'),ptest); cmt = cm.'; diagonal = diag(cmt); row_sum = sum(cmt,2);
    col_sum = sum(cmt,1);
    precision = diagonal./row_sum; recall = diagonal./col_sum';
    specificity = (sum(cmt(:)) - sum(cmt,2) - sum(cmt,1)' + diag(cmt)) ./ ...
                   (sum(cmt(:)) - sum(cmt,1)');
    F1_test = 2*((precision.*recall)./(precision+recall));
    disp("ANN testing F1 score: "); %--> Testing F1 score 
    disp(phases)
    disp(F1_test.');
    disp("ANN testing precision: "); %--> Testing precision score 
    disp(phases)
    disp(precision.');
    disp("ANN testing recall: "); %--> Testing recall score 
    disp(phases)
    disp(recall.');
    disp("ANN testing specificity: "); %--> Testing specificity score 
    disp(phases)
    disp(specificity.');
    
    % Save training parameters
    % Specify the variables you want to load from the .mat file
    variablesToSave = {'W', 'D', 'fsample', 'win_size', 'multiplier', 'seq_sz', 'phases', 'modalities'}; 
    
    % Create a structure to hold the variables
    data = struct();
    for i = 1:length(variablesToSave)
        varName = variablesToSave{i};
        data.(varName) = eval(varName);
    end
    
    % Convert the structure to a JSON string
    jsonString = jsonencode(data);
    
    % Specify the output JSON file
    jsonFile = string(fileparts(pwd)) + "\checkpoints" + "\phrase_model_" + activity + "_" + ...
        unseen_code + "_" + string(fsample) + "Hz.json"; 
    
    % Write the JSON string to a file
    fid = fopen(jsonFile, 'w');
    if fid == -1
        error('Cannot create JSON file');
    end
    fprintf(fid, '%s', jsonString);
    fclose(fid);
    
    disp(['Selected variables have been saved to ' jsonFile]);
end

function [dataseg] = DS_Segment(ds_name, unseen_sub, act)
    %{ 
    processes raw dataset to extract bouts of a particular activity and
    divide them among training, validation and testing sets.
    
    Parameters
    ----------
    ds_name : string
        Dataset name
    unseen_sub : string
        Unseen subject code
    act : string
        Activity of interest ('W': walking, 'RA': ramp ascent, 
        'RD': ramp descent, 'SA': stair ascent, 'SD': stair descent)
    
    Returns
    -------
    dataseg : struct array
        Sensor readings and labels for segmented activity bouts for 
        three subsets: training, validation and testing
    %}
    % Definition of activities and modalities
    activities = ["W","RA","RD","SA","SD"]; % Activity strings (walking, ramp ascent, .., etc.)
    modalities = ["Right_Shank_Ax","Right_Shank_Az","Right_Shank_Gy",...
        "Left_Shank_Ax","Left_Shank_Az","Left_Shank_Gy"]; %--> Sensor modalities
    
    % Dataset segmentation into train, validation and test sets
    csvFiles = string({dir(ds_name + "/*.csv").name}); % dataset csv files
    csv.test = csvFiles(contains(csvFiles, unseen_sub)); % unseen test circuits
    seen_ct = csvFiles(~contains(csvFiles, unseen_sub)); % seen circuits
    [~, ~, unique_seen_sub_inst] = unique(arrayfun(@(x) extractBefore(x, 6), ...
        seen_ct)); % repeated instances of unique seen subjects
    unique_sub_bounds = [0; find(diff(unique_seen_sub_inst)); length(unique_seen_sub_inst)]; % index bounds between seen subjets
    csv.train = strings(1, sum(floor(diff(unique_sub_bounds)*0.6))); % 0.6 of seen data
    for bound = 1:length(unique_sub_bounds)-1
        train_range = floor(unique_sub_bounds(bound)*0.6)+1:floor(unique_sub_bounds(bound+1)*0.6); 
        seen_range = unique_sub_bounds(bound)+1:unique_sub_bounds(bound)+length(train_range);
        csv.train(train_range) = seen_ct(seen_range); % 0.6 of each subject's data
    end
    csv.val = setdiff(seen_ct, csv.train); % 0.4 of seen subject data
    
    % Initialize empty subsets
    dataseg.train = []; % (X: readings, y: labels)
    dataseg.val = []; dataseg.test = [];
    subsets = ["train", "val", "test"];

    % Segment the dataset
    for set = 1:3
        for i = 1:size(csv.(subsets(set)),2)
            % Import circuit data from CSV files
            ct_full = readtable(pwd+"\"+ds_name+"\"+csv.(subsets(set))(i)); % Full circuit data
            act_idx = find(ct_full.Mode==find(activities==act)); % Indices of activity of interest
            act_trans_idx = find(diff(act_idx)~=1); % Activity transition indices
            bout_range = [act_idx([1;act_trans_idx+1]),...
                act_idx([act_trans_idx;size(act_idx,1)])]; % Ranges of bout instances
            % Extract segmented bouts
            for bout = 1:size(bout_range,1)
                % Get relevant data points with usable modality channels
                trial_modalities = table2array(ct_full(:,modalities));
                new_bout.X = ...
                    trial_modalities(bout_range(bout,1):bout_range(bout,2),:); 
                trial_labels = ct_full.phase;
                new_bout.y = ...
                    trial_labels(bout_range(bout,1):bout_range(bout,2))+1; 
                        % Increment y to make classes start from 1 instead of 0 
                dataseg.(subsets(set)) = [dataseg.(subsets(set)); new_bout];
            end
        end
    end
end

function [Feat] = Prior_Raw_Extract(ds, fs, win_sz, mul, activ, synthesize)
    %{ 
    return the raw prior features.
    
    Parameters
    ----------
    ds : struct array
        Segmented gait bouts of the dataset
    fs : double
        Sampling frequency
    win_sz : uint8
        Window size
    mul : uint8
        Prominence multiplier
    activ : string
        Activity of interest
    synthesize : bool
        Synthesize data for missing/scarce subsets?
    
    Returns
    -------
    Feat : struct
        Raw prior features
    %}
    % Initialize raw prior features
    if activ == 'W'
        phases = ["Rnp_zc", "Lpos", "Lneg", "Lnp_zc", "Rpos", "Rpn_zc", "Rneg"]; % Gait phases (encoded)
        Feat.Lpos.True = []; Feat.Lneg.True = []; Feat.Rpos.True = []; Feat.Rpn_zc.True = []; 
        Feat.Rneg.True = [];
        Feat.Lpos.False = []; Feat.Lneg.False = []; Feat.Rpos.False = []; Feat.Rpn_zc.False = []; 
        Feat.Rneg.False = [];
    else
        phases = ["Rnp_zc", "Lpos", "Lnp_zc", "Rpn_zc", "Rneg"];
        Feat.Lpos.True = []; Feat.Rpn_zc.True = []; 
        Feat.Rneg.True = [];
        Feat.Lpos.False = []; Feat.Rpn_zc.False = []; 
        Feat.Rneg.False = [];
    end
    
    % Loop over csv files and extract raw features
    fc = 10; % Cutoff frequency
    [b, a] = butter(4, fc/(fs/2), 'low'); % Fourth-order butterworth filter
    prom_win_sz = mul*win_sz; % Size of window of prominence for peaks
    acceptance_multiplier = 1; % Tolerance of error in terms of window multiplier
    for bout = 1:length(ds)
        % Process the bout signal window-by-window
        anchor_idx = 1; % Initial anchor index
        cand.Rnp_zc = []; cand.Lnp_zc = []; % Anchor candidates initially empty
        for terminal_idx = win_sz:win_sz:length(ds(bout).y)
            % Filter the gyro signal using a fourth order butterworth filter
            filtered_right = filtfilt(b, a, ds(bout).X(anchor_idx:terminal_idx, 3));
            filtered_left = filtfilt(b, a, ds(bout).X(anchor_idx:terminal_idx, 6));
            gt_phase = ds(bout).y(anchor_idx:terminal_idx); % Ground truth phase record
    
            % Find heuristic candidates
            [~, Rneg_locs] = findpeaks(-filtered_right, 'MinPeakHeight', 1.5, 'MinPeakDistance', min([terminal_idx-anchor_idx-1, prom_win_sz]));
            [~, Rpos_locs] = findpeaks(filtered_right, 'MinPeakHeight', 0.5, 'MinPeakDistance', min([terminal_idx-anchor_idx-1, prom_win_sz]));
            [~, Lneg_locs] = findpeaks(-filtered_left, 'MinPeakHeight', 1.5, 'MinPeakDistance', min([terminal_idx-anchor_idx-1, prom_win_sz]));
            np_zc_locs = find(diff(sign(filtered_right)) > 0);
            if ~isempty(Rneg_locs) && any(np_zc_locs>max(Rneg_locs)) && ...
                    length(np_zc_locs(np_zc_locs>max(Rneg_locs)))==1
                % A new right anchor event has been detected
                if isempty(cand.Lnp_zc) && isempty(cand.Rnp_zc)  
                    % No anchors have been detected before
                    shift = ceil(np_zc_locs(end)/win_sz)*win_sz-win_sz; % Temporal shift
                    anchor_idx = shift+anchor_idx; % New right zero crossing window becomes anchor
                    cand.Rnp_zc = np_zc_locs(end)-shift; % Displace right anchor location
                elseif ~isempty(cand.Lnp_zc) 
                    % Old left anchor exists
                    if ~isempty(cand.Rnp_zc) && abs(np_zc_locs(np_zc_locs>max(Rneg_locs))-cand.Rnp_zc)>win_sz
                        % if an old right anchor exists, compute right anchor features
                        if activ == 'W'
                            % right positive peak candidates
                            true_bound = reshape(find([0;diff(gt_phase)]~=0 & gt_phase==find(phases=='Rpos')) + ...
                                (-acceptance_multiplier*win_sz:acceptance_multiplier*win_sz), [], 1); % Indices of acceptable true locations
                            for c = 1:length(Rpos_locs)
                                %raw_feat_right = Temp_Normalize(filtered_right(cand.Rnp_zc:Rpos_locs(c)), win_sz);
                                raw_feat_left = Temp_Normalize(filtered_left(cand.Rnp_zc:Rpos_locs(c)), win_sz);
                                if any(Rpos_locs(c)==true_bound)
                                    Feat.Rpos.True = [Feat.Rpos.True; raw_feat_left];
                                else
                                    Feat.Rpos.False = [Feat.Rpos.False; raw_feat_left];
                                end
                            end
                        end
                        if activ == 'W'
                            % left negative peak candidates
                            true_bound = reshape(find([0;diff(gt_phase)]~=0 & gt_phase==find(phases=='Lneg')) + ...
                                (-acceptance_multiplier*win_sz:acceptance_multiplier*win_sz), [], 1); % Indices of acceptable true locations
                            for c = 1:length(Lneg_locs)
                                %raw_feat_right = Temp_Normalize(filtered_right(cand.Rnp_zc:Lneg_locs(c)), win_sz);
                                raw_feat_left = Temp_Normalize(filtered_left(cand.Rnp_zc:Lneg_locs(c)), win_sz);
                                if any(Lneg_locs(c)==true_bound)
                                    Feat.Lneg.True = [Feat.Lneg.True; raw_feat_left];
                                else
                                    Feat.Lneg.False = [Feat.Lneg.False; raw_feat_left];
                                end
                            end
                        end
                    end
                    shift = ceil(cand.Lnp_zc/win_sz)*win_sz-win_sz;
                    anchor_idx = shift+anchor_idx; % Last left zero crossing window becomes anchor
                    cand.Rnp_zc = np_zc_locs(end)-shift; % New right zero crossing location
                    cand.Lnp_zc = cand.Lnp_zc-shift; % Displace left anchor location   
                end
    
                % Update filtered signal and ground truth range
                if shift~=0
                    filtered_right = filtered_right(shift+1:end);
                    filtered_left = filtered_left(shift+1:end);
                    gt_phase = gt_phase(shift+1:end);
                end
            end
            [~, Lneg_locs] = findpeaks(-filtered_left, 'MinPeakHeight', 1.5, 'MinPeakDistance', min([terminal_idx-anchor_idx-1, prom_win_sz]));
            [~, Lpos_locs] = findpeaks(filtered_left, 'MinPeakHeight', 0.5, 'MinPeakDistance', min([terminal_idx-anchor_idx-1, prom_win_sz]));
            np_zc_locs = find(diff(sign(filtered_left)) > 0);
            pn_zc_locs = find(diff(sign(filtered_right)) < 0);
            if ~isempty(Lneg_locs) && any(np_zc_locs > max(Lneg_locs)) && ...
                    length(np_zc_locs(np_zc_locs>max(Lneg_locs)))==1 % check if faulty triggers occur
                % A new left anchor event has been detected
                if isempty(cand.Rnp_zc) && isempty(cand.Lnp_zc)  
                    % No anchors have been detected before
                    shift = ceil(np_zc_locs(end)/win_sz)*win_sz-win_sz; % Temporal shift
                    anchor_idx = shift+anchor_idx; % New left zero crossing window becomes anchor
                    cand.Lnp_zc = np_zc_locs(end)-shift; % Displace left anchor location
                elseif ~isempty(cand.Rnp_zc) 
                    % Old right anchor exists
                    if ~isempty(cand.Lnp_zc) && abs(np_zc_locs(np_zc_locs>max(Lneg_locs))-cand.Lnp_zc)>win_sz
                        % if an old left anchor exists, compute left anchor features
                        % left positive peak candidates
                        true_bound = reshape(find([0;diff(gt_phase)]~=0 & gt_phase==find(phases=='Lpos')) + ...
                            (-acceptance_multiplier*win_sz:acceptance_multiplier*win_sz), [], 1); % Indices of acceptable true locations
                        for c = 1:length(Lpos_locs)
                            raw_feat_right = Temp_Normalize(filtered_right(cand.Lnp_zc:Lpos_locs(c)), win_sz);
                            %raw_feat_left = Temp_Normalize(filtered_left(cand.Lnp_zc:Lpos_locs(c)), win_sz);
                            if any(Lpos_locs(c)==true_bound)
                                Feat.Lpos.True = [Feat.Lpos.True; raw_feat_right];
                            else
                                Feat.Lpos.False = [Feat.Lpos.False; raw_feat_right];
                            end
                        end 
                        % right positive to negative zero crossing candidates
                        true_bound = reshape(find([0;diff(gt_phase)]~=0 & gt_phase==find(phases=='Rpn_zc')) + ...
                            (-acceptance_multiplier*win_sz:acceptance_multiplier*win_sz), [], 1); % Indices of acceptable true locations
                        for c = 1:length(pn_zc_locs)
                            raw_feat_right = Temp_Normalize(filtered_right(cand.Lnp_zc:pn_zc_locs(c)), win_sz);
                            %raw_feat_left = Temp_Normalize(filtered_left(cand.Lnp_zc:pn_zc_locs(c)), win_sz);
                            if any(pn_zc_locs(c)==true_bound)
                                Feat.Rpn_zc.True = [Feat.Rpn_zc.True; raw_feat_right];
                            else
                                Feat.Rpn_zc.False = [Feat.Rpn_zc.False; raw_feat_right];
                            end
                        end
                        % right negative peak candidates
                        true_bound = reshape(find([0;diff(gt_phase)]~=0 & gt_phase==find(phases=='Rneg')) + ...
                            (-acceptance_multiplier*win_sz:acceptance_multiplier*win_sz), [], 1); % Indices of acceptable true locations
                        for c = 1:length(Rneg_locs)
                            raw_feat_right = Temp_Normalize(filtered_right(cand.Lnp_zc:Rneg_locs(c)), win_sz);
                            %raw_feat_left = Temp_Normalize(filtered_left(cand.Lnp_zc:Rneg_locs(c)), win_sz);
                            if any(Rneg_locs(c)==true_bound)
                                Feat.Rneg.True = [Feat.Rneg.True; raw_feat_right];
                            else
                                Feat.Rneg.False = [Feat.Rneg.False; raw_feat_right];
                            end
                        end 
                    end
                    shift = ceil(cand.Rnp_zc/win_sz)*win_sz-win_sz;
                    anchor_idx = shift+anchor_idx; % Last right zero crossing window becomes anchor
                    cand.Lnp_zc = np_zc_locs(end)-shift; % New left anchor location
                    cand.Rnp_zc = cand.Rnp_zc-shift; % Displace right anchor location   
                end
    
                % Update filtered signal and ground truth range
                %if shift~=0
                %    filtered_right = filtered_right(shift+1:end);
                %    filtered_left = filtered_left(shift+1:end);
                %end
            end
    
            %plot(1:length(filtered_right), filtered_right)
            %hold on
            %plot(1:length(filtered_left), filtered_left)
            %plot(cand.LR, filtered_right(cand.LR), '*r')
            %plot(cand.PSW, filtered_left(cand.PSW), '*g')
            %hold off
        end
    end
    if activ == "W"
        Feat.MST = Feat.Lpos; Feat.TS = Feat.Lneg;
        Feat.ISW = Feat.Rpos; Feat.MSW = Feat.Rpn_zc;
        Feat.TSW = Feat.Rneg;
        Feat = rmfield(Feat, {'Lpos', 'Lneg', 'Rpos', ...
            'Rpn_zc', 'Rneg'}); % Remove the old fields
    elseif activ == "SA" || active == "RA"
        Feat.PU = Feat.Lpos; Feat.FCL = Feat.Rpn_zc;
        Feat.FP = Feat.Rneg;
        Feat = rmfield(Feat, {'Lpos', ...
            'Rpn_zc', 'Rneg'}); % Remove the old fields
    else
        Feat.FCO = Feat.Lpos; Feat.LP = Feat.Rpn_zc;
        Feat.FP = Feat.Rneg;
        Feat = rmfield(Feat, {'Lpos', ...
            'Rpn_zc', 'Rneg'}); % Remove the old fields
    end

    % Fill missing fields with artificial data from the same distribution
    if synthesize
        fields = string(fieldnames(Feat));
        for idx = 1:length(fields)
            % Fill missing data
            if size(Feat.(fields(idx)).False, 1)==0
                % Assign arbitrary prior raw features to empty false non-anchor triggers
                Feat.(fields(idx)).False = rand(100, size(Feat.(fields(idx)).True, 2));
            elseif size(Feat.(fields(idx)).False, 1)>=1 && ...
                    size(Feat.(fields(idx)).False, 1)<50
                % Assign noisy linear combinations of existing raw features
                Feat.(fields(idx)).False = augment_features(Feat.(fields(idx)).False, ...
                    size(Feat.(fields(idx)).True, 1), 0.005);
            end
        end
    end
end

function X_aug = augment_features(X, n_new_features, noise_std)
    % AUGMENT_FEATURES - Create synthetic features via linear combinations + noise
    %
    % Inputs:
    %   X           - Original features [n_samples × n_features]
    %   n_new_features - Number of synthetic features to create
    %   noise_std   - Standard deviation of Gaussian noise (default: 0.1)
    %
    % Output:
    %   X_aug       - Augmented features [n_samples × (n_features + n_new_features)]
    
    if nargin < 3
        noise_std = 0.1;  % Default noise level
    end
    
    [n_features, n_samples] = size(X);
    
    % 1. Random normalized weights for linear combinations
    weights = abs(randn(n_new_features, n_features));
    weights = weights ./ sum(weights, 2);
    
    % 2. Create synthetic features: linear combination + random shift + Gaussian noise
    synthetic = weights * X + 2*(rand(n_new_features, 1)-0.5) + ...
        noise_std * randn(n_new_features, n_samples);
    
    % 3. Combine with original features
    X_aug = [X; synthetic];

end

function [norm_signal] = Temp_Normalize(signal, w)
    %{ 
    return the temporally normalized signal.
    
    Parameters
    ----------
    signal : array
        Array of unnormalized raw features
    w : uint8
        Window size
    
    Returns
    -------
    norm_signal : array
        Array of normalized raw features
    %}
    normalized_time = linspace(0, 99, 100); % Normalized time vector (100 points)
    t = 1:length(signal); % Origial timing
    original_normalized_time = 100 * (t - min(t)) / (max(t) - min(t)); % Normalizing time
    if length(signal) >= w
        norm_signal = interp1(original_normalized_time, signal, normalized_time); % Normalizing signal
    end
end

function [proc_feat] = Prior_Preprocess(raw_feat, mu_raw, sigma_raw, proj, num_components)
    %{ 
    return the processed principal component features.
    
    Parameters
    ----------
    raw_feat : array
        Array of raw features
    mu_raw : double
        Mean of raw features
    sigma_raw : double
        Standard deviation of raw features
    proj : array
        Projection matrix
    num_components : uint8
        Number of necessary/effective principal components
    
    Returns
    -------
    proc_feat : array
        Array of processed features
    %}
    proc_feat = ((raw_feat-mu_raw)./sigma_raw)*proj;
    proc_feat = proc_feat(:, 1:num_components);
end

function [distro] = Prior_Train(feat)
    %{ 
    compute prior gaussian distribution parameters of phase transitions.
    
    Parameters
    ----------
    feat : struct
        Raw features of true & false transitions per phase
        (time-normalized filtered gyro readings between anchor transition 
        and non-anchor transition)
    
    Returns
    -------
    distro : struct
        Distribution parameters
    %}
    disp("Extracting distribution parameters from training dataset...")
    phases = fieldnames(feat);
    % Loop over the non-anchor phases
    for idx = 1:numel(phases)
        % Standardize the data
        [standardized_data, mu_raw, sigma_raw] = zscore([feat.(phases{idx}).True; ...
            feat.(phases{idx}).False]);
        
        % Apply PCA to the standardized training data
        [coeff, ~, ~, ~, explained] = pca(standardized_data);
        
        % Decide on the number of principal components to retain 
        % (based on explained variance)
        num_components = NumPCA(explained, 99);
        true_feat_proc = Prior_Preprocess(feat.(phases{idx}).True, mu_raw, ...
            sigma_raw, coeff, num_components);
        false_feat_proc = Prior_Preprocess(feat.(phases{idx}).False, mu_raw, ...
            sigma_raw, coeff, num_components);
    
        % Compute the distribution parameters (mean and covariance of processed features)
        distro.(phases{idx}).true.mu = mean(true_feat_proc, 1); % distribution mean
        distro.(phases{idx}).false.mu = mean(false_feat_proc, 1); 
        distro.(phases{idx}).true.cov = diag(std(true_feat_proc, 0, 1)); % distribution covariance
        distro.(phases{idx}).false.cov = diag(std(false_feat_proc, 0, 1)); 
    
        % Add raw feature preprocessing parameters to the model
        distro.(phases{idx}).mu_raw = mu_raw; % standardization mean
        distro.(phases{idx}).sigma_raw = sigma_raw; % standardization std deviation
        distro.(phases{idx}).proj = coeff; % projection matrix
        distro.(phases{idx}).num_components = num_components; % number of principal components
    end
end

function num = NumPCA(explained, percent)
    %{ 
    return the number of principal components necessary to maintain >=percent% variance.
    
    Parameters
    ----------
    explained : array
        Array of explained variances from PCA
    percent : float
        PCA desired variance percentage
    
    Returns
    -------
    num : uint8
        Number of necessary principle components
    %}
    % Initialize the cumulative sum
    cumulative_sum = 0;
    
    % Loop through the array
    for num = 1:length(explained)
        % Add the current element to the cumulative sum
        cumulative_sum = cumulative_sum + explained(num);
        
        % Check if the cumulative sum is at least 'percent'
        if cumulative_sum >= percent
            break; % Exit the loop once the sum reaches at least 99
        end
    end
end

function [ds_processed,features] = NN_Pprocess(ds_unprocessed,w)
    %{ 
    processes phase dataset to extract useful features and segmented 
    labels for neural network training.
    
    Parameters
    ----------
    ds_unprocessed : struct array
        A structure array containing relevant bio-sensor readings and labels for
        separate activity datasets.
    w : int
        Segmentation window size.
    
    Returns
    -------
    ds_processed : struct array
        A structure array containing segmented dataset with features and labels.
    features : string array
        Preprocessing features/input to NN.
    %}
    features = ["mean","median","std","min","max","initial",...
        "final","MAV","WL"]; %--> Feature selection
    % Create the processed dataset
    ds_processed(size(ds_unprocessed)) = ...
        struct('X',[],'y',[]); %--> Initialize processed dataset
    shift_sz = w/5; %--> Window shift length
    for i=1:size(ds_unprocessed,1)
        yp = buffer(ds_unprocessed(i).y,w,w-shift_sz,"nodelay").'; 
        yp = yp(all(yp,2),end); %--> Processed label 
        Xp = zeros(size(yp,1),size(features,2)*size(ds_unprocessed(i).X,2));
        for k = 1:size(features,2)
            for j = 1:size(ds_unprocessed(i).X,2)
                if(features(k)=="mean")
                    MEAN = mean(buffer(ds_unprocessed(i).X(:,j),w,w-shift_sz,...
                    "nodelay").',2); %--> Segment mean
                    Xp(:,(k-1)*size(ds_unprocessed(i).X,2)+j) = MEAN(1:size(yp,1),:); 
                        %--> Processed feature vector
                elseif(features(k)=="median")
                    MEDIAN = median(buffer(ds_unprocessed(i).X(:,j),w,w-shift_sz,...
                    "nodelay").',2); %--> Segment median
                    Xp(:,(k-1)*size(ds_unprocessed(i).X,2)+j) = MEDIAN(1:size(yp,1),:); 
                        %--> Processed feature vector
                elseif(features(k)=="std")
                    STD = std(buffer(ds_unprocessed(i).X(:,j),w,w-shift_sz,...
                    "nodelay").',0,2); %--> Segment std deviation
                    Xp(:,(k-1)*size(ds_unprocessed(i).X,2)+j) = STD(1:size(yp,1),:); 
                        %--> Processed feature vector
                elseif(features(k)=="min")
                    MIN = min(buffer(ds_unprocessed(i).X(:,j),w,w-shift_sz,...
                    "nodelay").',[],2); %--> Segment minimum
                    Xp(:,(k-1)*size(ds_unprocessed(i).X,2)+j) = MIN(1:size(yp,1),:); 
                        %--> Processed feature vector
                elseif(features(k)=="max")
                    MAX = max(buffer(ds_unprocessed(i).X(:,j),w,w-shift_sz,...
                    "nodelay").',[],2); %--> Segment maximum
                    Xp(:,(k-1)*size(ds_unprocessed(i).X,2)+j) = MAX(1:size(yp,1),:); 
                        %--> Processed feature vector
                elseif(features(k)=="initial")
                    INIT = buffer(ds_unprocessed(i).X(:,j),w,w-shift_sz,...
                    "nodelay").'; INIT = INIT(:,1); %--> Segment initial value
                    Xp(:,(k-1)*size(ds_unprocessed(i).X,2)+j) = INIT(1:size(yp,1),:); 
                        %--> Processed feature vector
                elseif(features(k)=="final")
                    FINAL = buffer(ds_unprocessed(i).X(:,j),w,w-shift_sz,...
                    "nodelay").'; FINAL = FINAL(:,end); %--> Segment final value
                    Xp(:,(k-1)*size(ds_unprocessed(i).X,2)+j) = FINAL(1:size(yp,1),:); 
                        %--> Processed feature vector
                elseif(features(k)=="MAV")
                    MAV = mean(abs(buffer(ds_unprocessed(i).X(:,j),w,w-shift_sz,...
                    "nodelay").'),2); %--> Segment mean absolute value
                    Xp(:,(k-1)*size(ds_unprocessed(i).X,2)+j) = MAV(1:size(yp,1),:); 
                        %--> Processed feature vector
                elseif(features(k)=="SSI")
                    SSI = sum(abs(buffer(ds_unprocessed(i).X(:,j),w,w-shift_sz,...
                    "nodelay").').^2,2); %--> Segment simple square integral
                    Xp(:,(k-1)*size(ds_unprocessed(i).X,2)+j) = SSI(1:size(yp,1),:); 
                        %--> Processed feature vector
                elseif(features(k)=="IEMG")
                    IEMG = sum(abs(buffer(ds_unprocessed(i).X(:,j),w,w-shift_sz,...
                    "nodelay").'),2); %--> Segment integrated emg
                    Xp(:,(k-1)*size(ds_unprocessed(i).X,2)+j) = IEMG(1:size(yp,1),:); 
                        %--> Processed feature vector
                elseif(features(k)=="WL")
                    WL = sum(abs(diff(buffer(ds_unprocessed(i).X(:,j),w,w-shift_sz,...
                    "nodelay").',1,2)),2); %--> Segment waveform length
                    Xp(:,(k-1)*size(ds_unprocessed(i).X,2)+j) = WL(1:size(yp,1),:); 
                        %--> Processed feature vector
                end
            end
        end
        ds_processed(i).X=Xp; ds_processed(i).y=yp;
    end
end

function [W] = PNN_train(dsTrain,dsCv,dsTest)
    %{ 
    creates and trains a neural network for phase classification.
    
    Parameters
    ----------
    dsTrain : struct array
        A structure array containing training data.
    dsCv : struct array
        A structure array containing cross-validation data.
    dsTest : struct array
        A structure array containing test data.
    
    Returns
    -------
    W : matrix
        Weights of the network.
    %}
    % Neural network hyperparameters
    num_hL = 1; %--> Number of hidden layers
    num_hu = 100; %--> Number of hidden layer units
    activ_fn = "tanh"; %--> Activation function for hidden units
    max_iter = 250; %--> Maximum number of iterations
    lambda = 4; %--> Regularization parameter
    % Unroll datasets & shuffle training dataset
    X = cell2mat({dsTrain(:).X}.'); y = cell2mat({dsTrain(:).y}.'); %--> Unroll training features
    Xcv = cell2mat({dsCv(:).X}.'); ycv = cell2mat({dsCv(:).y}.'); %--> Unroll validation features
    Xtest = cell2mat({dsTest(:).X}.'); ytest = cell2mat({dsTest(:).y}.'); %--> Unroll testing features
    if(isprime(size(X, 1)))
        X(end, :) = []; %--> Make sure training dataset size is not prime number
        y(end, :) = [];
    end
    m = size(X, 1); 
    idx = randperm(m); %--> Shuffle indices
    X(idx, :) = X; y(idx, :) = y; %--> New shuffled training dataset
    % Convert label vectors to logical expected output matrices
    num_labels = size(unique(y),1); %--> Number of labels
    Y = zeros(2, size(X, 1)); %--> Initialize training output matrix for phases
    Ycv = zeros(2, size(Xcv, 1)); %--> Initialize validation output matrix for phases
    Ytest = zeros(2, size(Xtest, 1)); %--> Initialize testing output matrix for phases
    for i = 1:num_labels
        Y(i, :) = y == i; %--> Training expected output
        Ycv(i, :) = ycv == i; %--> Validation expected output
        Ytest(i, :) = ytest == i; %--> Testing expected output
    end
    % Network configuration
    L_u = [size(X,2),num_hu*ones(1, num_hL),num_labels]; %--> Number of units per layer
    W = repmat(struct('w',[],'b',[],'fn',[]),1,num_hL+1); %--> Network weights info.
    W_unroll = []; fn_unroll = []; %--> Unroll weights and their parameters
    A = repmat(struct('a',[],'da',[]),1,num_hL+2); %--> Create unit activations structure
    A(1).a = X.'; %--> Input layer activations are the input features
    A(1).da = zeros(size(X.')); %--> Derivative of input layer units is zero, initially
    Acv = repmat(struct('a',[],'da',[]),1,num_hL+2); %--> Create unit activations structure "for validation"
    Acv(1).a = Xcv.'; %--> Input features for cross validation
    Acv(1).da = zeros(size(Xcv.')); %--> Derivative of input layer for validation
    Atest = repmat(struct('a',[],'da',[]),1,num_hL+2); %--> Create unit activations structure "for test"
    Atest(1).a = Xtest.'; %--> Input features for testing
    Atest(1).da = zeros(size(Xtest.')); %--> Derivative of input layer for testing
    % Initialize network weights depending on activation function type
    for i = 2:size(L_u, 2)
        if(i==size(L_u, 2)||activ_fn == "sigmoid"||activ_fn=="tanh")
            epsilon_init = sqrt(6/(L_u(i)+L_u(i-1))); %--> Weight variance
            W(i-1).w = rand(L_u(i),L_u(i-1))*2*epsilon_init...
                - epsilon_init; %--> Initialize weights
            if(i==size(L_u,2))
                W(i-1).fn = "sigmoid";
            else
                W(i-1).fn = activ_fn; %--> Assign activation function
            end
        elseif(activ_fn=="relu"||activ_fn=="leaky relu")
            epsilon_init = sqrt(2/L_u(i-1)); %--> Weight variance
            W(i-1).w = rand(L_u(i),L_u(i-1))*2*epsilon_init...
                - epsilon_init; %--> Initialize weights
            W(i-1).fn = activ_fn; %--> Assign activation function
        end
        W(i-1).b = zeros(L_u(i),1); %--> Initialize bias weights
        W_unroll = [W_unroll;W(i-1).w(:);W(i-1).b]; %--> Unroll weights
        fn_unroll = [fn_unroll,W(i-1).fn]; %--> Unroll activation functions
    end
    % Training process
    disp('Training progress: ');
    W_unroll = optimize(A,Acv,Y,Ycv,W_unroll,fn_unroll,L_u,lambda,max_iter);
    count = 1; %--> Start of weight count during rolling
    for k = 1:size(W,2) %--> Re-roll network weights
        W(k).w = reshape(W_unroll(count:count-1+L_u(k + 1)*L_u(k),1),...
                [L_u(k + 1),L_u(k)]);
        W(k).b = W_unroll(count+L_u(k + 1)*L_u(k):...
                count-1+L_u(k+1)*L_u(k)+L_u(k+1),1);
        if(k~=size(W, 2))
            count = count+L_u(k + 1)*L_u(k)+L_u(k+1); %--> Update count
        end
    end
    % Metrics
    % Training
    [J_train, acc, ~] = prop(A, Y, W_unroll, fn_unroll, L_u, 0.0); %--> Propagation function
    disp("ANN training cost: " + string(J_train)); %--> Training cost
    disp("ANN training accuracy: " + string(acc) + "%"); %--> Training classification accuracy 
    % Validation
    [J_cv, acc, ~] = prop(Acv, Ycv, W_unroll, fn_unroll, L_u, 0.0); %--> Propagation function
    disp("ANN validation cost: " + string(J_cv)); %--> Validation cost
    disp("ANN validation accuracy: " + string(acc) + "%"); %--> Validation classification accuracy
    % Testing
    [~, acc, ~] = prop(Atest, Ytest, W_unroll, fn_unroll, L_u, 0.0); %--> Propagation function
    disp("ANN testing accuracy: " + string(acc) + "%"); %--> Testing classification accuracy 
end

function W_unroll = optimize(AAA, AAAcv, Y, Ycv, W_unroll, fn_unroll, L_u, lambda, max_iter)
    %{ 
    Cost minimization function.
    
    Parameters
    ----------
    AAA : array
        An array containing activation for training.
    AAA : array
        An array containing activation for validation.
    Y : array
        True output/True labels
    W_unroll : array
        Unrolled weights
    fn_unroll : array
        Unrolled activation function of hidden layers
    L_u : array
        Number of units per layer
    lambda : double
        Regularization parameter
    max_iter : int
        Number of epochs
    
    
    Returns
    -------
    W_unroll : array
        Updated unrolled weights
    %}
    % Create cost function and accuracy bins
    Costs = zeros(1, max_iter); Costscv = zeros(1, max_iter);
    Accs = zeros(1, max_iter); Accscv = zeros(1, max_iter);

    % Minimization algorithm
    RHO = 0.01; %--> Bunch of constants for line searches
    SIG = 0.5; %--> Constants in the Wolfe-Powell conditions
    INT = 0.1; %--> Don't reevaluate within 0.1 of the limit of the current bracket
    EXT = 3.0; %--> Extrapolate maximum 3 times the current bracket
    MAX = 20; %--> Max 20 function evaluations per line search
    RATIO = 100; %--> Maximum allowed slope ratio
    red = 1;
    i = 0; %--> Zero the run length counter
    ls_failed = 0; %--> No previous line search has failed
    fX = [];
    [J, ~, grad] = prop(AAA, Y, W_unroll, fn_unroll, L_u, lambda); %--> Update cost
    f1 = J; df1 = grad; %--> Get cost and gradient
    i = i + (max_iter < 0); %--> Count epochs?!
    s = -df1; %--> Search direction is steepest
    d1 = -s' * s; %--> This is the slope
    z1 = red / (1 - d1); %--> Initial step is red/(|s|+1)
    
    while i < abs(max_iter) %--> While not finished
        i = i + (max_iter > 0); %--> Count iterations?!            
        X0 = W_unroll; f0 = f1; df0 = df1; %--> Make a copy of current values
        W_unroll = W_unroll + z1 * s; %--> Begin line search
        [J, ~, grad] = prop(AAA, Y, W_unroll, fn_unroll, L_u, lambda); %--> Update cost
        f2 = J; df2 = grad; %--> Get cost and gradient
        i = i + (max_iter < 0); %--> Count epochs?!
        d2 = df2' * s;
        f3 = f1; d3 = d1; z3 = -z1; %--> initialize point 3 equal to point 1
        if max_iter > 0, M = MAX; else, M = min(MAX, -max_iter - i); end
        success = 0; limit = -1; %--> Initialize quantities
        while 1
          while ((f2 > f1 + z1 * RHO * d1) || ...
                  (d2 > -SIG * d1)) && (M > 0) 
            limit = z1; %--> Tighten the bracket
            if f2 > f1
              z2 = z3 - (0.5 * d3 * z3 * z3) / (d3 * z3 + f2 - f3); %--> Quadratic fit
            else
              A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3); %-> Cubic fit
              B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2);
              z2 = (sqrt(B * B - A * d2 * z3 * z3) - B) / A; %--> Numerical error possible - ok!
            end
            if isnan(z2) || isinf(z2)
              z2 = z3 / 2; %--> If we had a numerical problem then bisect
            end
            z2 = max(min(z2, INT * z3),(1 - INT) * z3); %--> Don't accept too close to limits
            z1 = z1 + z2; %--> Update the step
            W_unroll = W_unroll + z2 * s;
            [J, ~, grad] = prop(AAA, Y, W_unroll, fn_unroll, L_u, lambda); %--> Update cost
            f2 = J; df2 = grad; %--> Get cost and gradient
            M = M - 1; i = i + (max_iter < 0); %--> Count epochs?!
            d2 = df2' * s;
            z3 = z3 - z2; %--> z3 is now relative to the location of z2
          end
          if f2 > f1 + z1 * RHO * d1 || d2 > -SIG * d1
            break; %--> This is a failure
          elseif d2 > SIG * d1
            success = 1; break; %--> Success
          elseif M == 0
            break; %--> Failure
          end
          A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3); %--> Make cubic extrapolation
          B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2);
          z2 = -d2 * z3 * z3 / (B + sqrt(B * B - A * d2 * z3 * z3)); %--> Num. error possible - ok!
          if ~isreal(z2) || isnan(z2) || isinf(z2) || z2 < 0 %--> Num prob or wrong sign?
            if limit < -0.5 %--> If we have no upper limit
              z2 = z1 * (EXT-1); %--> The extrapolate the maximum amount
            else
              z2 = (limit - z1) / 2; %--> Otherwise bisect
            end
          elseif (limit > -0.5) && (z2+z1 > limit) %--> Extraplation beyond max?
            z2 = (limit-z1)/2; %--> Bisect
          elseif (limit < -0.5) && (z2 + z1 > z1 * EXT) %--> Extrapolation beyond limit
            z2 = z1 * (EXT - 1.0); %--> Set to extrapolation limit
          elseif z2 < -z3 * INT
            z2 = -z3 * INT;
          elseif (limit > -0.5) && (z2 < (limit - z1) * (1.0 - INT)) %--> Too close to limit?
            z2 = (limit - z1) * (1.0 - INT);
          end
          f3 = f2; d3 = d2; z3 = -z2; %--> Set point 3 equal to point 2
          z1 = z1 + z2; W_unroll = W_unroll + z2 * s; %--> Update current estimates
          [J, ~, grad] = prop(AAA, Y, W_unroll, fn_unroll, L_u, lambda); %--> Update cost
          f2 = J; df2 = grad; %--> Get cost and gradient
          M = M - 1; i = i + (max_iter < 0); %--> Count epochs?!
          d2 = df2' * s;
        end %--> End of line search
    
        if success %--> If line search succeeded
          f1 = f2; fX = [fX' f1]';
          s = (df2' * df2 - df1' * df2) / (df1' * df1) * s - df2; %--> Polack-Ribiere direction
          tmp = df1; df1 = df2; df2 = tmp; %--> Swap derivatives
          d2 = df1' * s;
          if d2 > 0 %--> New slope must be negative
            s = -df1; %--> Otherwise use steepest direction
            d2 = -s' * s;    
          end
          z1 = z1 * min(RATIO, d1 / (d2 - realmin)); %--> Slope ratio but max RATIO
          d1 = d2;
          ls_failed = 0; %--> This line search did not fail
        else
          W_unroll = X0; f1 = f0; df1 = df0; %--> Restore point from before failed line search
          if ls_failed || i > abs(max_iter) %--> Line search failed twice in a row
            break; %--> Or we ran out of time, so we give up
          end
          tmp = df1; df1 = df2; df2 = tmp; %--> Swap derivatives
          s = -df1; %--> Try steepest
          d1 = -s' * s;
          z1 = 1 / (1 - d1);                     
          ls_failed = 1; %--> This line search failed
        end
        % Update cost one last time
        [Costs(i), Accs(i), ~] = prop(AAA, Y, W_unroll, fn_unroll, L_u, lambda); %--> Update cost
        [Costscv(i), Accscv(i), ~] = prop(AAAcv, Ycv, W_unroll, fn_unroll, L_u, lambda); %--> Update validation cost
        fprintf('\b\b\b\b%d%%', round((i / max_iter) * 100));
    end
    fprintf('\b\b\b\b%d%%', 100);
    % Plot the cost function vs. iterations
    figure
    plot(1 : max_iter, Costs, 'LineWidth', 2)
    hold on
    plot(1 : max_iter, Costscv, 'LineWidth', 2)
    legend('Training','Validation')
    xlabel("Iterations",'FontSize', 12, 'Color', 'k')
    ylabel("Cost", 'FontSize', 12, 'Color', 'k')
    set(gcf,'color','w'); %--> Confusion matrix plot
    % Plot accuracies vs. iterations
    figure
    plot(1 : max_iter, Accs / 100.0, 'LineWidth', 2)
    hold on
    plot(1 : max_iter, Accscv / 100.0, 'LineWidth', 2)
    legend('Training','Validation')
    xlabel("Iterations",'FontSize', 12, 'Color', 'k')
    ylabel("Accuracy", 'FontSize', 12, 'Color', 'k')
    set(gcf,'color','w'); %--> Confusion matrix plot
end

function [J, Acc, grad] = prop(A, Y, W_unroll, fn_unroll, L_u, lambda) 
    %{ 
    compute cost and gradient through propagation.
    
    Parameters
    ----------
    A : array
        An array containing activation.
    Y : array
        True output/True labels
    W_unroll : array
        Unrolled weights
    fn_unroll : array
        Unrolled activation function of hidden layers
    L_u : array
        Number of units per layer
    lambda : double
        Regularization parameter
    
    
    Returns
    -------
    J : double
        Cost
    Acc : double
        Prediction accuracy
    grad : array
        Network cost gradient
    %}
    % Forward propagation
    wsq_sum = 0; %--> Squared sum of weights
    count = 1; %--> Start of weight count during rolling
    for k = 2 : size(A, 2)
        z = reshape(W_unroll(count : count - 1 + L_u(k) * L_u(k - 1), 1), ...
            [L_u(k), L_u(k - 1)]) * A(k - 1).a + ...
            W_unroll(count + L_u(k) * L_u(k - 1) : ...
            count - 1 + L_u(k) * L_u(k - 1) + L_u(k), 1); %--> Layer output
        A(k).a = activ(z, fn_unroll(k - 1)); %--> Layer activation
        A(k).da = dactiv(z, fn_unroll(k - 1)); %--> Derivative of activation units w.r.t layer output
        wsq_sum = wsq_sum + sum(W_unroll(count : count - 1 + L_u(k) * L_u(k - 1), 1).^2); %--> Add weight squares
        if(k ~= size(A, 2))
            count = count + L_u(k) * L_u(k - 1) + L_u(k); %--> Update count
        end
    end

    % Compute cost function
    size_b = size(A(1).a, 2); %--> Training batch size
    J = (-1 / size_b) * sum(sum(Y .* log(A(end).a) + (1 - Y) .* log(1 - A(end).a))) + ...
        (lambda / (2 * size_b)) * wsq_sum;

    % Compute accuracy
    p = double(bsxfun(@eq, A(end).a.', max(A(end).a.', [], 2))).'; %--> One hot representation of predictions
    Acc = ((size(Y, 2) - nnz(p - Y)/2) / size(Y, 2)) * 100; %--> Accuracy computed

    % Backward propagation
    grad = zeros(size(W_unroll)); %--> Gradient initialization
    dz = A(end).a - Y; %--> Derivative w.r.t layer output
    for k = size(A, 2) : -1 : 2
        % Gradient computation
        dw = (1 / size_b) * dz * A(k - 1).a.'; %--> Derivative w.r.t weights
        db = (1 / size_b) * sum(dz, 2); %--> Derivative w.r.t bias weight
        dz = (reshape(W_unroll(count : count - 1 + L_u(k) * L_u(k - 1), 1), ...
            [L_u(k), L_u(k - 1)]).' * dz) .* A(k - 1).da; %--> Derivative w.r.t layer output
        grad(count : count - 1 + L_u(k) * L_u(k - 1), 1) = dw(:); %--> Update gradient
        grad(count + L_u(k) * L_u(k - 1) : count - 1 + L_u(k) * L_u(k - 1) + L_u(k), ...
            1) = db(:); %--> Update gradient
        if(k ~= 2)
            count = count - L_u(k - 1) - L_u(k - 1) * L_u(k - 2); %--> Update count
        end
    end
end

function a = activ(z, fn) 
    %{ 
    compute sigmoid function.
    
    Parameters
    ----------
    z : array
        An array containing weighted sums of inputs.
    fn : string
        Activation function
    
    Returns
    -------
    a : array
        An array containing activations 
    %}
    if(fn == "sigmoid")
        a = 1.0 ./ (1.0 + exp(-z));
    elseif(fn == "tanh")
        a = (exp(z) - exp(-z)) ./ (exp(z) + exp(-z));
    elseif(fn == "relu")
        a = max(0, z);
    elseif(fn == "leaky relu")
        a = max(0.01 * z, z);
    end
end

function g = dactiv(z,fn) 
    %{ 
    compute activation gradient.
    
    Parameters
    ----------
    z : array
        An array containing weighted sums of inputs
    fn : string
        Activation function
    
    Returns
    -------
    g : array
        An array containing activation derivative 
    %}
    if(fn=="sigmoid")
        g = activ(z, fn) .* (1 - activ(z, fn));
    elseif(fn == "tanh")
        g = 1 - activ(z, fn).^2;
    elseif(fn == "relu")
        if(z < 0)
            g = 0;
        else
            g = 1;
        end
    elseif(fn == "leaky relu")
        if(z < 0)
            g = 0.01;
        else
            g = 1;
        end
    end
end

function p = Predict(X, W)
    %{ 
    compute prediction.
    
    Parameters
    ----------
    X : matrix
        An array containing input features
    W : array
        Network weights
    
    Returns
    -------
    p : array
        An array containing output predictions
    %}
    % Compute input activations
    num_hL = size(W,2)-1; %--> Number of hidden layers
    A = repmat(struct('a',[],'da',[]),1,num_hL+2); %--> Create unit activations structure
    A(1).a = X.'; %--> Input layer activations are the input features
    A(1).da = zeros(size(X.')); %--> Derivative of input layer units is zero, initially
    % Forward propagation
    for k = 2 : size(A, 2)
        z = W(k - 1).w * A(k - 1).a + W(k - 1).b;
        A(k).a = activ(z, W(k - 1).fn);
    end
    p = A(end).a.';
    % Get classification results
    [~, p] = max(p, [], 2); 
end

function pdf = diagonal_mvnpdf_vectorized(X, mu, sigma_sq)
    %{ 
    compute multivariate gaussian pdf.
    
    Parameters
    ----------
    X : matrix
        An array containing input features
    mu : array
        Feature mean
    sigma_sq: matrix
        Feature covariance matrix
    
    Returns
    -------
    p : float
        Probability density
    %}

    % sigma_sq: [1×d] vector of variances
    sigma_sq = diag(sigma_sq).';
    d = size(X, 2);
    sigma = sqrt(sigma_sq);
    
    % Normalization constant
    norm_const = 1 / ((2*pi)^(d/2) * prod(sigma));
    
    % Centered data
    X_centered = X - mu;
    
    % Quadratic form for ALL points at once
    quad_form = sum((X_centered.^2) ./ sigma_sq, 2);
    
    % PDF for all points
    pdf = norm_const * exp(-0.5 * quad_form);
end

%clear
%clc

% Training options
%ds = "BLISS_modified"; %--> Dataset
%fsample = 500; %--> Sampling frequency
%win_size = 15; %--> Window size
%multiplier = 7; % Prominence multiplier
%unseen_code = "AB941"; %--> Unseen subject
%activity = 'W'; %--> Activity