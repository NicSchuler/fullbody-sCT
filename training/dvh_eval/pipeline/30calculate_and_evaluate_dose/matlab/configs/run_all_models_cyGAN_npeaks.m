% Run DVH pipeline for all generated model config scripts.
%
% User options:
%   excludeModels: exact model names to skip.
%   includePattern: regex; only matching model names are run.
%   skipIfResultsExist: if true, skip models that already have an output folder.

excludeModels = {
    '2_experiment_cut_synthrad_abdomen_32p99',
    '2_experiment_cut_synthrad_abdomen_33nyul',
    '2_experiment_cut_synthrad_abdomen_sep_first_layer',
    '2_experiment_cyclegan_abdomen_32p99',
    '2_experiment_cyclegan_abdomen_33nyul',
    '2_experiment_cyclegan_abdomen_sep_first_layer',
    '2_experiment_pix2pix_synthrad_abdomen_32p99',
    '2_experiment_pix2pix_synthrad_abdomen_sep_input_layers'
};
includePattern = 'cyclegan_abdomen_34normalized_n4';
skipIfResultsExist = true;

thisFile = mfilename('fullpath');
configDir = fileparts(thisFile);
matlabDir = fileparts(configDir);
projectRoot = fileparts(matlabDir);
resultsRoot = fullfile(projectRoot, 'outputs', 'dvh_results');

% Ensure helper functions are resolvable (build_dvh_config, run_dvh_pipeline, ...).
addpath(matlabDir);

entries = dir(fullfile(configDir, 'run_model_*.m'));
entries = sort({entries.name});

for i = 1:numel(entries)
    entryName = entries{i};
    modelName = regexprep(entryName, '^run_model_(.+)\.m$', '$1');

    if any(strcmp(modelName, excludeModels))
        fprintf('[INFO] Skipping excluded model: %s\n', modelName);
        continue;
    end

    if ~isempty(includePattern) && isempty(regexp(modelName, includePattern, 'once'))
        fprintf('[INFO] Skipping non-matching model: %s\n', modelName);
        continue;
    end

    modelOutputDir = fullfile(resultsRoot, modelName);
    if skipIfResultsExist && exist(modelOutputDir, 'dir') == 7
        fprintf('[INFO] Skipping existing results model: %s\n', modelName);
        continue;
    end

    scriptPath = fullfile(configDir, entries{i});
    fprintf('[INFO] Running config script: %s\n', entryName);
    run(scriptPath);
end
