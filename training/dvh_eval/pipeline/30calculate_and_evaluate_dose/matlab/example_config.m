% Copy this file and adjust paths before running run_dvh_pipeline.

cfg = struct();
cfg.projectRoot = '/Users/flavianthur/Documents/dvh';
cfg.dicomRoot = fullfile(cfg.projectRoot, 'outputs', 'dicom', '2_experiment_cut_synthrad_abdomen_32p99');
cfg.caseIds = {'AB_1ABA068','AB_1ABB070','AB_1ABC127'};

% Set these to your local clones:
cfg.matRadRoot = '/Users/flavianthur/Documents/dvh/matRad';

% Assume one kidney is the target volume:
cfg.targetRoiName = 'kidney_left';

% Optional fallback spacing [mm] if reference_grid.json is missing:
% cfg.defaultResolution = [1 1 3];

cfg.outputRoot = fullfile(cfg.projectRoot, 'outputs', 'dvh_results', '2_experiment_cut_synthrad_abdomen_32p99');

run_dvh_pipeline(cfg);
