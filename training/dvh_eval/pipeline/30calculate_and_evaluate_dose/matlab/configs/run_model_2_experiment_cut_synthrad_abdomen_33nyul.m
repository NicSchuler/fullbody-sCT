% Auto-generated config script for model: 2_experiment_cut_synthrad_abdomen_33nyul

thisFile = mfilename('fullpath');
configDir = fileparts(thisFile);
matlabDir = fileparts(configDir);
addpath(matlabDir);

cfg = build_dvh_config('2_experiment_cut_synthrad_abdomen_33nyul');
run_dvh_pipeline(cfg);
