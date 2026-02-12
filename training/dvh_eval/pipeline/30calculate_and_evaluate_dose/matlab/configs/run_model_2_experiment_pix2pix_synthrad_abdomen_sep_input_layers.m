% Auto-generated config script for model: 2_experiment_pix2pix_synthrad_abdomen_sep_input_layers

thisFile = mfilename('fullpath');
configDir = fileparts(thisFile);
matlabDir = fileparts(configDir);
addpath(matlabDir);

cfg = build_dvh_config('2_experiment_pix2pix_synthrad_abdomen_sep_input_layers');
run_dvh_pipeline(cfg);
