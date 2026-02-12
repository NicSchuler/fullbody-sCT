function run_dvh_pipeline(config)
%RUN_DVH_PIPELINE End-to-end DVH evaluation wrapper.
%
% This script expects:
% 1) DICOM outputs from scripts/nifti_to_dicom.py
% 2) Local clone of matRad (dev branch recommended)
%
% Example:
%   cfg = struct();
%   cfg.projectRoot = '/Users/flavianthur/Documents/dhv';
%   cfg.dicomRoot = fullfile(cfg.projectRoot, 'outputs', 'dicom', '2_experiment_cut_synthrad_abdomen_32p99');
%   cfg.caseIds = {'AB_1ABA068','AB_1ABB070','AB_1ABC127'};
%   cfg.matRadRoot = '/path/to/matRad';
%   cfg.targetRoiName = 'kidney_left';
%   cfg.outputRoot = fullfile(cfg.projectRoot, 'outputs', 'dvh_results');
%   run_dvh_pipeline(cfg);

arguments
    config struct
end

required = {'projectRoot','dicomRoot','caseIds','matRadRoot','outputRoot'};
for i = 1:numel(required)
    f = required{i};
    if ~isfield(config, f)
        error('Missing config.%s', f);
    end
end

if ~exist(config.outputRoot, 'dir')
    mkdir(config.outputRoot);
end

if ~isfield(config, 'targetRoiName')
    config.targetRoiName = 'kidney_left';
end

addpath(genpath(config.matRadRoot));
matRadCfg = fullfile(config.matRadRoot, 'matRad_cfg.m');
if exist(matRadCfg, 'file') == 2
    run(matRadCfg);
end

for i = 1:numel(config.caseIds)
    caseId = config.caseIds{i};

    caseDicom = fullfile(config.dicomRoot, caseId);
    ctDir = fullfile(caseDicom, 'CT');
    sctDir = fullfile(caseDicom, 'sCT');
    rtstructFile = fullfile(caseDicom, 'RTSTRUCT', 'rtstruct.dcm');

    if exist(ctDir, 'dir') ~= 7
        error('Missing CT DICOM dir for case %s: %s', caseId, ctDir);
    end
    if exist(sctDir, 'dir') ~= 7
        error('Missing sCT DICOM dir for case %s: %s', caseId, sctDir);
    end

    outCaseDir = fullfile(config.outputRoot, caseId);
    if ~exist(outCaseDir, 'dir')
        mkdir(outCaseDir);
    end

    fprintf('[INFO] Running DVH workflow for case %s\n', caseId);
    params = struct();
    params.caseId = caseId;
    params.ctDir = ctDir;
    params.sctDir = sctDir;
    params.rtstructFile = rtstructFile;
    params.outputDir = outCaseDir;
    params.targetRoiName = config.targetRoiName;

    if exist(rtstructFile, 'file') ~= 2
        error(['Missing RTSTRUCT for case %s: %s\n' ...
               'DVH planning in this pipeline requires structures/target ROI.\n' ...
               'Regenerate DICOM with RTSTRUCT, e.g.:\n' ...
               '  python3 scripts/10nifti_to_dicom.py --data-root 11dvhEvalCases --out-root outputs/dicom --build-rtstruct\n' ...
               'Also ensure optional deps are installed:\n' ...
               '  python3 -m pip install nibabel rt-utils'], caseId, rtstructFile);
    end

    results = run_dvh_eval_local_photons(config, params);
    save(fullfile(outCaseDir, 'dvh_eval.mat'), 'results');

    save(fullfile(outCaseDir, 'pipeline_params.mat'), 'params');
end

fprintf('[INFO] DVH pipeline completed. Results root: %s\n', config.outputRoot);
end
