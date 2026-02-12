function results = run_dvh_eval_local_photons(config, params)
%RUN_DVH_EVAL_LOCAL_PHOTONS Self-contained CT vs sCT DVH evaluation.
% Uses matRad only. No dependency on synthetic_CT_generation runtime code.

results = struct();
results.caseId = params.caseId;
results.targetRoiName = params.targetRoiName;
results.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
resolution = resolve_resolution_mm(config, params.caseId);

% Import CT + RTSTRUCT (if available).
[ctRef, cstRef] = import_case_for_matrad(params.ctDir, params.rtstructFile, params.targetRoiName, resolution);

% Import sCT image; include RTSTRUCT when available for importer compatibility.
[ctSynth, ~] = import_case_for_matrad(params.sctDir, params.rtstructFile, params.targetRoiName, resolution);
assert_same_grid(ctRef, ctSynth, params.caseId);

pln = default_photon_plan(cstRef, ctRef);

% CT dose optimization
stf = matRad_generateStf(ctRef, cstRef, pln);
dij = calc_dij_compat(ctRef, cstRef, stf, pln);
resRef = matRad_fluenceOptimization(dij, cstRef, pln);
resRef = matRad_planAnalysis(resRef, ctRef, cstRef, stf, pln, 'showDVH', false, 'showQI', false);
dvhRef = resRef.dvh;
qiRef = resRef.qi;

% sCT dose calculation using CT-optimized weights
resSynth = calc_forward_compat(ctSynth, cstRef, stf, pln, resRef.w);
resSynth = matRad_planAnalysis(resSynth, ctSynth, cstRef, stf, pln, 'showDVH', false, 'showQI', false);
dvhSynth = resSynth.dvh;
qiSynth = resSynth.qi;

% Collect results
tblRef = qi_struct_to_table(qiRef, params.caseId, 'CT_ref');
tblSynth = qi_struct_to_table(qiSynth, params.caseId, 'sCT');
tblDelta = compute_qi_delta(tblRef, tblSynth);

writetable(tblRef, fullfile(params.outputDir, 'dvh_metrics_ct.csv'));
writetable(tblSynth, fullfile(params.outputDir, 'dvh_metrics_sct.csv'));
writetable(tblDelta, fullfile(params.outputDir, 'dvh_metrics_delta.csv'));

results.qiRef = qiRef;
results.qiSynth = qiSynth;
results.dvhRef = dvhRef;
results.dvhSynth = dvhSynth;
results.metricsRef = tblRef;
results.metricsSynth = tblSynth;
results.metricsDelta = tblDelta;

save(fullfile(params.outputDir, 'matrad_workspace.mat'), ...
    'ctRef', 'ctSynth', 'cstRef', 'pln', 'stf', 'dij', 'resRef', 'resSynth', '-v7.3');
end

function [ct, cst] = import_case_for_matrad(ctDir, rtstructFile, targetRoiName, resolution)
ctFiles = dir(fullfile(ctDir, '*.dcm'));
if isempty(ctFiles)
    error('No DICOM CT files found in: %s', ctDir);
end

files = struct();
files.ct = strings(numel(ctFiles), 1);
for i = 1:numel(ctFiles)
    files.ct(i) = string(fullfile(ctFiles(i).folder, ctFiles(i).name));
end
files.resx = resolution.x;
files.resy = resolution.y;
files.resz = resolution.z;
files.useDoseGrid = 0;
files.rtdose = {};
files.rtplan = {};

if nargin >= 2 && ~isempty(rtstructFile) && isfile(rtstructFile)
    files.rtss = {rtstructFile};
else
    files.rtss = {};
end

[ct, cst] = matRad_importDicom_local(files, targetRoiName);
end

function resolution = resolve_resolution_mm(config, caseId)
resolution = struct('x', NaN, 'y', NaN, 'z', NaN);

refPath = fullfile(config.projectRoot, '11dvhEvalCases', 'test_patients_shared', caseId, 'reference_grid.json');
if exist(refPath, 'file') == 2
    ref = jsondecode(fileread(refPath));
    if isfield(ref, 'spacing') && numel(ref.spacing) >= 3
        resolution.x = double(ref.spacing(1));
        resolution.y = double(ref.spacing(2));
        resolution.z = double(ref.spacing(3));
        return;
    end
end

if isfield(config, 'defaultResolution') && numel(config.defaultResolution) == 3
    resolution.x = double(config.defaultResolution(1));
    resolution.y = double(config.defaultResolution(2));
    resolution.z = double(config.defaultResolution(3));
    return;
end

error(['Could not determine voxel spacing without dicominfo/Image Processing Toolbox. ' ...
       'Provide reference_grid.json for this case or set cfg.defaultResolution = [x y z].']);
end

function pln = default_photon_plan(cst, ct)
pln = struct();
pln.radiationMode = 'photons';
pln.machine = 'Generic';
pln.propOpt.bioOptimization = 'none';
pln.numOfFractions = 1;

pln.propStf.gantryAngles = [0 23 50 75 95 110 150 170 190 210 250 265 285 310 337];
pln.propStf.couchAngles = zeros(1, numel(pln.propStf.gantryAngles));
pln.propStf.bixelWidth = 4;
pln.propStf.numOfBeams = numel(pln.propStf.gantryAngles);
pln.propStf.isoCenter = ones(pln.propStf.numOfBeams,1) * matRad_getIsoCenter(cst, ct, 0);

pln.propDoseCalc.doseGrid.resolution.x = 4;
pln.propDoseCalc.doseGrid.resolution.y = 4;
pln.propDoseCalc.doseGrid.resolution.z = 4;

pln.propOpt.runSequencing = 1;
pln.numOfLevels = 5;
pln.propOpt.runDAO = 0;
end

function assert_same_grid(ctRef, ctSynth, caseId)
refCube = get_ct_cube(ctRef);
synthCube = get_ct_cube(ctSynth);

if ~isequal(size(refCube), size(synthCube))
    error('%s: CT/sCT cube size mismatch in matRad import', caseId);
end

if isfield(ctRef, 'resolution') && isfield(ctSynth, 'resolution')
    r1 = ctRef.resolution;
    r2 = ctSynth.resolution;
    if abs(r1.x - r2.x) > 1e-3 || abs(r1.y - r2.y) > 1e-3 || abs(r1.z - r2.z) > 1e-3
        error('%s: CT/sCT resolution mismatch in matRad import', caseId);
    end
end
end

function cube = get_ct_cube(ct)
if isfield(ct, 'cube') && ~isempty(ct.cube) && numel(ct.cube) >= 1
    cube = ct.cube{1};
    return;
end

if isfield(ct, 'cubeHU') && ~isempty(ct.cubeHU) && numel(ct.cubeHU) >= 1
    cube = ct.cubeHU{1};
    return;
end

if isfield(ct, 'cubeIV') && ~isempty(ct.cubeIV) && numel(ct.cubeIV) >= 1
    cube = ct.cubeIV{1};
    return;
end

error('CT struct has no recognized cube field (expected cube, cubeHU, or cubeIV).');
end

function tbl = qi_struct_to_table(qi, caseId, imageType)
tbl = struct2table(qi);
tbl.caseId = repmat(string(caseId), height(tbl), 1);
tbl.imageType = repmat(string(imageType), height(tbl), 1);
end

function out = compute_qi_delta(refTbl, synthTbl)
% Match by structure name if available.
if any(strcmp(refTbl.Properties.VariableNames, 'name')) && any(strcmp(synthTbl.Properties.VariableNames, 'name'))
    refTbl = sortrows(refTbl, 'name');
    synthTbl = sortrows(synthTbl, 'name');
end

out = refTbl;

vars = intersect(refTbl.Properties.VariableNames, synthTbl.Properties.VariableNames);
for i = 1:numel(vars)
    v = vars{i};
    if isnumeric(refTbl.(v)) && isnumeric(synthTbl.(v))
        out.(v) = synthTbl.(v) - refTbl.(v);
    end
end

out.imageType = repmat("sCT_minus_CT", height(out), 1);
end

function dij = calc_dij_compat(ct, cst, stf, pln)
if exist('matRad_calcDoseInfluence', 'file') == 2
    dij = matRad_calcDoseInfluence(ct, cst, stf, pln);
    return;
end

% Legacy fallback
dij = matRad_calcPhotonDose(ct, stf, pln, cst);
end

function resultGUI = calc_forward_compat(ct, cst, stf, pln, w)
if exist('matRad_calcDoseForward', 'file') == 2
    resultGUI = matRad_calcDoseForward(ct, cst, stf, pln, w);
    return;
end

% Legacy fallback
resultGUI = matRad_calcDoseDirect(ct, stf, pln, cst, w);
end
