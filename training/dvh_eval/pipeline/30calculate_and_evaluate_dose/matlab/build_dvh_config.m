function cfg = build_dvh_config(modelName, varargin)
%BUILD_DVH_CONFIG Build standard DVH pipeline config for a model.
%
% cfg = build_dvh_config(modelName)
% cfg = build_dvh_config(modelName, 'targetRoiName', 'kidney_right')

p = inputParser;
p.addRequired('modelName', @(x) ischar(x) || isstring(x));
p.addParameter('targetRoiName', 'kidney_left', @(x) ischar(x) || isstring(x));
p.addParameter('matRadRoot', '', @(x) ischar(x) || isstring(x));
p.parse(modelName, varargin{:});

modelName = char(p.Results.modelName);

% Resolve repository root from this file location.
thisFile = mfilename('fullpath');
matlabDir = fileparts(thisFile);
projectRoot = fileparts(matlabDir);

casesRoot = fullfile(projectRoot, '11dvhEvalCases', 'test_patients_shared');
caseDirs = dir(casesRoot);
caseDirs = caseDirs([caseDirs.isdir]);
caseIds = {caseDirs.name};
caseIds = caseIds(~ismember(caseIds, {'.', '..'}));
caseIds = sort(caseIds);

if isempty(caseIds)
    error('No shared test cases found in %s', casesRoot);
end

cfg = struct();
cfg.projectRoot = projectRoot;
cfg.dicomRoot = fullfile(projectRoot, 'outputs', 'dicom', modelName);
cfg.caseIds = caseIds;

if strlength(string(p.Results.matRadRoot)) > 0
    cfg.matRadRoot = char(p.Results.matRadRoot);
else
    cfg.matRadRoot = fullfile(projectRoot, 'matRad');
end

cfg.targetRoiName = char(p.Results.targetRoiName);
cfg.outputRoot = fullfile(projectRoot, 'outputs', 'dvh_results', modelName);
