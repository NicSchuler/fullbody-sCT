function cst = matRad_createCst_local(structures, targetRoiName)
%MATRAD_CREATECST_LOCAL Build cst with a configurable target ROI.

nStructures = numel(structures);
cst = cell(nStructures, 6);
defaultColors = colorcube(max(nStructures, 1));
targetNorm = normalize_roi_name(targetRoiName);

for i = 1:nStructures
    cst{i,1} = i - 1;
    cst{i,2} = structures(i).structName;
    cst{i,4}{1} = structures(i).indices;

    hasVoxels = ~isempty(cst{i,4}{1});
    if ~hasVoxels
        cst{i,3} = 'IGNORED';
        cst{i,5}.Priority = 3;
        cst{i,6} = [];
    elseif strcmp(normalize_roi_name(cst{i,2}), targetNorm)
        cst{i,3} = 'TARGET';
        cst{i,5}.Priority = 1;
        objective = DoseObjectives.matRad_MinDVH;
        objective.penalty = 1;
        objective.parameters = {54,95};
        cst{i,6}{1} = struct(objective);
    else
        cst{i,3} = 'OAR';
        cst{i,5}.Priority = 2;
        objective = DoseObjectives.matRad_MeanDose;
        objective.penalty = 1;
        objective.parameters = {15};
        cst{i,6}{1} = struct(objective);
    end

    cst{i,5}.alphaX = 0.1;
    cst{i,5}.betaX = 0.05;
    cst{i,5}.Visible = 1;

    if isfield(structures(i), 'structColor') && ~isempty(structures(i).structColor)
        cst{i,5}.visibleColor = structures(i).structColor' ./ 255;
    else
        cst{i,5}.visibleColor = defaultColors(i,:);
    end
end

allNames = string(cst(:,2));
allNamesNorm = strings(size(allNames));
for k = 1:numel(allNames)
    allNamesNorm(k) = normalize_roi_name(allNames(k));
end
if ~any(strcmp(allNamesNorm, targetNorm))
    error('Target ROI "%s" not found in RTSTRUCT. Available: %s', targetRoiName, strjoin(allNames, ', '));
end
end

function out = normalize_roi_name(nameIn)
name = lower(char(string(nameIn)));
name = regexprep(name, '[^a-z0-9]+', '_');
name = regexprep(name, '^_+|_+$', '');
out = string(name);
end
