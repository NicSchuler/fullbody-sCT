function [ct, cst] = matRad_importDicom_local(files, targetRoiName)
%MATRAD_IMPORTDICOM_LOCAL Minimal local importer (CT + optional RTSTRUCT).

if ~isfield(files, 'resx') || ~isfield(files, 'resy') || ~isfield(files, 'resz')
    error('files.resx/resy/resz are required');
end

if exist('license', 'builtin') == 5
    if ~license('test', 'image_toolbox')
        error(['matRad DICOM import requires MATLAB Image Processing Toolbox. ' ...
               'Please install/activate it, then rerun the pipeline.']);
    end
end

% Prefer modern class-based importer (current matRad API).
if exist('matRad_DicomImporter', 'class') == 8
    ctDir = fileparts(char(files.ct(1)));
    importer = matRad_DicomImporter(ctDir);

    % Override auto-detected sets with the exact files from the pipeline.
    importer.importFiles.ct = cell(numel(files.ct), 1);
    for i = 1:numel(files.ct)
        importer.importFiles.ct{i,1} = char(files.ct(i));
    end
    importer.importFiles.resx = num2str(files.resx);
    importer.importFiles.resy = num2str(files.resy);
    importer.importFiles.resz = num2str(files.resz);
    importer.importFiles.useImportGrid = false;
    importer.importFiles.rtdose = {};
    importer.importFiles.rtplan = {};

    if isfield(files, 'rtss') && ~isempty(files.rtss)
        importer.importFiles.rtss = {char(files.rtss{1})};
    else
        importer.importFiles.rtss = {};
    end

    importer.matRad_importDicom();

    ct = importer.ct;
    cst = importer.cst;

    if isfield(files, 'rtss') && ~isempty(files.rtss)
        cst = matRad_createCst_local(importer.importRtss.structures, targetRoiName);
    end
    return;
end

% Legacy functional importer fallback.
resolution.x = files.resx;
resolution.y = files.resy;
resolution.z = files.resz;

legacyImported = false;
legacyErr = '';

try
    ct = matRad_importDicomCt(files.ct, resolution, false);
    legacyImported = true;
catch ME
    legacyErr = ME.message;
end

if ~legacyImported
    try
        ct = matRad_importDicomCt(files.ct, resolution);
        legacyImported = true;
    catch ME
        legacyErr = [legacyErr ' | ' ME.message];
    end
end

if ~legacyImported
    error('Could not import DICOM CT with current matRad API. Details: %s', legacyErr);
end

if isfield(files, 'rtss') && ~isempty(files.rtss)
    structures = matRad_importDicomRtss(files.rtss{1}, ct.dicomInfo);
    for i = 1:numel(structures)
        structures(i).indices = matRad_convRtssContours2Indices(structures(i), ct);
    end
    cst = matRad_createCst_local(structures, targetRoiName);
else
    cst = matRad_dummyCst(ct);
end
end
