#!/usr/bin/env bash
#
# Full inference pipeline for MR-to-CT synthesis.
#
# Pipeline steps:
#   0. Prepare input data (GC format -> 1initNifti with region prefix)
#   1. Resample MR (no crop to bbox)
#   2. Normalize MR (baseline normalization)
#   3. Create 2D slices (MR only)
#   4. Run CUT model inference
#   5. Reconstruct 3D volume
#   6. Copy results to output (10reconstruction -> GC output)
#
# Usage:
#   ./synthrad2025_inference.sh [options]
#
# Examples:
#   # Run with defaults:
#   ./synthrad2025_inference.sh
#
#   # Specify GPU:
#   ./synthrad2025_inference.sh --gpu 1
#
#   # Custom temp directory:
#   ./synthrad2025_inference.sh \
#       --output-dir /path/to/temp \
#       --gpu 0
#
#   # Process specific patients:
#   ./synthrad2025_inference.sh --patient-ids AB_1ABA005 AB_1ABA006
#
# Pipeline output structure:
#   output-dir/
#   ├── 2resampledNifti/{patient_id}/MR/*.nii.gz
#   ├── {normalization}/3normalized/{patient_id}/MR/*.nii.gz
#   ├── {normalization}/5slices/model_2d/full/A/*.nii
#   ├── 9inference/{model_name}/test_{epoch}/fake_nifti/*.nii
#   └── 10reconstruction/{patient_id}/sCT_original_dim_reconstructed_alignment.nii.gz

set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Conda environments
PREPROC_ENV="preprocessing_env_docker"
MODEL_ENV="inference_env_docker"

# Default values (can be overridden via command line)
GC_INPUT_DIR="/local/scratch/datasets/FullbodySCT/nicolas_test_pipeline/input"
GC_OUTPUT_DIR="/local/scratch/datasets/FullbodySCT/nicolas_test_pipeline/output"
OUTPUT_DIR="/local/scratch/datasets/FullbodySCT/nicolas_test_pipeline/temp"
CHECKPOINT_DIR="/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints"
MODEL_NAME="cut_synthrad_allregions_final"
EPOCH="50"
GPU="1"
NORMALIZATION="31baseline"
PATIENT_IDS=()
SKIP_PREPROCESSING=false
SKIP_INFERENCE=false
SKIP_RECONSTRUCTION=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gc-input)
            GC_INPUT_DIR="$2"
            shift 2
            ;;
        --gc-output)
            GC_OUTPUT_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --epoch)
            EPOCH="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --normalization)
            NORMALIZATION="$2"
            shift 2
            ;;
        --patient-ids)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                PATIENT_IDS+=("$1")
                shift
            done
            ;;
        --skip-preprocessing)
            SKIP_PREPROCESSING=true
            shift
            ;;
        --skip-inference)
            SKIP_INFERENCE=true
            shift
            ;;
        --skip-reconstruction)
            SKIP_RECONSTRUCTION=true
            shift
            ;;
        --help|-h)
            echo "MR-to-CT Synthesis Inference Pipeline"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --gc-input DIR            Grand Challenge input directory"
            echo "                            Default: $GC_INPUT_DIR"
            echo "  --gc-output DIR           Grand Challenge output directory"
            echo "                            Default: $GC_OUTPUT_DIR"
            echo "  --output-dir DIR          Temp/working directory"
            echo "                            Default: $OUTPUT_DIR"
            echo "  --checkpoint-dir DIR      Checkpoint directory"
            echo "                            Default: $CHECKPOINT_DIR"
            echo "  --model-name NAME         Model name (folder in checkpoint dir)"
            echo "                            Default: $MODEL_NAME"
            echo "  --epoch EPOCH             Epoch number"
            echo "                            Default: $EPOCH"
            echo "  --gpu GPU_ID              GPU ID (-1 for CPU)"
            echo "                            Default: $GPU"
            echo "  --normalization METHOD    Normalization method (31baseline or 32p99)"
            echo "                            Default: $NORMALIZATION"
            echo "  --patient-ids ID1 ID2...  Specific patient IDs to process"
            echo "  --skip-preprocessing      Skip preprocessing steps"
            echo "  --skip-inference          Skip model inference"
            echo "  --skip-reconstruction     Skip volume reconstruction"
            echo "  --help, -h                Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Run with defaults:"
            echo "  $0"
            echo ""
            echo "  # Specify GPU:"
            echo "  $0 --gpu 1"
            echo ""
            echo "  # Process specific patient:"
            echo "  $0 --patient-ids AB_1ABA005"
            echo ""
            echo "  # Custom temp directory:"
            echo "  $0 --output-dir /my/temp --gpu 0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Derived paths
INPUT_DIR="${OUTPUT_DIR}/1initNifti"
PREPROC_SCRIPTS="${SCRIPT_DIR}/preprocessing/preprocessing_synthrad"
TRAINING_DIR="${SCRIPT_DIR}/training_cut"

DIR_RESAMPLED="${OUTPUT_DIR}/2resampledNifti"
DIR_NORMALIZED="${OUTPUT_DIR}/${NORMALIZATION}/3normalized"
DIR_SLICES="${OUTPUT_DIR}/${NORMALIZATION}/5slices"
DIR_INFERENCE="${OUTPUT_DIR}/9inference"
DIR_RECONSTRUCTION="${OUTPUT_DIR}/10reconstruction"

# Build patient filter arguments
PATIENT_ARGS=()
if [[ ${#PATIENT_IDS[@]} -gt 0 ]]; then
    PATIENT_ARGS=("--patient-ids" "${PATIENT_IDS[@]}")
fi

# =========================================================================
# Print configuration
# =========================================================================

echo ""
echo "======================================================================"
echo "MR-to-CT Synthesis Inference Pipeline"
echo "======================================================================"
echo "GC Input:        ${GC_INPUT_DIR}"
echo "GC Output:       ${GC_OUTPUT_DIR}"
echo "Temp dir:        ${OUTPUT_DIR}"
echo "Checkpoint dir:  ${CHECKPOINT_DIR}"
echo "Model:           ${MODEL_NAME}"
echo "Epoch:           ${EPOCH}"
echo "GPU:             ${GPU}"
echo "Normalization:   ${NORMALIZATION}"
if [[ ${#PATIENT_IDS[@]} -gt 0 ]]; then
    echo "Patient IDs:     ${PATIENT_IDS[*]}"
fi
echo "======================================================================"
echo ""

# =========================================================================
# Validation
# =========================================================================

if [[ ! -d "${GC_INPUT_DIR}" ]]; then
    echo "ERROR: GC input directory does not exist: ${GC_INPUT_DIR}"
    exit 1
fi

if [[ ! -f "${GC_INPUT_DIR}/region.json" ]]; then
    echo "ERROR: region.json not found in: ${GC_INPUT_DIR}"
    exit 1
fi

CHECKPOINT_PATH="${CHECKPOINT_DIR}/${MODEL_NAME}/${EPOCH}_net_G.pth"
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "WARNING: Checkpoint not found: ${CHECKPOINT_PATH}"
    echo "         Inference step will fail if not skipped."
fi

# Create output directories
mkdir -p "${DIR_RESAMPLED}" "${DIR_NORMALIZED}" "${DIR_SLICES}" "${DIR_INFERENCE}" "${DIR_RECONSTRUCTION}"

# =========================================================================
# Helper: run a pipeline step with logging
# =========================================================================

run_step() {
    local show_cmd=true
    if [[ "$1" == "--no-cmd" ]]; then
        show_cmd=false
        shift
    fi
    local step_name="$1"
    shift
    echo ""
    echo "======================================================================"
    echo "STEP: ${step_name}"
    echo "======================================================================"
    if [[ "${show_cmd}" == true ]]; then
        echo "Command: $*"
    fi
    echo ""

    "$@"

    echo ""
    echo "COMPLETED: ${step_name}"
    echo "----------------------------------------------------------------------"
}

# =========================================================================
# STEP 0: PREPARE INPUT DATA (GC format -> 1initNifti)
# =========================================================================

run_step --no-cmd "0. Preparing input data" \
    conda run -n "${PREPROC_ENV}" python -c "
import json, shutil, os, sys, glob
import SimpleITK as sitk

gc_input = sys.argv[1]
init_dir = sys.argv[2]

def convert_to_nii_gz(src, dst):
    \"\"\"Copy if already .nii.gz, otherwise convert via SimpleITK.\"\"\"
    if src.endswith('.nii.gz'):
        shutil.copy2(src, dst)
    else:
        img = sitk.ReadImage(src)
        sitk.WriteImage(img, dst)

# Read region
with open(os.path.join(gc_input, 'region.json')) as f:
    region = json.load(f)

prefix_map = {'Head and Neck': 'HN', 'Abdomen': 'AB', 'Thorax': 'TH'}
prefix = prefix_map.get(region)
if not prefix:
    print(f'ERROR: Unknown region: {region}')
    sys.exit(1)

print(f'Region: {region} -> Prefix: {prefix}')

# Discover MRI files from input/images/mri/ (.nii.gz, .nii, .mha)
mri_dir = os.path.join(gc_input, 'images', 'mri')
if not os.path.isdir(mri_dir):
    print(f'ERROR: MRI directory not found: {mri_dir}')
    sys.exit(1)

mri_files = sorted(
    glob.glob(os.path.join(mri_dir, '*.nii.gz'))
    + glob.glob(os.path.join(mri_dir, '*.nii'))
    + glob.glob(os.path.join(mri_dir, '*.mha'))
)
if not mri_files:
    print(f'ERROR: No image files found in {mri_dir}')
        sys.exit(1)

# Discover body mask files (optional, matched to MRI files by sorted index)
body_dir = os.path.join(gc_input, 'images', 'body')
body_files = []
if os.path.isdir(body_dir):
    body_files = sorted(
        glob.glob(os.path.join(body_dir, '*.nii.gz'))
        + glob.glob(os.path.join(body_dir, '*.nii'))
        + glob.glob(os.path.join(body_dir, '*.mha'))
    )

# Mapping: patient_id -> original MRI filename (to restore format in Step 6)
input_mapping = {}

for idx, mri_path in enumerate(mri_files):
    basename = os.path.basename(mri_path)
    patient_id = f'{prefix}_{idx + 1:03d}'

    input_mapping[patient_id] = basename

    # Create directory structure
    mr_dir = os.path.join(init_dir, patient_id, 'MR')
    mask_dir = os.path.join(init_dir, patient_id, 'masks')
    os.makedirs(mr_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # Copy/convert MRI to .nii.gz
    dst_mri = os.path.join(mr_dir, f'{patient_id}_MR.nii.gz')
    convert_to_nii_gz(mri_path, dst_mri)
    print(f'  MRI:  {mri_path} -> {dst_mri}')

    # Copy/convert body mask if available (matched by sorted index)
    if idx < len(body_files):
        dst_body = os.path.join(mask_dir, f'{patient_id}_mask.nii.gz')
        convert_to_nii_gz(body_files[idx], dst_body)
        print(f'  Mask: {body_files[idx]} -> {dst_body}')

# Save mapping for Step 6
mapping_path = os.path.join(init_dir, 'input_mapping.json')
with open(mapping_path, 'w') as f:
    json.dump(input_mapping, f)
print(f'  Mapping saved to {mapping_path}')
" "${GC_INPUT_DIR}" "${INPUT_DIR}"

# =========================================================================
# PREPROCESSING STEPS
# =========================================================================

if [[ "${SKIP_PREPROCESSING}" == false ]]; then

    # Step 1: Resampling (MR only, no crop to bbox)
    run_step "1. Resampling MR volumes" \
        conda run -n "${PREPROC_ENV}" python \
            "${PREPROC_SCRIPTS}/20resampling.py" \
            --src-root "${INPUT_DIR}" \
            --out-root "${DIR_RESAMPLED}" \
            --mr-only \
            --skip-crop-to-bbox \
            "${PATIENT_ARGS[@]+"${PATIENT_ARGS[@]}"}"

    # Step 2: Normalization (MR only)
    run_step "2. Normalizing MR volumes" \
        conda run -n "${PREPROC_ENV}" python \
            "${PREPROC_SCRIPTS}/31baseline_standardization.py" \
            --src-root "${DIR_RESAMPLED}" \
            --out-root "${DIR_NORMALIZED}" \
            --mr-only \
            --all-data \
            "${PATIENT_ARGS[@]+"${PATIENT_ARGS[@]}"}"

    # Step 3: Create slices (MR only)
    run_step "3. Creating 2D slices" \
        conda run -n "${PREPROC_ENV}" python \
            "${PREPROC_SCRIPTS}/40slice_creator.py" \
            "${NORMALIZATION}" \
            --base-root "${OUTPUT_DIR}" \
            --mr-only \
            "${PATIENT_ARGS[@]+"${PATIENT_ARGS[@]}"}"

else
    echo ""
    echo "SKIPPING: Preprocessing steps (--skip-preprocessing)"
fi

# =========================================================================
# MODEL INFERENCE
# =========================================================================

if [[ "${SKIP_INFERENCE}" == false ]]; then

    SLICES_DIR="${DIR_SLICES}/model_2d/full/A"

    if [[ ! -d "${SLICES_DIR}" ]]; then
        echo "ERROR: Slices directory does not exist: ${SLICES_DIR}"
        echo "       Run preprocessing first or check --skip-preprocessing flag."
        exit 1
    fi

    # Step 4: Model inference
    run_step "4. Running CUT model inference" \
        env CUDA_VISIBLE_DEVICES="${GPU}" \
        conda run -n "${MODEL_ENV}" python \
            "${TRAINING_DIR}/inference_synth.py" \
            --dataroot "${SLICES_DIR}" \
            --name "${MODEL_NAME}" \
            --checkpoints_dir "${CHECKPOINT_DIR}" \
            --epoch "${EPOCH}" \
            --results_dir "${DIR_INFERENCE}" \
            --model cut \
            --input_nc 1 \
            --output_nc 1 \
            --preprocess none \
            --no_flip \
            --eval

else
    echo ""
    echo "SKIPPING: Model inference (--skip-inference)"

    # Copy model inputs to where model outputs would be, so reconstruction can proceed
    SLICES_DIR="${DIR_SLICES}/model_2d/full/A"
    FAKE_NIFTI_DIR="${DIR_INFERENCE}/${MODEL_NAME}/test_${EPOCH}/fake_nifti"
    mkdir -p "${FAKE_NIFTI_DIR}"
    echo "Copying input slices to fake_nifti directory (renaming - to _)..."
    for f in "${SLICES_DIR}"/*.nii*; do
        [[ -f "${f}" ]] || continue
        dest_name="$(basename "${f}" | tr '-' '_')"
        cp -p "${f}" "${FAKE_NIFTI_DIR}/${dest_name}"
    done
    echo "  ${SLICES_DIR} -> ${FAKE_NIFTI_DIR}"
fi

# =========================================================================
# VOLUME RECONSTRUCTION
# =========================================================================

if [[ "${SKIP_RECONSTRUCTION}" == false ]]; then

    FAKE_DIR="${DIR_INFERENCE}/${MODEL_NAME}/test_${EPOCH}"

    if [[ ! -d "${FAKE_DIR}/fake_nifti" ]]; then
        echo "ERROR: Fake slices directory does not exist: ${FAKE_DIR}/fake_nifti"
        echo "       Run inference first or check --skip-inference flag."
        exit 1
    fi

    # Step 5: Reconstruct 3D volumes
    run_step "5. Reconstructing 3D volumes" \
        conda run -n "${PREPROC_ENV}" python \
            "${PREPROC_SCRIPTS}/81sct_volume_reconstructor.py" \
            "${MODEL_NAME}/test_${EPOCH}" \
            --test-images-dir "${DIR_INFERENCE}" \
            --init-dir "${INPUT_DIR}" \
            --resampled-dir "${DIR_RESAMPLED}" \
            --use-mr-reference

    # Copy final outputs to reconstruction directory
    RECONSTRUCTION_SRC="${FAKE_DIR}/reconstruction"
    if [[ -d "${RECONSTRUCTION_SRC}" ]]; then
        echo ""
        echo "Copying final outputs to reconstruction directory..."
        for patient_dir in "${RECONSTRUCTION_SRC}"/*/; do
            [[ -d "${patient_dir}" ]] || continue
            patient_name="$(basename "${patient_dir}")"
            dest="${DIR_RECONSTRUCTION}/${patient_name}"
            mkdir -p "${dest}"
            for f in "${patient_dir}"*.nii*; do
                [[ -f "${f}" ]] || continue
                cp -p "${f}" "${dest}/"
                echo "  Copied: $(basename "${f}") -> ${dest}"
            done
        done
    fi

else
    echo ""
    echo "SKIPPING: Volume reconstruction (--skip-reconstruction)"
fi

# =========================================================================
# STEP 6: COPY RESULTS TO OUTPUT (10reconstruction -> GC output)
# =========================================================================

run_step --no-cmd "6. Copying results to output" \
    conda run -n "${PREPROC_ENV}" python -c "
import json, shutil, os, sys
import SimpleITK as sitk

gc_input = sys.argv[1]
gc_output = sys.argv[2]
reconstruction_dir = sys.argv[3]
init_dir = sys.argv[4]

# Read input mapping (patient_id -> original filename) saved in Step 0
mapping_path = os.path.join(init_dir, 'input_mapping.json')
with open(mapping_path) as f:
    input_mapping = json.load(f)

# Output goes to {gc_output}/images/synthetic-ct/{original_filename}
output_dir = os.path.join(gc_output, 'images', 'synthetic-ct')
os.makedirs(output_dir, exist_ok=True)

case_results = []

for patient_id, original_basename in sorted(input_mapping.items()):
        src = os.path.join(reconstruction_dir, patient_id, 'sCT_original_dim_reconstructed_alignment.nii.gz')
        if not os.path.exists(src):
            print(f'ERROR: Reconstructed file not found: {src}')
            sys.exit(1)

    # Output uses the original filename (preserving .mha / .nii / .nii.gz)
    output_path = os.path.join(output_dir, original_basename)

    if original_basename.endswith('.nii.gz'):
        shutil.copy2(src, output_path)
    else:
        # Convert from .nii.gz back to original format (.mha, .nii)
        img = sitk.ReadImage(src)
        # Strip NIfTI-specific metadata that MetaImage format does not support
        for key in ['ITK_FileNotes', 'aux_file', 'descrip', 'intent_name']:
            if img.HasMetaDataKey(key):
                img.EraseMetaData(key)
        sitk.WriteImage(img, output_path)

        print(f'  {src} -> {output_path}')

    case_results.append({
        'outputs': [{'type': 'metaio_image', 'filename': output_path}],
        'inputs': [
            {'type': 'metaio_image', 'filename': os.path.join(gc_input, 'images', 'mri', original_basename)},
        ],
        'error_messages': [],
    })

# Write results.json
results_path = os.path.join(gc_output, 'results.json')
with open(results_path, 'w') as f:
    json.dump(case_results, f)
print(f'  Results written to {results_path}')
" "${GC_INPUT_DIR}" "${GC_OUTPUT_DIR}" "${DIR_RECONSTRUCTION}" "${INPUT_DIR}"

# =========================================================================
# SUMMARY
# =========================================================================

echo ""
echo "======================================================================"
echo "PIPELINE COMPLETE"
echo "======================================================================"
echo ""
echo "Output directories:"
echo "  Resampled MR:     ${DIR_RESAMPLED}"
echo "  Normalized MR:    ${DIR_NORMALIZED}"
echo "  MR slices:        ${DIR_SLICES}/model_2d/full/A"
echo "  Fake CT slices:   ${DIR_INFERENCE}/${MODEL_NAME}/test_${EPOCH}/fake_nifti"
echo "  Final sCT:        ${DIR_RECONSTRUCTION}"
echo "  GC Output:        ${GC_OUTPUT_DIR}"
echo ""
echo "======================================================================"
