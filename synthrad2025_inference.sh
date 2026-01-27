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

if [[ ! -f "${GC_INPUT_DIR}/expected_output.json" ]]; then
    echo "ERROR: expected_output.json not found in: ${GC_INPUT_DIR}"
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
    python3 -c "
import json, shutil, os, sys

gc_input = sys.argv[1]
init_dir = sys.argv[2]

# Read region
with open(os.path.join(gc_input, 'region.json')) as f:
    region = json.load(f)

prefix_map = {'Head and Neck': 'HN', 'Abdomen': 'AB', 'Thorax': 'TH'}
prefix = prefix_map.get(region)
if not prefix:
    print(f'ERROR: Unknown region: {region}')
    sys.exit(1)

print(f'Region: {region} -> Prefix: {prefix}')

# Read expected_output.json
with open(os.path.join(gc_input, 'expected_output.json')) as f:
    entries = json.load(f)

for entry in entries:
    mri_rel = None
    body_rel = None
    for inp in entry.get('inputs', []):
        if '/mri/' in inp['filename']:
            mri_rel = inp['filename']
        elif '/body/' in inp['filename']:
            body_rel = inp['filename']

    if not mri_rel:
        print('ERROR: No MRI input found in expected_output.json')
        sys.exit(1)

    # Extract stem from filename (e.g., '1HNAxxx' from '/input/images/mri/1HNAxxx.nii.gz')
    basename = os.path.basename(mri_rel)
    stem = basename.replace('.nii.gz', '') if basename.endswith('.nii.gz') else basename.replace('.nii', '')
    patient_id = f'{prefix}_{stem}'

    # Create directory structure
    mr_dir = os.path.join(init_dir, patient_id, 'MR')
    mask_dir = os.path.join(init_dir, patient_id, 'masks')
    os.makedirs(mr_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # Copy MRI: /input/images/mri/X.nii.gz -> 1initNifti/{PREFIX}_{STEM}/MR/{PREFIX}_{STEM}_MR.nii.gz
    src_mri = gc_input + mri_rel[len('/input'):]
    dst_mri = os.path.join(mr_dir, f'{patient_id}_MR.nii.gz')
    shutil.copy2(src_mri, dst_mri)
    print(f'  MRI:  {src_mri} -> {dst_mri}')

    # Copy body mask if available
    if body_rel:
        src_body = gc_input + body_rel[len('/input'):]
        dst_body = os.path.join(mask_dir, f'{patient_id}_mask.nii.gz')
        shutil.copy2(src_body, dst_body)
        print(f'  Mask: {src_body} -> {dst_body}')
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
    python3 -c "
import json, shutil, os, sys

gc_input = sys.argv[1]
gc_output = sys.argv[2]
reconstruction_dir = sys.argv[3]

# Read region for prefix
with open(os.path.join(gc_input, 'region.json')) as f:
    region = json.load(f)

prefix_map = {'Head and Neck': 'HN', 'Abdomen': 'AB', 'Thorax': 'TH'}
prefix = prefix_map.get(region)
if not prefix:
    print(f'ERROR: Unknown region: {region}')
    sys.exit(1)

# Read expected_output.json
with open(os.path.join(gc_input, 'expected_output.json')) as f:
    entries = json.load(f)

for entry in entries:
    # Find MRI input to derive patient ID
    mri_rel = None
    for inp in entry.get('inputs', []):
        if '/mri/' in inp['filename']:
            mri_rel = inp['filename']
            break

    if not mri_rel:
        print('ERROR: No MRI input found in expected_output.json')
        sys.exit(1)

    basename = os.path.basename(mri_rel)
    stem = basename.replace('.nii.gz', '') if basename.endswith('.nii.gz') else basename.replace('.nii', '')
    patient_id = f'{prefix}_{stem}'

    # Find output path(s) and copy reconstructed sCT
    for out in entry.get('outputs', []):
        output_rel = out['filename']  # e.g., /output/images/synthetic-ct/1HNAxxx.nii.gz
        output_path = gc_output + output_rel[len('/output'):]

        src = os.path.join(reconstruction_dir, patient_id, 'sCT_original_dim_reconstructed_alignment.nii.gz')
        if not os.path.exists(src):
            print(f'ERROR: Reconstructed file not found: {src}')
            sys.exit(1)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.copy2(src, output_path)
        print(f'  {src} -> {output_path}')
" "${GC_INPUT_DIR}" "${GC_OUTPUT_DIR}" "${DIR_RECONSTRUCTION}"

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
