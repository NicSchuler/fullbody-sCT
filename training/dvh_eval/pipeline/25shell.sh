#python 25prepare_nifti_export_for_slicer.py \
#  --sct-base /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/9latestTestImages \
#  --export-root /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/11dvhEvalCases \
#  --model-name 2_experiment_cut_synthrad_abdomen_32p99 \
#  --epoch 50



folders=(
    #2_experiment_cut_synthrad_abdomen_32p99
    2_experiment_cut_synthrad_abdomen_33nyul
    2_experiment_cut_synthrad_abdomen_sep_first_layer
    2_experiment_cyclegan_abdomen_32p99
    2_experiment_cyclegan_abdomen_33nyul
    2_experiment_cyclegan_abdomen_sep_first_layer
    2_experiment_pix2pix_synthrad_abdomen_32p99
    2_experiment_pix2pix_synthrad_abdomen_33nyul
    2_experiment_pix2pix_synthrad_abdomen_sep_input_layers
)
for f in "${folders[@]}"; do
    echo "Running: $f"
    python 25prepare_nifti_export_for_slicer.py \
      --sct-base /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/9latestTestImages \
      --export-root /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/11dvhEvalCases \
      --model-name $f \
      --epoch 50
done

