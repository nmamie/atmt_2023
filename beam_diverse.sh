#!/bin/bash

# Define paths
data_path="data/en-fr/prepared"
dicts_path="data/en-fr/prepared"
checkpoint_path="assignments/03/baseline/checkpoints_bpe/checkpoint_best.pt"
output_dir="assignments/05/model_translations_beam_diverse"
test_en="data/en-fr/raw/test.en"

# Ensure the output directory exists
mkdir -p $output_dir

# Best beam size is 3
beam_size=3

# Loop over lmdba values between 0.2 and 2 in steps of 0.2
for lmbda in $(seq 0.2 0.2 2)

do
    echo "Processing with beam size $beam_size and lambda $lmbda"  

    # Perform the translation
    python translate_beam_diverse.py \
        --data $data_path \
        --dicts $dicts_path \
        --checkpoint-path $checkpoint_path \
        --beam-size $beam_size \
        --output $output_dir/model_translations_beam_${beam_size}_diverse_${lmbda} \
        --lmbda $lmbda

    # Post-process
    bash scripts/postprocess.sh $output_dir/model_translations_beam_${beam_size}_diverse_${lmbda} ${output_dir}/model_translations_beam_${beam_size}_diverse_${lmbda}_post en
done

echo "Translation and evaluation completed."
