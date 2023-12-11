#!/bin/bash

# Define paths
data_path="data/en-fr/prepared"
dicts_path="data/en-fr/prepared"
checkpoint_path="assignments/03/baseline/checkpoints_bpe/checkpoint_best.pt"
output_dir="assignments/05/model_translations_beam"
test_en="data/en-fr/raw/test.en"

# Ensure the output directory exists
mkdir -p $output_dir

# File to save BLEU scores
scores_file="bleu_scores.txt"
echo "Beam Size, BLEU Score" > $scores_file

# # preprocess the data
# bash assignments/03/preprocess_data.sh

# Loop over beam sizes from 1 to 25
for beam_size in {1..25}
do
    echo "Processing with beam size $beam_size"    

    # Perform the translation
    python translate_beam.py \
      --data $data_path \
      --dicts $dicts_path \
      --checkpoint-path $checkpoint_path \
      --beam-size $beam_size \
      --output $output_dir/model_translations_beam_$beam_size \
      --cuda

    # Post-process
    bash scripts/postprocess.sh $output_dir/model_translations_beam_$beam_size ${output_dir}/model_translations_beam_${beam_size}_post en

    # Evaluate with sacrebleu and parse with jq
    score=$(cat ${output_dir}/model_translations_beam_${beam_size}_post | sacrebleu $test_en --format=json | jq .score)

    # Save the score
    echo "$beam_size, $score" >> $scores_file
done

echo "Translation and evaluation completed. Scores saved in $scores_file."
