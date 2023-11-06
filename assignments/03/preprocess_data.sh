#!/bin/bash
# -*- coding: utf-8 -*-

set -e

pwd=`dirname "$(readlink -f "$0")"`
base=$pwd/../..
src=fr
tgt=en
data=$base/data/$tgt-$src/

# change into base directory to ensure paths are valid
cd $base

# create preprocessed directory
mkdir -p $data/preprocessed/

# normalize and tokenize raw data
cat $data/raw/train.$src | perl moses_scripts/normalize-punctuation.perl -l $src | perl moses_scripts/tokenizer.perl -l $src -a -q > $data/preprocessed/train.$src.p
cat $data/raw/train.$tgt | perl moses_scripts/normalize-punctuation.perl -l $tgt | perl moses_scripts/tokenizer.perl -l $tgt -a -q > $data/preprocessed/train.$tgt.p

# train truecase models
perl moses_scripts/train-truecaser.perl --model $data/preprocessed/tm.$src --corpus $data/preprocessed/train.$src.p
perl moses_scripts/train-truecaser.perl --model $data/preprocessed/tm.$tgt --corpus $data/preprocessed/train.$tgt.p

# apply truecase models to splits
cat $data/preprocessed/train.$src.p | perl moses_scripts/truecase.perl --model $data/preprocessed/tm.$src > $data/preprocessed/train.$src 
cat $data/preprocessed/train.$tgt.p | perl moses_scripts/truecase.perl --model $data/preprocessed/tm.$tgt > $data/preprocessed/train.$tgt

# prepare remaining splits with learned models
for split in valid test tiny_train
do
    cat $data/raw/$split.$src | perl moses_scripts/normalize-punctuation.perl -l $src | perl moses_scripts/tokenizer.perl -l $src -a -q | perl moses_scripts/truecase.perl --model $data/preprocessed/tm.$src > $data/preprocessed/$split.$src
    cat $data/raw/$split.$tgt | perl moses_scripts/normalize-punctuation.perl -l $tgt | perl moses_scripts/tokenizer.perl -l $tgt -a -q | perl moses_scripts/truecase.perl --model $data/preprocessed/tm.$tgt > $data/preprocessed/$split.$tgt
done

# remove tmp files
rm $data/preprocessed/train.$src.p
rm $data/preprocessed/train.$tgt.p

# preprocess all files for model training
# python preprocess_bpe_autoencoder.py --target-lang $tgt --source-lang $src --dest-dir $data/prepared/ --train-prefix $data/preprocessed/train --valid-prefix $data/preprocessed/valid --test-prefix $data/preprocessed/test --tiny-train-prefix $data/preprocessed/tiny_train --input $data/preprocessed/train.fr $data/preprocessed/train.en --output $data/preprocessed/bpe_codes --write-vocabulary $data/preprocessed/vocab.fr $data/preprocessed/vocab.en --threshold-src 1 --threshold-tgt 1 --num-words-src 4000 --num-words-tgt 4000 --symbols 10000 --train-bpe --dropout 0.1 --num-workers 16
python preprocess_bpe_autoencoder.py --target-lang $tgt --source-lang $src --dest-dir $data/prepared/ --train-prefix $data/preprocessed/train --valid-prefix $data/preprocessed/valid --test-prefix $data/preprocessed/test --tiny-train-prefix $data/preprocessed/tiny_train --input $data/preprocessed/train.fr $data/preprocessed/train.en --output $data/preprocessed/bpe_codes --output-bpe $data/preprocessed/train.bpe.fr $data/preprocessed/train.bpe.en --vocabulary $data/preprocessed/vocab.fr $data/preprocessed/vocab.en --write-vocabulary $data/preprocessed/vocab.fr $data/preprocessed/vocab.en --threshold-src 1 --threshold-tgt 1 --num-words-src 4000 --num-words-tgt 4000 --symbols 10000 --train-bpe --dropout 0.1 --num-workers 16

echo "done!"


# python preprocess_bpe_autoencoder.py \
# --target-lang fr \
# --source-lang en \
# --dest-dir data/en-fr/prepared/ \
# --train-prefix data/en-fr/raw/train \
# --valid-prefix data/en-fr/raw/valid \
# --test-prefix data/en-fr/raw/test \
# --tiny-train-prefix data/en-fr/raw/tiny_train \
# --input data/en-fr/raw/train.fr data/en-fr/raw/train.en \
# --output data/en-fr/raw/bpe_codes \
# --output-bpe data/en-fr/raw/train.bpe.fr data/en-fr/raw/train.bpe.en \
# --vocabulary data/en-fr/raw/vocab.fr data/en-fr/raw/vocab.en \
# --write-vocabulary data/en-fr/raw/vocab.fr data/en-fr/raw/vocab.en \
# --threshold-src 1 --threshold-tgt 1 \
# --num-words-src 4000 --num-words-tgt 4000 \
# --symbols 4000 --train-bpe --dropout 0.1 --num-workers 16