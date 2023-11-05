python train.py \
    --data data/en-sv/infopankki/prepared \
    --source-lang sv \
    --target-lang en \
    --save-dir assignments/01/baseline/checkpoints \
    --cuda
    
    
python translate.py \
    --data data/en-sv/infopankki/prepared \
    --dicts data/en-sv/infopankki/prepared \
    --checkpoint-path assignments/01/baseline/checkpoints/checkpoint_last.pt \
    --output assignments/01/baseline/infopankki_translations.txt \
    --cuda

bash scripts/postprocess.sh \
    assignments/01/baseline/infopankki_translations.txt \
    assignments/01/baseline/infopankki_translations.p.txt en

cat \
    assignments/01/baseline/infopankki_translations.p.txt \
    | sacrebleu data/en-sv/infopankki/raw/test.en