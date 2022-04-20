CORPUS=$1
DIR=../../data/${CORPUS}

code=${DIR}/code.all
dir=${DIR}.sbert_trgonly/merge.top1/
for tvt in train_h40000 dev test; do
        subword-nmt apply-bpe -c $code < $dir/${CORPUS}_$tvt.en.with_match.tkn > $dir/${CORPUS}_$tvt.en.tkn.bpe
        subword-nmt apply-bpe -c $code < $dir/${CORPUS}_$tvt.ja.with_match.tkn > $dir/${CORPUS}_$tvt.ja.tkn.bpe
done
