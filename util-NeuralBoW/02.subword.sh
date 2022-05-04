TYPE=""
CORPUS=$1
DIR=../../data/${CORPUS}

code=${DIR}/code.all
dir=${DIR}.sbert$TYPE.oracle/
for tvt in train_h40000 dev test; do
        subword-nmt apply-bpe -c $code < $dir/${CORPUS}_$tvt.oracle.src.tkn > $dir/${CORPUS}_$tvt.oracle.src.tkn.bpe
        subword-nmt apply-bpe -c $code < $dir/${CORPUS}_$tvt.oracle.trg.tkn > $dir/${CORPUS}_$tvt.oracle.trg.tkn.bpe
done
