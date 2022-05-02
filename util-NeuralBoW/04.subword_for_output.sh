CORPUS=$1
DIR=../../data/${CORPUS}

code=${DIR}/code.all
dir=../../experiments/nbow.${CORPUS}_h40000.sbert_srconly_top100
for tvt in test; do
        subword-nmt apply-bpe -c $code < $dir/${CORPUS}_$tvt.assisted.src.tkn > $dir/${CORPUS}_$tvt.assisted.src.tkn.bpe
        
done
