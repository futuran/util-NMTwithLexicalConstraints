TYPE=""
TOP100="_top1"
CORPUS=$1
DIR=../../data/${CORPUS}

code=${DIR}/code.all
dir=../../experiments/nbow.${CORPUS}_h40000.sbert$TYPE
for tvt in test; do
        subword-nmt apply-bpe -c $code < $dir/${CORPUS}_$tvt.assisted$TOP100.src.tkn > $dir/${CORPUS}_$tvt.assisted$TOP100.src.tkn.bpe
        
done
