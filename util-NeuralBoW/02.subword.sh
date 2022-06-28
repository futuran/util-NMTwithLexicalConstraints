CORPUS=$1
if [ "$CORPUS" = "" ]; then
    echo "Please designate CORPUS!!!"
    exit
fi
DIR=../../data/${CORPUS}
MODEL=$2
if [ "$MODEL" = "" ]; then
    echo "Please designate MODEL!!!"
    exit
fi


code=${DIR}/code.all

for topk in 1 10 100; do
    dir=${DIR}.$MODEL.oracle/srctrg_top$topk/
    for tvt in train_h40000 dev test; do
            subword-nmt apply-bpe -c $code < $dir/${CORPUS}_$tvt.oracle.src.tkn > $dir/${CORPUS}_$tvt.oracle.src.tkn.bpe &
            subword-nmt apply-bpe -c $code < $dir/${CORPUS}_$tvt.oracle.trg.tkn > $dir/${CORPUS}_$tvt.oracle.trg.tkn.bpe &
    done
done
