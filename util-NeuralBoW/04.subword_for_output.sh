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

dir=../../experiments/nbow.${CORPUS}_h40000.$MODEL.srcsrc_top10
subword-nmt apply-bpe -c $code < $dir/${CORPUS}_test.assisted.src.tkn > $dir/${CORPUS}_test.assisted.src.tkn.bpe
for topk in `seq 1 100`; do
    subword-nmt apply-bpe -c $code < $dir/topk/${CORPUS}_test.assisted.src.tkn.with_upto_No.$topk > $dir/topk/${CORPUS}_test.assisted.src.tkn.with_upto_No.$topk.bpe &
done