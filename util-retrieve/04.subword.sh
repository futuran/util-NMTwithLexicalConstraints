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

for TYPE in _srcsrc _srctrg; do
    for k in 1 10 100; do
        dir=${DIR}.$MODEL/merge$TYPE.top$k
        for prefix in test dev train_h40000 ; do
            subword-nmt apply-bpe -c $code < $dir/${CORPUS}_$prefix.en.with_match.tkn > $dir/${CORPUS}_$prefix.en.tkn.bpe
            subword-nmt apply-bpe -c $code < $dir/${CORPUS}_$prefix.ja.with_match.tkn > $dir/${CORPUS}_$prefix.ja.tkn.bpe
        done
    done
done
