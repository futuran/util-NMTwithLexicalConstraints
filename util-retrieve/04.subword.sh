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

for TYPE in _srcsrc ; do
    for k in 1 10 100; do
        dir=${DIR}.$MODEL/merge$TYPE.top$k
        for prefix in test valid ; do
            subword-nmt apply-bpe -c $code < $dir/${CORPUS}_$prefix.en.with_match.tkn > $dir/${CORPUS}_$prefix.en.tkn.bpe
            subword-nmt apply-bpe -c $code < $dir/${CORPUS}_$prefix.fr.with_match.tkn > $dir/${CORPUS}_$prefix.fr.tkn.bpe
        done
    done
done
