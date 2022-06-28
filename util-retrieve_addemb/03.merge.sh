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

# 3 merge
for TYPE in _srcsrc _srctrg; do
    for k in 1 10 100; do
        mkdir ${DIR}.$MODEL/merge$TYPE.top$k
        for prefix in test dev train_h40000 ; do
            python 03.merge.py \
                -s  ${DIR}/${CORPUS}_$prefix.en.tkn \
                -se ${DIR}.$MODEL/emb/$prefix.en.emb \
                -t ${DIR}/${CORPUS}_$prefix.ja.tkn \
                -te ${DIR}.$MODEL/emb/$prefix.ja.emb \
                --match-file ${DIR}.$MODEL/match$TYPE/${prefix}.match \
                -tmt ${DIR}/${CORPUS}_train.ja.tkn \
                -o   ${DIR}.$MODEL/merge$TYPE.top$k/ \
                --topk $k --threshold 0.00 \
                --concat-symbol ' | '
            wait
        done
    done
done
