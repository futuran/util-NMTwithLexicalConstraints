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
for TYPE in _srcsrc; do
    for k in 1 10 100; do
        mkdir ${DIR}.$MODEL/merge$TYPE.top$k
        for prefix in test valid train ; do
            python 03.merge.py \
                -s ${DIR}/${CORPUS}_$prefix.en.tkn \
                -t ${DIR}/${CORPUS}_$prefix.fr.tkn \
                --match-file ${DIR}.$MODEL/match$TYPE/${prefix}.match \
                -tmt ${DIR}/${CORPUS}_train.fr.tkn \
                -o   ${DIR}.$MODEL/merge$TYPE.top$k/ \
                --topk $k --threshold 0.00 \
                --concat-symbol ' | '
            wait
        done
    done
done
