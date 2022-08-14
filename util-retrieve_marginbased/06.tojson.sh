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

for TYPE in _srcsrc _srctrg; do
    dir=${DIR}.$MODEL/merge$TYPE.top1
    for tvt in train_h40000 dev test;
    do
        python ./05.tojson.py -src $dir/${CORPUS}_$tvt.en.with_match.tkn \
                            -trg $dir/${CORPUS}_$tvt.ja.with_match.tkn \
                            -out $dir/${CORPUS}_$tvt.tkn.json
    done
done