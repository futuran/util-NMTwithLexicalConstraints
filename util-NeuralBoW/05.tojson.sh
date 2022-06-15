CORPUS=$1
if [ "$CORPUS" = "" ]; then
    echo "Please designate CORPUS!!!"
    exit
fi
MODEL=$2
if [ "$MODEL" = "" ]; then
    echo "Please designate MODEL!!!"
    exit
fi

dir_oracle=../../data/${CORPUS}.$MODEL.oracle/srctrg_top10
dir_assisted=../../experiments/nbow.${CORPUS}_h40000.$MODEL.srctrg_top10

for tvt in train_h40000 dev test;
do
    python ./05.tojson.py -src $dir_oracle/${CORPUS}_$tvt.oracle.src.tkn \
                          -trg $dir_oracle/${CORPUS}_$tvt.oracle.trg.tkn \
                          -out $dir_oracle/${CORPUS}_$tvt.oracle.tkn.json
done

# python ./05.tojson.py -src $dir_assisted/${CORPUS}_test.assisted.src.tkn \
#                       -trg $dir_oracle/${CORPUS}_test.oracle.trg.tkn \
#                       -out $dir_oracle/${CORPUS}_test.assisted.tkn.json

python ./05.tojson.py -src $dir_assisted/${CORPUS}_test.assisted.src.tkn \
                      -trg $dir_oracle/${CORPUS}_test.oracle.trg.tkn \
                      -out $dir_assisted/${CORPUS}_test.assisted.tkn.json

for TOPK in `seq 1 100`; do
    python ./05.tojson.py -src $dir_assisted/topk/${CORPUS}_test.assisted.src.tkn.with_upto_No.$TOPK \
                            -trg $dir_oracle/${CORPUS}_test.oracle.trg.tkn \
                            -out $dir_assisted/topk/${CORPUS}_test.assisted.src.tkn.with_upto_No.$TOPK.json
done