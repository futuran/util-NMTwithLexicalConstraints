CORPUS=$1
if [ "$CORPUS" = "" ]; then
    echo "Please designate CORPUS!!!"
    exit
fi

DIR=../../data/${CORPUS}
mkdir -p ${DIR}.ed/match_trgtrg

for prefix in test dev train_h40000 train ; do
    python 01.search_ed.py  -s ${DIR}/${CORPUS}_train.en.tkn \
                            -t ${DIR}/${CORPUS}_train.ja.tkn \
                            -q ${DIR}/${CORPUS}_$prefix.ja.tkn \
                            -o ${DIR}.ed/match_trgtrg/${prefix}.match \
                            -k 100
    wait
done
