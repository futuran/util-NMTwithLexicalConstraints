CORPUS=$1
if [ "$CORPUS" = "" ]; then
    echo "Please designate CORPUS!!!"
    exit
fi

DIR=../../data/${CORPUS}
mkdir -p ${DIR}.ed/match_srcsrc

for prefix in valid ; do
    python 01.search_ed.py  -s ${DIR}/${CORPUS}_train.en.tkn \
                            -t ${DIR}/${CORPUS}_train.fr.tkn \
                            -q ${DIR}/${CORPUS}_$prefix.en.tkn \
                            -o ${DIR}.ed/match_srcsrc/${prefix}.match \
                            -k 100
    wait
done
