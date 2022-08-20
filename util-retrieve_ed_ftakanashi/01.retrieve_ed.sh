CORPUS=$1
if [ "$CORPUS" = "" ]; then
    echo "Please designate CORPUS!!!"
    exit
fi

DIR=../../data/${CORPUS}
mkdir -p ${DIR}.ed/match_srcsrc

for prefix in train ; do

    if [ $prefix = 'test' ]; then
        PM='--include-perfect-match'
    else
        PM=''
    fi
    python 01.retrieve_ed.py  -q ${DIR}/${CORPUS}_${prefix}.en.tkn \
                            -tms ${DIR}/${CORPUS}_train.en.tkn \
                            -o ${DIR}.ed/match_srcsrc/${prefix}.match \
                            -sss_lambda 0.0 \
                            --ed_lambda 0.0 \
                            --format-n 100 $PM
    wait
done
