CORPUS=$1
if [ "$CORPUS" = "" ]; then
    echo "Please designate CORPUS!!!"
    exit
fi

DIR=../../data/${CORPUS}

# 1 cal embs
mkdir -p ${DIR}.labse/emb
mkdir -p ${DIR}.msbert/emb

for prefix in test dev train_h40000 train ; do
    python 01.cal-emb.py    -s  ${DIR}/${CORPUS}_$prefix.en.tkn \
                            -t  ${DIR}/${CORPUS}_$prefix.ja.tkn \
                            -so ${DIR}.msbert/emb/${prefix}.en.emb \
                            -to ${DIR}.msbert/emb/${prefix}.ja.emb \
                            --model-dir 'LaBSE' \
                            --normalize-vectors

    python 01.cal-emb.py    -s  ${DIR}/${CORPUS}_$prefix.en.tkn \
                            -t  ${DIR}/${CORPUS}_$prefix.ja.tkn \
                            -so ${DIR}.labse/emb/${prefix}.en.emb \
                            -to ${DIR}.labse/emb/${prefix}.ja.emb \
                            --model-dir 'paraphrase-multilingual-mpnet-base-v2' \
                            --normalize-vectors
    wait
done