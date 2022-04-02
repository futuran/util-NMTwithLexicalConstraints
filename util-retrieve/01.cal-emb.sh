CORPUS=$1
DIR=../../data/${CORPUS}

# 1 cal embs
mkdir -p ${DIR}.sbert/emb
for prefix in test dev train train_h40000 ; do
    python 01.cal-emb.py    -s  ${DIR}/${CORPUS}_$prefix.en.tkn \
                            -t  ${DIR}/${CORPUS}_$prefix.ja.tkn \
                            -so ${DIR}.sbert/emb/${prefix}.en.emb \
                            -to ${DIR}.sbert/emb/${prefix}.ja.emb \
                            --model-dir 'paraphrase-multilingual-mpnet-base-v2' \
                            --normalize-vectors
    wait
done