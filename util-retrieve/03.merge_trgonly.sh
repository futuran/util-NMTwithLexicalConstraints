CORPUS=$1
DIR=../../data/${CORPUS}

# 3 merge
for k in 1 10; do
    mkdir ${DIR}.sbert_trgonly/merge.top$k
    for prefix in test dev train_h40000 ; do
        python 03.merge.py \
            -s ${DIR}/${CORPUS}_$prefix.en.tkn \
            -t ${DIR}/${CORPUS}_$prefix.ja.tkn \
            --match-file ${DIR}.sbert_trgonly/match/${prefix}.match \
            -tmt ${DIR}/${CORPUS}_train.ja.tkn \
            -o   ${DIR}.sbert_trgonly/merge.top$k/ \
            --topk $k --threshold 0.00 \
            --concat-symbol ' | '
        wait
    done
done

