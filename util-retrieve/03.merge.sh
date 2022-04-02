CORPUS=$1
DIR=../../data/${CORPUS}

# 3 merge
for k in 1 10; do
    mkdir ${DIR}.sbert/merge.top$k
    for prefix in test dev train_h40000 train ; do
        python 03.merge.py \
            -s ${DIR}/${COURPUS}_$prefix.en.tkn \
            -t ${DIR}/${COURPUS}_$prefix.ja.tkn \
            --match-file ${DIR}.sbert/match/${prefix}.match \
            -tmt ${DIR}/${COURPUS}_train.ja.tkn \
            -o   ${DIR}.sbert/merge.top$k/ \
            --topk $k --threshold 0.00 \
            --concat-symbol ' | '
        wait
    done
done

