CORPUS=$1
if [ "$CORPUS" = "" ]; then
    echo "Please designate CORPUS!!!"
    exit
fi

DIR=../../data/${CORPUS}

# 1 cal embs
mkdir -p ${DIR}.labse/emb
# mkdir -p ${DIR}.msbert/emb
# mkdir -p ${DIR}.sbert/emb
# mkdir -p ${DIR}.msbert_finetuned/emb

for prefix in test valid train ; do
    python 01.cal-emb.py    -s  ${DIR}/${CORPUS}_$prefix.en.tkn \
                            -t  ${DIR}/${CORPUS}_$prefix.fr.tkn \
                            -so ${DIR}.labse/emb/${prefix}.en.emb \
                            -to ${DIR}.labse/emb/${prefix}.fr.emb \
                            --model-dir 'LaBSE' \
                            --normalize-vectors

    # python 01.cal-emb.py    -s  ${DIR}/${CORPUS}_$prefix.en.tkn \
    #                         -t  ${DIR}/${CORPUS}_$prefix.fr.tkn \
    #                         -so ${DIR}.msbert/emb/${prefix}.en.emb \
    #                         -to ${DIR}.msbert/emb/${prefix}.fr.emb \
    #                         --model-dir 'paraphrase-multilingual-mpnet-base-v2' \
    #                         --normalize-vectors

    # python 01.cal-emb.py    -s  ${DIR}/${CORPUS}_$prefix.en.tkn \
    #                         -t  ${DIR}/${CORPUS}_$prefix.fr.tkn \
    #                         -so ${DIR}.sbert/emb/${prefix}.en.emb \
    #                         -to ${DIR}.sbert/emb/${prefix}.fr.emb \
    #                         --model-dir 'all-mpnet-base-v2' \
    #                         --normalize-vectors

    # python 01.cal-emb.py    -s  ${DIR}/${CORPUS}_$prefix.en.tkn \
    #                         -t  ${DIR}/${CORPUS}_$prefix.fr.tkn \
    #                         -so ${DIR}.msbert_finetuned/emb/${prefix}.en.emb \
    #                         -to ${DIR}.msbert_finetuned/emb/${prefix}.fr.emb \
    #                         --model-dir '/mnt/work/20220419_basic_trasformer/sbert/out' \
    #                         --normalize-vectors
    wait
done
