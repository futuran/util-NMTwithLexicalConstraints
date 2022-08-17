CORPUS=$1
if [ "$CORPUS" = "" ]; then
    echo "Please designate CORPUS!!!"
    exit
fi

DIR=../../data/${CORPUS}

# 1 cal embs
mkdir -p ${DIR}.labse_wo_normalize/emb
mkdir -p ${DIR}.msbert_wo_normalize/emb
mkdir -p ${DIR}.sbert_wo_normalize/emb
mkdir -p ${DIR}.sbert2_wo_normalize/emb
# mkdir -p ${DIR}.msbert_wo_normalize_finetuned/emb

for prefix in test valid train ; do
    # python 01.cal-emb.py    -s  ${DIR}/${CORPUS}_$prefix.en.tkn \
    #                         -t  ${DIR}/${CORPUS}_$prefix.fr.tkn \
    #                         -so ${DIR}.labse_wo_normalize/emb/${prefix}.en.emb \
    #                         -to ${DIR}.labse_wo_normalize/emb/${prefix}.fr.emb \
    #                         --model-dir 'LaBSE' 

    # python 01.cal-emb.py    -s  ${DIR}/${CORPUS}_$prefix.en.tkn \
    #                         -t  ${DIR}/${CORPUS}_$prefix.fr.tkn \
    #                         -so ${DIR}.msbert_wo_normalize/emb/${prefix}.en.emb \
    #                         -to ${DIR}.msbert_wo_normalize/emb/${prefix}.fr.emb \
    #                         --model-dir 'paraphrase-multilingual-mpnet-base-v2' 

    # python 01.cal-emb.py    -s  ${DIR}/${CORPUS}_$prefix.en.tkn \
    #                         -t  ${DIR}/${CORPUS}_$prefix.fr.tkn \
    #                         -so ${DIR}.sbert_wo_normalize/emb/${prefix}.en.emb \
    #                         -to ${DIR}.sbert_wo_normalize/emb/${prefix}.fr.emb \
    #                         --model-dir 'all-mpnet-base-v2' 
    
    python 01.cal-emb.py    -s  ${DIR}/${CORPUS}_$prefix.en.tkn \
                            -t  ${DIR}/${CORPUS}_$prefix.fr.tkn \
                            -so ${DIR}.sbert2_wo_normalize/emb/${prefix}.en.emb \
                            -to ${DIR}.sbert2_wo_normalize/emb/${prefix}.fr.emb \
                            --model-dir 'multi-qa-mpnet-base-dot-v1' 

    # python 01.cal-emb.py    -s  ${DIR}/${CORPUS}_$prefix.en.tkn \
    #                         -t  ${DIR}/${CORPUS}_$prefix.fr.tkn \
    #                         -so ${DIR}.msbert_wo_normalize_finetuned/emb/${prefix}.en.emb \
    #                         -to ${DIR}.msbert_wo_normalize_finetuned/emb/${prefix}.fr.emb \
    #                         --model-dir '/mnt/work/20220419_basic_trasformer/sbert/out' \
    #                         --normalize-vectors
    wait
done
