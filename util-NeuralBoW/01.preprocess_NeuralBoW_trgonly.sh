# XLM-RoBERTaに基づくフレーズ抽出モデル訓練のためのデータセット作成
CORPUS=$1
DIR=../../data/${CORPUS}

mkdir -p ${DIR}.sbert_trgonly.oracle
mkdir -p ${DIR}.sbert_trgonly.NBoW

for tvt in dev test train_h40000
do
    python 01.preprocess_NeuralBoW.py  -src            ${DIR}.sbert_trgonly/merge.top10/${CORPUS}_$tvt.en.with_match.tkn \
                                    -ref            ${DIR}.sbert_trgonly/merge.top10/${CORPUS}_$tvt.ja.with_match.tkn \
                                    -out_in4oracle  ${DIR}.sbert_trgonly.oracle/${CORPUS}_$tvt.oracle.src.tkn \
                                    -out_ref4oracle ${DIR}.sbert_trgonly.oracle/${CORPUS}_$tvt.oracle.trg.tkn \
                                    -out_in4nbow    ${DIR}.sbert_trgonly.NBoW/${CORPUS}_$tvt.in4nbow \
                                    -out_ref4nbow   ${DIR}.sbert_trgonly.NBoW/${CORPUS}_$tvt.ref4nbow \
                                    -out_numofsim   ${DIR}.sbert_trgonly.NBoW/${CORPUS}_$tvt.numofsim
done


# Top100

# mkdir -p ${DIR}.sbert_trgonly_top100.oracle
# mkdir -p ${DIR}.sbert_trgonly_top100.NBoW

# for tvt in dev test train_h40000
# do
#     python 01.preprocess_NeuralBoW.py  -src            ${DIR}.sbert_trgonly_top100/merge.top100/${CORPUS}_$tvt.en.with_match.tkn \
#                                     -ref            ${DIR}.sbert_trgonly_top100/merge.top100/${CORPUS}_$tvt.ja.with_match.tkn \
#                                     -out_in4oracle  ${DIR}.sbert_trgonly_top100.oracle/${CORPUS}_$tvt.oracle.src.tkn \
#                                     -out_ref4oracle ${DIR}.sbert_trgonly_top100.oracle/${CORPUS}_$tvt.oracle.trg.tkn \
#                                     -out_in4nbow    ${DIR}.sbert_trgonly_top100.NBoW/${CORPUS}_$tvt.in4nbow \
#                                     -out_ref4nbow   ${DIR}.sbert_trgonly_top100.NBoW/${CORPUS}_$tvt.ref4nbow \
#                                     -out_numofsim   ${DIR}.sbert_trgonly_top100.NBoW/${CORPUS}_$tvt.numofsim
# done