# XLM-RoBERTaに基づくフレーズ抽出モデル訓練のためのデータセット作成
CORPUS=$1
DIR=../../data/${CORPUS}

mkdir -p ${DIR}.sbert_srconly.oracle
mkdir -p ${DIR}.sbert_srconly.NBoW

for tvt in dev test train_h40000
do
    python 01.preprocess_NeuralBoW.py  -src            ${DIR}.sbert_srconly/merge.top10/${CORPUS}_$tvt.en.with_match.tkn \
                                    -ref            ${DIR}.sbert_srconly/merge.top10/${CORPUS}_$tvt.ja.with_match.tkn \
                                    -out_in4oracle  ${DIR}.sbert_srconly.oracle/${CORPUS}_$tvt.oracle.src.tkn \
                                    -out_ref4oracle ${DIR}.sbert_srconly.oracle/${CORPUS}_$tvt.oracle.trg.tkn \
                                    -out_in4nbow    ${DIR}.sbert_srconly.NBoW/${CORPUS}_$tvt.in4nbow \
                                    -out_ref4nbow   ${DIR}.sbert_srconly.NBoW/${CORPUS}_$tvt.ref4nbow \
                                    -out_numofsim   ${DIR}.sbert_srconly.NBoW/${CORPUS}_$tvt.numofsim
done


