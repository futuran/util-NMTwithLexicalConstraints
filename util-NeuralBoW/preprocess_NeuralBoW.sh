# XLM-RoBERTaに基づくフレーズ抽出モデル訓練のためのデータセット作成
CORPUS=$1
DIR=../../data/${CORPUS}

mkdir -p ${DIR}.sbert.oracle
mkdir -p ${DIR}.sbert.NBoW

for tvt in dev test train_h40000
do
    python preprocess_NeuralBoW.py  -src            ${DIR}.sbert/merge.top10/${CORPUS}_$tvt.en.with_match.tkn \
                                    -ref            ${DIR}.sbert/merge.top10/${CORPUS}_$tvt.ja.with_match.tkn \
                                    -out_in4oracle  ${DIR}.sbert.oracle/${CORPUS}_$tvt.oracle.src.tkn \
                                    -out_ref4oracle ${DIR}.sbert.oracle/${CORPUS}_$tvt.oracle.trg.tkn \
                                    -out_in4nbow    ${DIR}.sbert.NBoW/${CORPUS}_$tvt.in4nbow \
                                    -out_ref4nbow   ${DIR}.sbert.NBoW/${CORPUS}_$tvt.ref4nbow \
                                    -out_numofsim   ${DIR}.sbert.NBoW/${CORPUS}_$tvt.numofsim
done


