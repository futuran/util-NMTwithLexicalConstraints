# 語彙制約で参照訳文をどれだけカバーできるかを調査

TYPE="_srconly"
CORPUS=$1
dir_ref=../../data/${CORPUS}/
dir_nfr=../../data/${CORPUS}.sbert${TYPE}_top100/merge.top100/
dir_oracle=../../data/${CORPUS}.sbert${TYPE}_top100.oracle/
dir_assisted=../../experiments/nbow.${CORPUS}_h40000.sbert$TYPE/

# ベースラインの出力
#baseline_out=../../experiments/ex.${CORPUS}_h40000/test_20000_${CORPUS}.out
baseline_out=/mnt/work/20220401_NMTwithLexicalConstraints/experiments.hf.aspec/mbart-large-50_baseline/generated_predictions.txt
# baseline_out=/mnt/work/20220401_NMTwithLexicalConstraints/experiments.hf.aspec/mbart-large-50_proposed_srconly.top100/generated_predictions.txt

# カバー率の調査
dir=sbleu.aspec
python coverage.py -ref      $dir_ref/${CORPUS}_test.ja.tkn \
                   -out      $baseline_out \
                   -sim      $dir_nfr/${CORPUS}_test.en.with_match.tkn \
                   -multiple_proposed $dir_assisted/for_analyze/${CORPUS}_test.assisted_top100.src.tkn.with_upto_No. \
                   -topk 100 \