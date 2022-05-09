# 語彙制約で参照訳文をどれだけカバーできるかを調査

TYPE=""
CORPUS=$1
dir_ref=../../data/${CORPUS}/
dir_nfr=../../data/${CORPUS}.sbert${TYPE}_top100/merge.top100/
dir_oracle=../../data/${CORPUS}.sbert${TYPE}_top100.oracle/
dir_assisted=../../experiments/nbow.${CORPUS}_h40000.sbert$TYPE/

# カバー率の調査
dir=sbleu.aspec
python coverage.py -ref      $dir_ref/${CORPUS}_test.ja.tkn \
                   -topk 100 \
                   -sim      $dir_nfr/${CORPUS}_test.en.with_match.tkn \
                   -oracle   $dir_oracle/${CORPUS}_test.oracle.src.tkn \
                   -proposed $dir_assisted/${CORPUS}_test.assisted_top100.src.tkn \
                   -multiple_proposed $dir_assisted/for_analyze/${CORPUS}_test.assisted_top100.src.tkn.with_upto_No. \