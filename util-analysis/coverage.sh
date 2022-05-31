# 語彙制約で参照訳文をどれだけカバーできるかを調査

TYPE="_srcsrc"
CORPUS=$1
MODEL="labse"
ORIG=""
dir_ref=../../data/${CORPUS}/
dir_nfr=../../data/${CORPUS}.$MODEL/${ORIG}merge$TYPE.top100/
dir_oracle=../../data/${CORPUS}.$MODEL$TYPE.oracle/
dir_assisted=../../experiments/nbow.${CORPUS}_h40000.$MODEL$TYPE/

# カバー率の調査
python coverage.py -ref      $dir_ref/${CORPUS}_test.ja.tkn \
                   -sim      $dir_nfr/${CORPUS}_test.en.with_match.tkn \
                   -topk 100


                #    -oracle   $dir_oracle/${CORPUS}_test.oracle.src.tkn \
                #    -proposed $dir_assisted/${CORPUS}_test.assisted.src.tkn \