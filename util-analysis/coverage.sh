TYPE="_srconly"
CORPUS=$1
dir_ref=../../data/${CORPUS}/
dir_nfr=../../data/${CORPUS}.sbert$TYPE/merge.top10/
dir_oracle=../../data/${CORPUS}.sbert$TYPE.oracle/
dir_assisted=../../experiments/nbow.${CORPUS}_h40000.sbert$TYPE/

# カバー率の調査
dir=sbleu.aspec
python coverage.py -ref      $dir_ref/${CORPUS}_test.ja.tkn \
                   -sim      $dir_nfr/${CORPUS}_test.en.with_match.tkn \
                   -oracle   $dir_oracle/${CORPUS}_test.oracle.src.tkn \
                   -proposed $dir_assisted/${CORPUS}_test.assisted.src.tkn \
                   -topk 10