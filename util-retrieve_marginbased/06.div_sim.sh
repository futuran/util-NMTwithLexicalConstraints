# 入力文 | 類似文1 | 類似文2 | ... | 類似文N となっているファイルを
# 入力文 | 類似文 のフォーマットのN個のファイルに分割

CORPUS=$1
if [ "$CORPUS" = "" ]; then
    echo "Please designate CORPUS!!!"
    exit
fi
MODEL=$2
if [ "$MODEL" = "" ]; then
    echo "Please designate MODEL!!!"
    exit
fi

DIR=../../data/$CORPUS.$MODEL/merge_srctgt_margin.top10
mkdir $DIR/tmp
mkdir $DIR.div

for tvt in train valid test; do
    FILE=$DIR/ecb_$tvt.en.tkn.bpe
    cut -d '|' -f1 $FILE > $DIR/tmp/src
    for topk in `seq 2 11`; do
        echo `expr $topk - 1`
        cut -d '|' -f$topk $FILE > $DIR/tmp/sim_$topk
        paste -d '|' $DIR/tmp/src $DIR/tmp/sim_$topk > $DIR.div/ecb_$tvt.en.tkn.bpe.`expr $topk - 1`
    done
done

# 第N近傍文までを別事例として１ファイルにまとめる
# rm $DIR.div/ecb_train.en.tkn.bpe.1to5
# rm $DIR.div/ecb_train.fr.tkn.bpe.1to5
# for topk in `seq 1 5`; do
#     echo $topk
#     cat $DIR.div/ecb_train.en.tkn.bpe.$topk >> $DIR.div/ecb_train.en.tkn.bpe.1to5
#     cat ../ecb/ecb_train.fr.tkn.bpe >> $DIR.div/ecb_train.fr.tkn.bpe.1to5
# done