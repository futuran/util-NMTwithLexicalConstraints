CORPUS=$1
if [ "$CORPUS" = "" ]; then
    echo "Please designate CORPUS!!!"
    exit
fi
DIR=../../data/${CORPUS}
MODEL=$2
if [ "$MODEL" = "" ]; then
    echo "Please designate MODEL!!!"
    exit
fi

mkdir -p $DIR.$MODEL/match_srctgt_margin/

for tvt in test valid train; do
    python 03.search_marginbased.py -s2t $DIR.$MODEL/match_srctgt/$tvt.match \
                                    -t2s $DIR.$MODEL/match_srctgt_inv/$tvt.match \
                                    -o   $DIR.$MODEL/match_srctgt_margin/$tvt.match
done