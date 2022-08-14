
mkdir -p /mnt/work/20220729_cxmi/data/ecb.labse/match_srctgt_margin/
for tvt in test valid train; do
    python 03.search_marginbased.py -s2t /mnt/work/20220729_cxmi/data/ecb.labse/match_srctgt/$tvt.match \
                                    -t2s /mnt/work/20220729_cxmi/data/ecb.labse/match_srctgt_inv/$tvt.match \
                                    -o   /mnt/work/20220729_cxmi/data/ecb.labse/match_srctgt_margin/$tvt.match
done