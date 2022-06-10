
nfr_dir=/mnt/work/20220522_NMTwithLexicalConstraints/data/aspec.labse/merge_srctrg.top1/
oracle_dir=/mnt/work/20220522_NMTwithLexicalConstraints/data/aspec.labse.oracle/srctrg_top100/
proposed_dir=/mnt/work/20220522_NMTwithLexicalConstraints/experiments/nbow.aspec_h40000.labse.srctrg_top10/topk/

nfr_oracle_dir=/mnt/work/20220522_NMTwithLexicalConstraints/data/aspec.labse.nfr_oracle
nfr_proposed_dir=/mnt/work/20220522_NMTwithLexicalConstraints/data/aspec.labse.nfr_proposed

for tvt in test dev train_h40000; do
    # oracle
    cut -d '|' -f2- $oracle_dir/aspec_$tvt.oracle.src.tkn.bpe > tmp/a
    paste -d '|' $nfr_dir/aspec_$tvt.en.tkn.bpe tmp/a > $nfr_oracle_dir/aspec_$tvt.nfr_oracle.src.tkn.bpe
done

for tvt in test; do
    # proposed
    cut -d '|' -f2- $proposed_dir/aspec_$tvt.assisted.src.tkn.with_upto_No.100.bpe > tmp/a
    paste -d '|' $nfr_dir/aspec_$tvt.en.tkn.bpe tmp/a > $nfr_proposed_dir/aspec_$tvt.nfr_proposed.src.tkn.bpe
done