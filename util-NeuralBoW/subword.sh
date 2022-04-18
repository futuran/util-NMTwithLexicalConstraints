# aspec
code=../aspec/code.all

dir=../aspec.sbert.oracle/
for tvt in dev test train_h40000 train; do
        subword-nmt apply-bpe -c $code < $dir/aspec_$tvt.oracle.src.tkn > $dir/aspec_$tvt.oracle.src.tkn.bpe
        subword-nmt apply-bpe -c $code < $dir/aspec_$tvt.oracle.trg.tkn > $dir/aspec_$tvt.oracle.trg.tkn.bpe
done

