cat data_asg4/raw_data/train.fr | perl moses_scripts/normalize-punctuation.perl -l fr | perl moses_scripts/tokenizer.perl -l fr -a -q > data_asg4/preprocessed_data/train.fr.p

cat data_asg4/raw_data/train.en | perl moses_scripts/normalize-punctuation.perl -l en | perl moses_scripts/tokenizer.perl -l en -a -q > data_asg4/preprocessed_data/train.en.p

perl moses_scripts/train-truecaser.perl --model data_asg4/preprocessed_data/tm.fr --corpus data_asg4/preprocessed_data/train.fr.p

perl moses_scripts/train-truecaser.perl --model data_asg4/preprocessed_data/tm.en --corpus data_asg4/preprocessed_data/train.en.p

cat data_asg4/preprocessed_data/train.fr.p | perl moses_scripts/truecase.perl --model data_asg4/preprocessed_data/tm.fr > data_asg4/preprocessed_data/train.fr

cat data_asg4/preprocessed_data/train.en.p | perl moses_scripts/truecase.perl --model data_asg4/preprocessed_data/tm.en > data_asg4/preprocessed_data/train.en

cat data_asg4/preprocessed_data/train.fr  data_asg4/preprocessed_data/train.en >  data_asg4/preprocessed_data/train.all

subword-nmt learn-bpe -s 2000 <  data_asg4/preprocessed_data/train.all >  data_asg4/preprocessed_data/bpe.codes

subword-nmt apply-bpe -c  data_asg4/preprocessed_data/bpe.codes <  data_asg4/preprocessed_data/train.fr >  data_asg4/preprocessed_data/train.fr.p

subword-nmt apply-bpe -c  data_asg4/preprocessed_data/bpe.codes <  data_asg4/preprocessed_data/train.en >  data_asg4/preprocessed_data/train.en.p

mv data_asg4/preprocessed_data/train.fr.p  data_asg4/preprocessed_data/train.fr
mv data_asg4/preprocessed_data/train.en.p  data_asg4/preprocessed_data/train.en

cat data_asg4/raw_data/valid.fr | perl moses_scripts/normalize-punctuation.perl -l fr | perl moses_scripts/tokenizer.perl -l fr -a -q | perl moses_scripts/truecase.perl --model data_asg4/preprocessed_data/tm.fr | subword-nmt apply-bpe -c  data_asg4/preprocessed_data/bpe.codes > data_asg4/preprocessed_data/valid.fr

cat data_asg4/raw_data/valid.en | perl moses_scripts/normalize-punctuation.perl -l en | perl moses_scripts/tokenizer.perl -l en -a -q | perl moses_scripts/truecase.perl --model data_asg4/preprocessed_data/tm.en | subword-nmt apply-bpe -c  data_asg4/preprocessed_data/bpe.codes > data_asg4/preprocessed_data/valid.en

cat data_asg4/raw_data/test.fr | perl moses_scripts/normalize-punctuation.perl -l fr | perl moses_scripts/tokenizer.perl -l fr -a -q | perl moses_scripts/truecase.perl --model data_asg4/preprocessed_data/tm.fr | subword-nmt apply-bpe -c  data_asg4/preprocessed_data/bpe.codes > data_asg4/preprocessed_data/test.fr

cat data_asg4/raw_data/test.en | perl moses_scripts/normalize-punctuation.perl -l en | perl moses_scripts/tokenizer.perl -l en -a -q | perl moses_scripts/truecase.perl --model data_asg4/preprocessed_data/tm.en | subword-nmt apply-bpe -c  data_asg4/preprocessed_data/bpe.codes > data_asg4/preprocessed_data/test.en

cat data_asg4/raw_data/tiny_train.fr | perl moses_scripts/normalize-punctuation.perl -l fr | perl moses_scripts/tokenizer.perl -l fr -a -q | perl moses_scripts/truecase.perl --model data_asg4/preprocessed_data/tm.fr | subword-nmt apply-bpe -c  data_asg4/preprocessed_data/bpe.codes > data_asg4/preprocessed_data/tiny_train.fr

cat data_asg4/raw_data/tiny_train.en | perl moses_scripts/normalize-punctuation.perl -l en | perl moses_scripts/tokenizer.perl -l en -a -q | perl moses_scripts/truecase.perl --model data_asg4/preprocessed_data/tm.en | subword-nmt apply-bpe -c  data_asg4/preprocessed_data/bpe.codes > data_asg4/preprocessed_data/tiny_train.en

python preprocess.py --target-lang en --source-lang fr --dest-dir data_asg4/prepared_data/ --train-prefix data_asg4/preprocessed_data/train --valid-prefix data_asg4/preprocessed_data/valid --test-prefix data_asg4/preprocessed_data/test --tiny-train-prefix data_asg4/preprocessed_data/tiny_train --threshold-src 1 --threshold-tgt 1 --num-words-src 4000 --num-words-tgt 4000
