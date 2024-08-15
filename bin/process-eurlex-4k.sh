#/bin/bash

mkdir -p Datasets

cd Datasets; gdown 1A_sL_mzpkmnr6g0DSZ0_xJTr4GN-rIfi; tar -xvzf Eurlex-4K.tar.gz; mv Eurlex-4K EURLex-4K
mkdir -p ./EURLex-4K/raw;

cd EURLex-4K; mv train_raw_texts.txt raw/trn_X.txt; mv test_raw_texts.txt raw/tst_X.txt; mv label_map.txt raw/Y.txt

cd ../..
python3 utils/tokenization_utils.py --data-path Datasets/EURLex-4K/raw/trn_X.txt --tf-max-len 128
python3 utils/tokenization_utils.py --data-path Datasets/EURLex-4K/raw/tst_X.txt --tf-max-len 128
python3 utils/tokenization_utils.py --data-path Datasets/EURLex-4K/raw/Y.txt --tf-max-len 128

rm Datasets/Eurlex-4K.tar.gz