# DEXML
Adapted from https://github.com/nilesh2797/DEXML: Codebase for learning dual-encoder models for (extreme) multi-label classification tasks.
Original citation remains in place.

> [Dual-encoders for Extreme Multi-label Classification](https://arxiv.org/pdf/2310.10636v2.pdf) <br>
> Nilesh Gupta, Devvrit Khatri, Ankit S. Rawat, Srinadh Bhojanapalli, Prateek Jain, Inderjit S. Dhillon <br>
> ICLR 2024

## Highlights
- Adapted to focus on negative sampling in Dual Encoder architectures
- Offers In-batch sampling (default), [Mixed Negative Sampling](https://dl.acm.org/doi/10.1145/3366424.3386195), [Batch-Mix Negative Sampling](https://dl.acm.org/doi/10.1145/3583780.3614789), [Cross-Batch Negative Sampling](https://dl.acm.org/doi/10.1145/3404835.3463032) and two new  novel approaches: Cross-Batch Mixed Negative Sampling and Mixed-Bayesian Negative Sampling.
- Multi-label retrieval losses DecoupledSoftmax and SoftTopk (replacement for InfoNCE (Softmax) loss in multi-label and top-k retrieval settings)
- Distributed dual-encoder training using gradient caching (allows for a large pool of labels in loss computation without getting OOM)
- State-of-the-art dual-encoder models for extreme multi-label classification benchmarks

## Training DEXML
### Datasets
The following XMC Datasets may be used: EURLex-4K, LF-AmazonTitles-131K, LF-AmazonTitles-1.3M, LF-Wikipedia-500K. Please see the [XML Repository](http://manikvarma.org/downloads/XC/XMLRepository.html) for more information on these open datasets. Helper scripts have been provided in the /bin directory which download these datasets.

To prepare the datasets into their expected data structure: <br>
```shell
# Run pip install first
pip install -r requirements.txt

# EURLex-4K
chmod a+x bin/process-eurlex-4k.sh
bin/process_eurlex_4k.sh

# LF-AmazonTitles-*
python3 bin/process-amazon.py LF-AmazonTitles-131K
python3 bin/process-amazon.py LF-AmazonTitles-1.3M
```

#### Manual Setup
Alternatively, if the datasets were manually downloaded, the codebase expects them in the following structure

```shell
Datasets/
|- EURLex-4K # Dataset name
    |-- raw
    |   +-- trn_X.txt # train input file, ith line is the text input for ith train data point
    |   +-- tst_X.txt # test input file, ith line is the text input for ith test data point
    |   +-- Y.txt # label input file, ith line is the text input for ith label in the dataset
    |-- Y.trn.npz # train relevance matrix (stored in scipy sparse npz format), num_train x num_labels
    |-- Y.tst.npz # test relevance matrix (stored in scipy sparse npz format), num_test x num_labels
```

The input features will then need to be converted to Bert's tokenised input indices. 
```shell
dataset="EURLex-4K"
python3 utils/tokenization_utils.py --data-path Datasets/${dataset}/raw/Y.txt --tf-max-len 128 --tf-token-type bert-base-uncased
python3 utils/tokenization_utils.py --data-path Datasets/${dataset}/raw/trn_X.txt --tf-max-len 128 --tf-token-type bert-base-uncased
python3 utils/tokenization_utils.py --data-path Datasets/${dataset}/raw/tst_X.txt --tf-max-len 128 --tf-token-type bert-base-uncased
```

## Docker
If using the [University of Bath Cloud](https://hex.cs.bath.ac.uk/usage), a Docker file exists. A helper script has been provided that will build the docker image and will load a user into a bash shell to begin training. Please note that rather than using the docker command, the helper script uses [`Hare`](https://hex.cs.bath.ac.uk/wiki/Docker-via-Hare.md). If you wish to use Docker outside of the University's cloud, then simply changing the command in the helper script to `Docker` will allow you to run on a local or other environment setup.
```shell
# Execute run.sh
chmod a+x bin/run.sh
bin/run.sh
```

For some extreme classification benchmark datasets such as LF-AmazonTitles-131K and LF-AmazonTitles-1.3M, you additionally need test time label filter files (`Datasets/${dataset}/filter_labels_test.txt)`) to get the right results. Please see note on these filter files [here](http://manikvarma.org/downloads/XC/XMLRepository.html#ba-pair) to know more.


## Experiment results
The results of the experiments run for the purpose of the dissertation were captured in Weights and Biases and the reports are available here:

EURLex-4K: https://api.wandb.ai/links/gnsiva/nvbfegl6 <br>
LF-AmazonTitles-131K: https://api.wandb.ai/links/gnsiva/e48t5lb1


## Training commands
Training code assumes all hyperparameter and runtime arguments are specified in a config yaml file. Please see `configs/dual_encoder.yaml` for a brief description of all parameters (you can keep most of the parameters same across experiments). See `configs/EURLex-4K/dist-de-hnm_decoupled-softmax.yaml` to see some of the important hyperparameters that you may want to change for different experiments.
```shell
# Single GPU
dataset="EURLex-4K"
python train.py configs/${dataset}/dist-de-all_decoupled-softmax.yaml

# Multi GPU
num_gpus=4
accelerate launch --config_file configs/accelerate.yaml --num_processes ${num_gpus} train.py configs/${dataset}/dist-de-hnm_decoupled-softmax.yaml
```

@modified by Sivananda Ganesamoorthy under the [original Apache 2.0 license](https://github.com/nilesh2797/DEXML/blob/main/LICENSE)

## Cite
```bib
@InProceedings{DEXML,
  author    = "Gupta, N. and Khatri, D. and Rawat, A-S. and Bhojanapalli, S. and Jain, P. and Dhillon, I.",
  title     = "Dual-encoders for Extreme Multi-label Classification",
  booktitle = "International Conference on Learning Representations",
  month     = "May",
  year      = "2024"
}
```
