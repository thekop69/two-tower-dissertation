# /bin/bash

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -gpu|--gpu)
      GPU="$2"
      shift # past argument
      shift # past value
      ;;
    -tf-max-len|--tf-max-len)
      TF_MAX_LEN="$2"
      shift # past argument
      shift # past value
      ;;
    -dataset|--dataset)
      DATASET="$2"
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

if [[ -z $GPU || -z $TF_MAX_LEN || -z $DATASET ]]; then
    echo "Missing required arguments"
    echo "GPU        = ${GPU}"
    echo "TF MAX LEN = ${TF_MAX_LEN}"
    echo "DATASET    = ${DATASET}"
    exit 1
fi


# Create "pytorch-gpu" image from the Dockerfile  
hare build -t sg2487/pytorch-gpu -f ../Dockerfile .

hare run --rm -v "$(pwd)":/app --name sg2487-xmc --gpus device=${GPU} sg2487/pytorch-gpu /bin/bash -c "python3 utils/tokenization_utils.py --data-path Datasets/${DATASET}/raw/Y.txt --tf-max-len ${TF_MAX_LEN} --tf-token-type bert-base-uncased && \
python3 utils/tokenization_utils.py --data-path Datasets/${DATASET}/raw/trn_X.txt --tf-max-len ${TF_MAX_LEN} --tf-token-type bert-base-uncased && \
python3 utils/tokenization_utils.py --data-path Datasets/${DATASET}/raw/tst_X.txt --tf-max-len ${TF_MAX_LEN} --tf-token-type bert-base-uncased && \
python3 train.py configs/${DATASET}/dist-de-all_decoupled-softmax.yaml"
