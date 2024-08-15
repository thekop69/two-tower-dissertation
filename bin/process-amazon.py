import pandas, gzip, os, sys
import scipy.sparse as sp
import numpy as np

args = sys.argv
if len(args) > 1:
  opt = args[1]

def usage():
  print('Usage: python3 process-amazon.py {dataset}]')
  print('  Example Usage:')
  print('  dataset=LF-AmazonTitles-131K')
  print('  python3 process-amazon.py ${dataset}')
  print('  ====================================')
  print('  Dataset: LF-AmazonTitles-131K')
  print('  Dataset: LF-AmazonTitles-1.3M')
  sys.exit(0)

def to_scipy_matrix(lol_inds, num_cols):
  cols = np.concatenate(lol_inds)
  rows = np.concatenate([[i]*len(x) for i, x in enumerate(lol_inds)])
  data = np.concatenate([[1.0]*len(x) for i, x in enumerate(lol_inds)])
  return sp.coo_matrix((data, (rows, cols)), (len(lol_inds), num_cols)).tocsr()

def process_lf_amazon_datasets(dataset):
  print('Reading raw dataset files...')
  trn_df = pandas.read_json(gzip.open(f'Datasets/{dataset}/trn.json.gz'), lines=True)
  tst_df = pandas.read_json(gzip.open(f'Datasets/{dataset}/tst.json.gz'), lines=True)
  lbl_df = pandas.read_json(gzip.open(f'Datasets/{dataset}/lbl.json.gz'), lines=True)

  print('Processing Y (label) files...')
  trn_X_Y = to_scipy_matrix(trn_df.target_ind.values, lbl_df.shape[0])
  tst_X_Y = to_scipy_matrix(tst_df.target_ind.values, lbl_df.shape[0])

  sp.save_npz(f'Datasets/{dataset}/Y.trn.npz', trn_X_Y)
  sp.save_npz(f'Datasets/{dataset}/Y.tst.npz', tst_X_Y)

  print('Processing X (input) files...')
  print(*trn_df.title.apply(lambda x: x.strip()).values, sep='\n', file=open(f'Datasets/{dataset}/raw/trn_X.txt', 'w'))
  print(*tst_df.title.apply(lambda x: x.strip()).values, sep='\n', file=open(f'Datasets/{dataset}/raw/tst_X.txt', 'w'))
  print(*lbl_df.title.apply(lambda x: x.strip()).values, sep='\n', file=open(f'Datasets/{dataset}/raw/Y.txt', 'w'))

  print('Tokenizing X (input) files...')
  max_len = 32 
  os.system(f"python3 utils/tokenization_utils.py --data-path Datasets/{dataset}/raw/trn_X.txt --tf-max-len {max_len}")
  os.system(f"python3 utils/tokenization_utils.py --data-path Datasets/{dataset}/raw/tst_X.txt --tf-max-len {max_len}")
  os.system(f"python3 utils/tokenization_utils.py --data-path Datasets/{dataset}/raw/Y.txt --tf-max-len {max_len}")

try:
  if not opt or opt == '--help':
    usage()
except NameError:
  print('Error: Missing dataset argument')
  usage()

dataset = opt
if dataset == 'LF-AmazonTitles-131K':
  download = 'LF-Amazon-131K'
  download_link = '1WuquxCAg8D4lKr-eZXPv4nNw2S2lm7_E'
else:
  download = 'LF-Amazon-1.3M'
  download_link = '12zH4mL2RX8iSvH0VCNnd3QxO4DzuHWnK'

cmd = 'mkdir -p Datasets;'
cmd += 'cd Datasets; gdown ' + download_link + ';'
cmd += 'unzip ' + download + '.raw.zip; mv '+ download + ' ' + dataset +'; mkdir -p '+ dataset + '/raw;'
cmd += 'rm ' + download + '.raw.zip'
os.system(cmd)

process_lf_amazon_datasets(dataset)