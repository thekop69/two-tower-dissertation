import os, torch
import numpy as np
import scipy.sparse as sp
import sklearn.preprocessing
import random
from multiprocessing import Queue, Lock
from transformers import AutoTokenizer
from torch.distributions.beta import Beta
from torch.distributions.uniform import Uniform

from utils.helper_utils import _c, get_inv_prop, load_filter_mat, _filter, compute_xmc_metrics
from utils.dl_utils import unwrap, csr_to_bow_tensor, csr_to_pad_tensor, bert_fts_batch_to_tensor, expand_multilabel_dataset

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, labels: sp.csr_matrix, sample=None, filter_mat=None):
        super().__init__()
        self.sample = np.arange(labels.shape[0]) if sample is None else sample
        self.labels = labels[self.sample]
        self.filter_mat = filter_mat[self.sample] if filter_mat is not None else None

    def __getitem__(self, index):
        return {'index': index}

    def __len__(self):
        return len(self.sample)

class SimpleDataset(BaseDataset):
    def __init__(self, features, labels, **super_kwargs):
        super().__init__(labels, **super_kwargs)
        self.features = features
    
    def get_fts(self, indices):
        if isinstance(self.features, sp.csr_matrix):
            return csr_to_bow_tensor(self.features[self.sample[indices]])
        else:
            return torch.Tensor(self.features[self.sample[indices]])

class OfflineBertDataset(BaseDataset):
    def __init__(self, fname, labels, max_len, token_type='bert-base-uncased', **super_kwargs):
        super().__init__(labels, **super_kwargs)
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(token_type, clean_up_tokenization_spaces=True)
        nr, nc, dtype = open(f'{fname}.meta').readline().split()
        self.X_ii = np.memmap(f"{fname}", mode='r', shape=(int(nr), int(nc)), dtype=dtype)
    
    def get_fts(self, indices):
        X_ii = np.array(self.X_ii[self.sample[indices]]).reshape(-1, self.X_ii.shape[1])
        X_am = (X_ii != self.tokenizer.pad_token_id)
        return bert_fts_batch_to_tensor(X_ii, X_am)
    
class OnlineBertDataset(BaseDataset):
    def __init__(self, X, labels, max_len, token_type='bert-base-uncased', **super_kwargs):
        super().__init__(labels, **super_kwargs)
        self.max_len = max_len
        self.X = np.array(X, dtype=object)
        self.tokenizer = AutoTokenizer.from_pretrained(token_type, clean_up_tokenization_spaces=True)
    
    def get_fts(self, indices):
        return self.tokenizer.batch_encode_plus(list(self.X[self.sample[indices]]), 
                                                max_length=self.max_len, 
                                                padding=True, 
                                                truncation=True, 
                                                return_tensors='pt', 
                                                return_token_type_ids=False).data

class XMCCollator():
    def __init__(self, dataset):
        self.dataset = dataset
        self.numy = self.dataset.labels.shape[1]
    
    def __call__(self, batch):
        batch_size = len(batch)
        ids = torch.LongTensor([b['index'] for b in batch])
        
        b = {'batch_size': torch.LongTensor([batch_size]),
             'numy': torch.LongTensor([self.numy]),
             'y': csr_to_pad_tensor(self.dataset.labels[ids], self.numy),
             'ids': ids,
             'xfts': self.dataset.get_fts(ids)}
             
        return b

class TwoTowerDataset(BaseDataset):
    def __init__(self, x_dataset, y_dataset, shorty=None):
        super().__init__(labels=x_dataset.labels, filter_mat=x_dataset.filter_mat)
        self.x_dataset = x_dataset
        self.y_dataset = y_dataset
        self.shorty = shorty
            
    def __getitem__(self, index):
        ret = {'index': index}
        return ret

    def get_fts(self, indices, source):
        if source == 'x':
            return self.x_dataset.get_fts(indices)
        elif source == 'y':
            return self.y_dataset.get_fts(indices)
        
    def get_rating_vector(self):
        rating_vector = np.array(self.labels.sum(axis=0)).flatten()
        return rating_vector

    def __len__(self):
        return self.labels.shape[0]
    
class CrossDataset(BaseDataset):
    def __init__(self, x_dataset, y_dataset, shorty=None, iterate_over='labels'):
        super().__init__(labels=x_dataset.labels, filter_mat=x_dataset.filter_mat)
        self.x_dataset = x_dataset
        self.y_dataset = y_dataset
        self.iterate_over = iterate_over
        self.labels = self.x_dataset.labels
        self.shorty = shorty
        # Assumption: always binarizing shortlist
        self.shorty.data[:] = 1.0

        if self.iterate_over == 'labels':
            self.rows, self.cols = self.labels.nonzero()
            self.shorty = _filter(self.shorty, self.labels, copy=False)
        elif self.iterate_over == 'shorty':
            self.rows, self.cols = self.shorty.nonzero()
            
    def __getitem__(self, index):
        ret = {'index': index}
        return ret

    def get_fts(self, indices, source):
        if source == 'x':
            return self.x_dataset.get_fts(indices)
        elif source == 'y':
            return self.y_dataset.get_fts(indices)

    def __len__(self):
        return self.rows.shape[0]

class CrossCollator():
    def __init__(self, dataset: CrossDataset, num_neg_samples=1):
        self.numy = dataset.labels.shape[1]
        self.dataset = dataset
        self.num_neg_samples = num_neg_samples
    
    def __call__(self, batch):
        batch_size = len(batch)
        ids = torch.LongTensor([b['index'] for b in batch])
        
        b = {'batch_size': torch.LongTensor([batch_size]),
             'numy': torch.LongTensor([self.numy]),
             'ids': ids
             }
        
        b_query_inds = torch.tensor(self.dataset.rows[ids])
        b_label_inds = torch.tensor(self.dataset.cols[ids])
        b_targets = None
        
        if self.dataset.shorty is not None and self.num_neg_samples > 0 and self.dataset.iterate_over == 'labels':
            batch_shorty = csr_to_pad_tensor(self.dataset.shorty[b_query_inds], self.numy-1)
            batch_shorty_inds = torch.multinomial(torch.maximum(batch_shorty['vals'].double(), torch.tensor(1e-8)), self.num_neg_samples)
            batch_shorty = torch.gather(batch_shorty['inds'], 1, batch_shorty_inds)

            b_query_inds = b_query_inds.unsqueeze(1).repeat(1, self.num_neg_samples+1).view(-1)
            b_targets = torch.hstack([torch.ones(*b_label_inds.shape).unsqueeze(1), torch.zeros(*batch_shorty.shape)]).view(-1)
            b_label_inds = torch.hstack([b_label_inds.unsqueeze(1), batch_shorty]).view(-1)

        b['xfts'] = self.dataset.get_fts(b_query_inds, 'x')
        b['yfts'] = self.dataset.get_fts(b_label_inds, 'y')
        b['query_inds'] = b_query_inds
        b['label_inds'] = b_label_inds
        b['targets'] = b_targets
            
        return b

import torch.distributed as dist
class TwoTowerTrainCollator():
    def __init__(self, dataset: TwoTowerDataset, neg_type='in-batch', num_neg_samples=1, num_pos_samples=1, mixing_alpha=1, mixing_beta=1, use_fifo_queue = False, queue_size=2048):
        self.numy = dataset.labels.shape[1]
        self.dataset = dataset
        self.neg_type = neg_type
        self.num_neg_samples = num_neg_samples
        self.num_pos_samples = num_pos_samples
        self.mixing_alpha = mixing_alpha
        self.mixing_beta = mixing_beta
        self.use_fifo_queue = use_fifo_queue
        self.queue_size = queue_size
        self.neg_sample_queue = Queue(maxsize=self.queue_size)
        self.queue_lock = Lock()
        self.mask = torch.zeros(self.numy+1).long()
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    def __call__(self, batch):
        batch_size = len(batch)
        ids = np.array([b['index'] for b in batch])
        
        batch_data = {'batch_size': batch_size,
                      'numy': self.numy,
                      'y': csr_to_pad_tensor(self.dataset.labels[ids], self.numy),
                      'ids': torch.Tensor([b['index'] for b in batch]).long(),
                      'xfts': self.dataset.get_fts(ids, 'x')
                     }
        
        batch_y = None
        
        if self.neg_type == 'mbns': 
            batch_y, batch_data = self._mixed_bayesian_neg_sampling(batch_data, batch_size)

        if self.neg_type == 'cbmns': 
            batch_y, batch_data = self._cross_batch_mixed_neg_sampling(batch_data, batch_size)

        if self.neg_type == 'cbns': 
            batch_y, batch_data = self._cross_batch_neg_sampling(batch_data, batch_size)

        if self.neg_type == 'bmns': 
            batch_y, batch_data = self._batch_mix_neg_sampling(batch_data, batch_size)

        elif self.neg_type == 'mns':
            batch_y, batch_data = self._mixed_neg_sampling(batch_data, batch_size)

        elif self.neg_type == 'in-batch':
            batch_y, batch_data = self._in_batch_neg_sampling(batch_data, batch_size)

        elif self.neg_type == 'all':
            batch_y = torch.arange(self.numy)
            batch_data['targets'] = torch.zeros((batch_size, batch_y.shape[0]+1)).scatter_(1, batch_data['y']['inds'], 1.0)[:,:-1]
          
        if batch_y is not None:
            batch_data['batch_y'] = batch_y
            batch_data['yfts'] = self.dataset.get_fts(batch_y.numpy(), 'y')

        return batch_data

    # @see https://github.com/liubin06/BNS/blob/main/BNS_MF/sampling.py
    def _bns(self, i, negative_items, negative_index, rating_vector, prior, size, alpha):
        '''
        :param i:  positive item
        :param negative_items: negative items
        :param negative_index:
        :param rating_vector:
        :param prior: prior probability
        :param size:  the size of candidate set
        :param alpha: weight
        :return: optimal negative instance l
        '''
        def sigmoid(x):
            if x > 0:
                return 1 / (1 + np.exp(-x))
            else:
                return np.exp(x) / (1 + np.exp(x))

        x_ui = rating_vector[int(i)]
        negative_scores = rating_vector[negative_index]
        lenth = len(negative_scores) + 1
        candidate_set = np.random.choice(negative_items, size=size, replace=False)  #O(|I|)
        candidate_scores = [rating_vector[int(l)] for l in candidate_set]

        # step 1 : computing info(l)
        info = np.array([1 - sigmoid(x_ui - x_ul)  for x_ul in candidate_scores])                #O(1)
        # step 2 : computing prior probability
        p_fn = np.array([prior[int(l)] for l in candidate_set ])                                 #O(1)
        # step 3 : computing empirical distribution function (likelihood)
        F_n = np.array([np.sum(negative_scores <= x_ul) / lenth for x_ul in candidate_scores])   #O(|I|)
        # step 4: computing posterior probability
        unbias = (1 - F_n) * (1 - p_fn) / (1 - F_n - p_fn + 2 * F_n * p_fn)                      #O(1)
        # step 5: computing conditional sampling risk
        conditional_risk = (1-unbias) * info - alpha * unbias * info                             #O(1)
        j = candidate_set[conditional_risk.argsort()[0]]
        return j

    def _prior(self, labels):
        item_counts = np.array(labels.sum(axis=0)).flatten()
        prior = item_counts / item_counts.sum()
        return prior
    
    # Mixed Baysian Negative Sampling
    # Custom approach created
    # An Investigation into Negative Sampling of Two-Tower 
    # Neural Networks for Large Corpus Item Recommendations
    # @Siva Ganesamoorthy - August 2024
    def _mixed_bayesian_neg_sampling(self, batch_data, batch_size):
        batch_y_inds = torch.multinomial(batch_data['y']['vals'].double(), min(self.num_pos_samples, batch_data['y']['vals'].shape[0]), replacement=True)
        batch_pos_y = torch.gather(batch_data['y']['inds'], 1, batch_y_inds).squeeze()

        # Mine additional negative samples from the entire corpus 
        # with Bayesian Negative Sampling
        rating_vector = self.dataset.get_rating_vector()
        prior = self._prior(self.dataset.labels)

        corpus_neg_y = []
        for i in range(batch_size):
            negative_items = np.setdiff1d(np.arange(self.numy), batch_pos_y.numpy())
            negative_index = np.where(np.isin(np.arange(self.numy), negative_items))[0]
            neg_item = self._bns(i, negative_items, negative_index, rating_vector, prior, self.num_neg_samples, 10)
            corpus_neg_y.append(neg_item)

        corpus_neg_y = torch.LongTensor(corpus_neg_y)
        
        # Combine the mini-batch and additional corpus samples
        batch_y = torch.LongTensor(np.union1d(batch_pos_y, corpus_neg_y))
        batch_y = batch_y[batch_y != self.numy]
       
        self.mask[batch_y] = torch.arange(batch_y.shape[0])
        batch_data['pos-inds'] = self.mask[batch_pos_y].reshape(-1, 1)
        self.mask[batch_y] = 0

        batch_data['targets'] = torch.zeros((batch_size, batch_y.shape[0]))
        for i in range(batch_size):
            self.mask[batch_data['y']['inds'][i]] = True
            batch_data['targets'][i][self.mask[batch_y].bool()] = 1.0
            self.mask[batch_data['y']['inds'][i]] = False 

        return batch_y, batch_data

    # Cross Batch Mix Negative Sampling
    # Custom approach created
    # An Investigation into Negative Sampling of Two-Tower 
    # Neural Networks for Large Corpus Item Recommendations
    # @Siva Ganesamoorthy - August 2024
    def _cross_batch_mixed_neg_sampling(self, batch_data, batch_size):
        batch_y_inds = torch.multinomial(batch_data['y']['vals'].double(), min(self.num_pos_samples, batch_data['y']['vals'].shape[0]), replacement=True)
        batch_pos_y = torch.gather(batch_data['y']['inds'], 1, batch_y_inds).squeeze()
        
        # Mine additional negative samples from the entire corpus using
        # multinomial distribution
        corpus = csr_to_pad_tensor(self.dataset.labels, self.numy)
        corpus_neg_y_inds = torch.multinomial(corpus['vals'].double(), min(self.num_neg_samples, corpus['vals'].shape[0]), replacement=True)
        corpus_neg_y = torch.gather(corpus['inds'], 1, corpus_neg_y_inds).squeeze()

        # Combine the mini-batch and corpus samples
        combined_y = torch.LongTensor(np.union1d(batch_pos_y, corpus_neg_y))

        # Set up each GPU to pull a fraction of the queue
        pull_size = self.queue_size//self.world_size
        iter = 0

        # Pull additional samples (previous mini-batch negative samples)
        # from the FIFO queue
        additional_negs = [ [0]*batch_pos_y.shape[1] for _ in range(pull_size//batch_pos_y.shape[1])] # Maintain the shape of the mini-batch
        queue_items = []

        with self.queue_lock:
            while not self.neg_sample_queue.empty() and iter < pull_size:
                queue_items.append(self.neg_sample_queue.get(block=True))

                iter += 1
            
            for i in range(len(additional_negs)):
                for j in range(batch_pos_y.shape[1]):
                    additional_negs[i][j] = queue_items.pop(0) if len(queue_items) > 0 else 0

        additional_negs = torch.as_tensor(additional_negs)
        
        batch_y = torch.LongTensor(np.union1d(combined_y, additional_negs))
        batch_y = batch_y[batch_y != self.numy]
       
        self.mask[batch_y] = torch.arange(batch_y.shape[0])
        batch_data['pos-inds'] = self.mask[batch_pos_y].reshape(-1, 1)
        self.mask[batch_y] = 0

        batch_data['targets'] = torch.zeros((batch_size, batch_y.shape[0]))
        for i in range(batch_size):
            self.mask[batch_data['y']['inds'][i]] = True
            batch_data['targets'][i][self.mask[batch_y].bool()] = 1.0
            self.mask[batch_data['y']['inds'][i]] = False 

        # Update the FIFO queue with complete batch negative samples
        self._update_neg_sample_queue(batch_y)

        return batch_y, batch_data


    # Cross Batch negative sampling
    # @see https://dl.acm.org/doi/10.1145/3404835.3463032
    def _cross_batch_neg_sampling(self, batch_data, batch_size):
        batch_y_inds = torch.multinomial(batch_data['y']['vals'].double(), min(self.num_pos_samples, batch_data['y']['vals'].shape[0]), replacement=True)
        batch_pos_y = torch.gather(batch_data['y']['inds'], 1, batch_y_inds).squeeze()
        
        # Set up each GPU to pull a fraction of the queue
        pull_size = self.queue_size//self.world_size
        iter = 0

        # Pull additional samples (previous mini-batch negative samples)
        # from the FIFO queue
        additional_negs = [ [0]*batch_pos_y.shape[1] for _ in range(pull_size//batch_pos_y.shape[1])] # Maintain the shape of the mini-batch
        queue_items = []

        with self.queue_lock:
            while not self.neg_sample_queue.empty() and iter < pull_size:
                queue_items.append(self.neg_sample_queue.get(block=True))

                iter += 1
            
            for i in range(len(additional_negs)):
                for j in range(batch_pos_y.shape[1]):
                    additional_negs[i][j] = queue_items.pop(0) if len(queue_items) > 0 else 0

        additional_negs = torch.as_tensor(additional_negs)

        batch_y = torch.LongTensor(np.union1d(batch_pos_y, additional_negs))
        batch_y = batch_y[batch_y != self.numy]

        self.mask[batch_y] = torch.arange(batch_y.shape[0])
        batch_data['pos-inds'] = self.mask[batch_pos_y].reshape(-1, 1)
        self.mask[batch_y] = 0        
        
        batch_data['targets'] = torch.zeros((batch_size, batch_y.shape[0]))
        for i in range(batch_size):
            self.mask[batch_data['y']['inds'][i]] = True
            batch_data['targets'][i][self.mask[batch_y].bool()] = 1.0
            self.mask[batch_data['y']['inds'][i]] = False
        
        # Update the FIFO queue with complete batch negative samples
        self._update_neg_sample_queue(batch_y)

        return batch_y, batch_data
    
    # Maintain the size of the FIFO queue
    def _update_neg_sample_queue(self, samples):
        with self.queue_lock:
            for s in samples:
                if self.neg_sample_queue.full():
                    # Pop the first element from the queue
                    self.neg_sample_queue.get(block=True)
                
                self.neg_sample_queue.put(s.tolist(), block=True)

    # Batch-mix negative sampling
    # @see https://dl.acm.org/doi/10.1145/3583780.3614789
    def _batch_mix_neg_sampling(self, batch_data, batch_size):
        batch_y_inds = torch.multinomial(batch_data['y']['vals'].double(), min(self.num_pos_samples, batch_data['y']['vals'].shape[0]), replacement=True)
        batch_pos_y = torch.unique(torch.gather(batch_data['y']['inds'], 1, batch_y_inds).squeeze())

        embeddings = torch.nn.Embedding(self.numy, 1)
        batch_y_embeds = embeddings(batch_pos_y)
        
        additional_negs = []
        for i in range(batch_size):
            # Sample mixing co-efficient from a Beta distribution for each item
            # @see https://github.com/hongyi-zhang/mixup/blob/master/cifar/utils.py
            alpha = self.mixing_alpha
            beta = self.mixing_beta
            mix_coeff = Beta(alpha, beta).sample()

            index = torch.randperm(batch_y_embeds.shape[0])
            
            neg_sample = mix_coeff.item() * batch_y_embeds + (1 - mix_coeff.item()) * batch_y_embeds[index,:]
            neg_sample = torch.squeeze(neg_sample)
            additional_negs.append(neg_sample)

        # Shuffle the list before slicing to avoid bias
        random.seed(42)
        random.shuffle(additional_negs)       

        additional_negs = torch.stack(additional_negs[:min(self.num_neg_samples, batch_data['y']['vals'].shape[0])])
        additional_negs = torch.unique(additional_negs).squeeze()                

        # Combine both the mini-batch and additional samples
        batch_y = torch.LongTensor(np.union1d(batch_pos_y, additional_negs.flatten().detach().numpy()))
        batch_y = batch_y[batch_y != self.numy]

        self.mask[batch_y] = torch.arange(batch_y.shape[0])
        batch_data['pos-inds'] = self.mask[batch_pos_y].reshape(-1, 1)
        self.mask[batch_y] = 0

        batch_data['targets'] = torch.zeros((batch_size, batch_y.shape[0]))
        for i in range(batch_size):
            self.mask[batch_data['y']['inds'][i]] = True
            batch_data['targets'][i][self.mask[batch_y].bool()] = 1.0
            self.mask[batch_data['y']['inds'][i]] = False

        return batch_y, batch_data

    # Mixed negative sampling
    # @see https://dl.acm.org/doi/10.1145/3366424.3386195
    def _mixed_neg_sampling(self, batch_data, batch_size):
        # Mine samples from the mini-batch
        batch_y_inds = torch.multinomial(batch_data['y']['vals'].double(), min(self.num_pos_samples, batch_data['y']['vals'].shape[0]), replacement=True)
        batch_pos_y = torch.gather(batch_data['y']['inds'], 1, batch_y_inds).squeeze()

        # Mine additional negative samples from the entire corpus
        corpus = csr_to_pad_tensor(self.dataset.labels, self.numy)
        corpus_neg_y_inds = Uniform(torch.tensor([0.0]), torch.tensor([self.numy])).sample((corpus['vals'].shape[0], min(self.num_neg_samples, corpus['vals'].shape[1]))).squeeze().long()
        corpus_neg_y = torch.gather(corpus['inds'], 0, corpus_neg_y_inds).squeeze()
       
        # Combine both the mini-batch and corpus samples
        batch_y = torch.LongTensor(np.union1d(batch_pos_y, corpus_neg_y))
        batch_y = batch_y[batch_y != self.numy]
       
        self.mask[batch_y] = torch.arange(batch_y.shape[0])
        batch_data['pos-inds'] = self.mask[batch_pos_y].reshape(-1, 1)
        self.mask[batch_y] = 0

        batch_data['targets'] = torch.zeros((batch_size, batch_y.shape[0]))
        for i in range(batch_size):
            self.mask[batch_data['y']['inds'][i]] = True
            batch_data['targets'][i][self.mask[batch_y].bool()] = 1.0
            self.mask[batch_data['y']['inds'][i]] = False

        return batch_y, batch_data

    # In-batch negative sampling
    # Addapted from original code base
    def _in_batch_neg_sampling(self, batch_data, batch_size):
        batch_y_inds = torch.multinomial(batch_data['y']['vals'].double(), min(self.num_pos_samples, batch_data['y']['vals'].shape[1]))
        batch_y = torch.unique(torch.gather(batch_data['y']['inds'], 1, batch_y_inds).squeeze())

        if self.numy == batch_y[-1]:
            batch_y = batch_y[:-1]

        if self.use_fifo_queue:
            self._update_neg_sample_queue(batch_y)

        batch_data['pos-inds'] = torch.arange(batch_size).reshape(-1, 1)
        batch_data['targets'] = torch.zeros((batch_size, batch_y.shape[0]))
        for i in range(batch_size):
            self.mask[batch_data['y']['inds'][i]] = True
            batch_data['targets'][i][self.mask[batch_y].bool()] = 1.0
            self.mask[batch_data['y']['inds'][i]] = False

        return batch_y, batch_data

class XMCDataManager():
    def __init__(self, args):
        self.trn_X_Y = sp.load_npz(f'{args.DATA_DIR}/Y.trn.npz')
        self.tst_X_Y = sp.load_npz(f'{args.DATA_DIR}/Y.tst.npz')
        self.tst_filter_mat = load_filter_mat(f'{args.DATA_DIR}/filter_labels_test.txt', self.tst_X_Y.shape)
        self.trn_filter_mat = load_filter_mat(f'{args.DATA_DIR}/filter_labels_train.txt', self.trn_X_Y.shape)
        self.inv_prop = get_inv_prop(self.trn_X_Y, args.dataset)


        self.numy = args.numy = self.trn_X_Y.shape[1] # Number of labels
        self.trn_numx = self.trn_X_Y.shape[0] # Number of train data points 
        self.tst_numx = self.tst_X_Y.shape[0] # Number of test data points

        self.data_tokenization = args.data_tokenization
        self.tf_max_len = args.tf_max_len
        self.tf_token_type = args.tf_token_type = 'roberta-base' if 'roberta' in args.tf else 'bert-base-uncased' if 'bert' in args.tf else args.tf # Token type
        self.DATA_DIR = args.DATA_DIR
        self.num_val_points = args.num_val_points
        self.bsz = args.bsz

        if self.num_val_points > 0:
            if os.path.exists(f'{args.DATA_DIR}/val_inds_{args.num_val_points}.npy'): 
                self.val_inds = np.load(f'{args.DATA_DIR}/val_inds_{args.num_val_points}.npy')
            else: 
                self.val_inds = np.random.choice(np.arange(self.trn_numx), size=args.num_val_points, replace=False)
                np.save(f'{args.DATA_DIR}/val_inds_{args.num_val_points}.npy', self.val_inds)
            self.trn_inds = np.setdiff1d(np.arange(self.trn_numx), self.val_inds)
        else:
            self.trn_inds = self.val_inds = None

    def load_raw_texts(self):
        self.trnX = [x.strip() for x in open(f'{self.DATA_DIR}/raw/trn_X.txt')]
        self.tstX = [x.strip() for x in open(f'{self.DATA_DIR}/raw/tst_X.txt')]
        self.Y = [x.strip() for x in open(f'{self.DATA_DIR}/raw/Y.txt')]
        return self.trnX, self.tstX, self.Y

    def load_bow_fts(self, normalize=True):
        trn_X_Xf = sp.load_npz(f'{self.DATA_DIR}/X.trn.npz')
        tst_X_Xf = sp.load_npz(f'{self.DATA_DIR}/X.tst.npz')

        if normalize:
            sklearn.preprocessing.normalize(trn_X_Xf, copy=False)
            sklearn.preprocessing.normalize(tst_X_Xf, copy=False)

        self.trn_X_Xf = trn_X_Xf[self.trn_inds] if self.trn_inds is not None else trn_X_Xf
        self.val_X_Xf = trn_X_Xf[self.val_inds] if self.val_inds is not None else tst_X_Xf
        self.tst_X_Xf = tst_X_Xf

        return self.trn_X_Xf, self.val_X_Xf, self.tst_X_Xf

    def build_datasets(self):
        if self.data_tokenization == 'offline':
            self.trn_dataset = OfflineBertDataset(f'{self.DATA_DIR}/raw/trn_X.{self.tf_token_type}_{self.tf_max_len}.dat', self.trn_X_Y, self.tf_max_len, self.tf_token_type, sample=self.trn_inds, filter_mat=self.trn_filter_mat)
            self.val_dataset = OfflineBertDataset(f'{self.DATA_DIR}/raw/trn_X.{self.tf_token_type}_{self.tf_max_len}.dat', self.trn_X_Y, self.tf_max_len, self.tf_token_type, sample=self.val_inds, filter_mat=self.trn_filter_mat)
            self.tst_dataset = OfflineBertDataset(f'{self.DATA_DIR}/raw/tst_X.{self.tf_token_type}_{self.tf_max_len}.dat', self.tst_X_Y, self.tf_max_len, self.tf_token_type, sample = None, filter_mat=self.tst_filter_mat)
        elif self.data_tokenization == 'online':
            trnX = [x.strip() for x in open(f'{self.DATA_DIR}/raw/trn_X.txt').readlines()]
            tstX = [x.strip() for x in open(f'{self.DATA_DIR}/raw/tst_X.txt').readlines()]
            self.trn_dataset = OnlineBertDataset(trnX, self.trn_X_Y, self.tf_max_len, self.tf_token_type, sample=self.trn_inds, filter_mat=self.trn_filter_mat)
            self.val_dataset = OnlineBertDataset(trnX, self.trn_X_Y, self.tf_max_len, self.tf_token_type, sample=self.val_inds, filter_mat=self.trn_filter_mat)
            self.tst_dataset = OnlineBertDataset(tstX, self.tst_X_Y, self.tf_max_len, self.tf_token_type, sample=None, filter_mat=self.tst_filter_mat)
        else:
            raise Exception(f"Unrecongnized data_tokenization argument: {self.data_tokenization}")
        
        if self.num_val_points <= 0:
            self.val_dataset = self.tst_dataset

        return self.trn_dataset, self.val_dataset, self.tst_dataset

    def build_data_loaders(self):
        if not hasattr(self, "trn_dataset"):
            self.build_datasets()

        data_loader_args = {
            'batch_size': self.bsz,
            'num_workers': 4,
            'collate_fn': XMCCollator(self.trn_dataset),
            'shuffle': True,
            'pin_memory': True
        }

        self.trn_loader = torch.utils.data.DataLoader(self.trn_dataset, **data_loader_args)

        data_loader_args['shuffle'] = False
        data_loader_args['collate_fn'] = XMCCollator(self.val_dataset)
        data_loader_args['batch_size'] = 2*self.bsz
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, **data_loader_args)

        data_loader_args['collate_fn'] = XMCCollator(self.tst_dataset)
        self.tst_loader = torch.utils.data.DataLoader(self.tst_dataset, **data_loader_args)

        return self.trn_loader, self.val_loader, self.tst_loader

class XMCEmbedDataManager(XMCDataManager):
    def __init__(self, args):
        super().__init__(args)
        self.embed_id = args.embed_id
    
    def build_datasets(self):
        trn_embs = np.load(f'{self.DATA_DIR}/embeddings/{self.embed_id}/trn_embs.npy')
        tst_embs = np.load(f'{self.DATA_DIR}/embeddings/{self.embed_id}/tst_embs.npy')
        self.trn_dataset = SimpleDataset(trn_embs, self.trn_X_Y, sample=self.trn_inds, filter_mat=self.trn_filter_mat)
        self.val_dataset = SimpleDataset(trn_embs, self.trn_X_Y, sample=self.val_inds, filter_mat=self.trn_filter_mat)
        self.tst_dataset = SimpleDataset(tst_embs, self.tst_X_Y, sample=None, filter_mat=self.tst_filter_mat)
        if self.num_val_points <= 0:
            self.val_dataset = self.tst_dataset
        return self.trn_dataset, self.val_dataset, self.tst_dataset

class TwoTowerDataManager(XMCDataManager):
    def __init__(self, args):
        super().__init__(args)
        self.transpose_trn_dataset = args.transpose_trn_dataset
        self.neg_type = args.neg_type
        self.only_keep_trn_labels = args.only_keep_trn_labels if hasattr(args, 'only_keep_trn_labels') else False
        self.num_neg_samples = args.num_neg_samples if hasattr(args, 'num_neg_samples') else 1
        self.num_pos_samples = args.num_pos_samples if hasattr(args, 'num_pos_samples') else 1
        self.mixing_alpha = args.mixing_alpha if hasattr(args, 'mixing_alpha') else 1
        self.mixing_beta = args.mixing_beta if hasattr(args, 'mixing_beta') else 1
        self.use_fifo_queue = args.use_fifo_queue if hasattr(args, 'use_fifo_queue') else False
        self.fifo_queue_size = args.fifo_queue_size if hasattr(args, 'fifo_queue_size') else 2048
        self.trn_shorty = sp.load_npz(args.trn_shorty) if hasattr(args, 'trn_shorty') and os.path.exists(args.trn_shorty) else None

    def build_datasets(self):
        trnx_dataset, valx_dataset, tstx_dataset = super().build_datasets()
        if self.only_keep_trn_labels:
            lbl_sample = np.union1d(np.where(self.trn_X_Y.getnnz(0).ravel() > 0)[0], np.where(self.tst_X_Y.getnnz(0).ravel() > 0)[0])
            trnx_dataset.labels = trnx_dataset.labels[:, lbl_sample]
            tstx_dataset.labels = tstx_dataset.labels[:, lbl_sample]
            if self.num_val_points > 0:
                valx_dataset.labels = valx_dataset.labels[:, lbl_sample]
        else:
            lbl_sample = None

        if self.data_tokenization == 'offline':
            self.lbl_dataset = OfflineBertDataset(f'{self.DATA_DIR}/raw/Y.{self.tf_token_type}_{self.tf_max_len}.dat', self.trn_X_Y.T.tocsr(), self.tf_max_len, self.tf_token_type, sample=lbl_sample, filter_mat=None)
        elif self.data_tokenization == 'online':
            Y = [x.strip() for x in open(f'{self.DATA_DIR}/raw/Y.txt').readlines()]
            self.lbl_dataset = OnlineBertDataset(Y, self.trn_X_Y.T.tocsr(), self.tf_max_len, self.tf_token_type, sample=lbl_sample, filter_mat=None)
        else:
            raise Exception(f"Unrecongnized data_tokenization argument: {self.data_tokenization}")
    
        
        if self.transpose_trn_dataset:
            assert (self.trn_shorty is None) or (self.trn_shorty.shape == self.lbl_dataset.shape)
            # self.lbl_dataset.sample = np.where(self.lbl_dataset.labels.getnnz(1).ravel() > 0)[0]
            self.trn_dataset = TwoTowerDataset(self.lbl_dataset, trnx_dataset, self.trn_shorty)
        else:
            self.trn_dataset = TwoTowerDataset(trnx_dataset, self.lbl_dataset, self.trn_shorty)
        self.val_dataset = TwoTowerDataset(valx_dataset, self.lbl_dataset)
        self.tst_dataset = TwoTowerDataset(tstx_dataset, self.lbl_dataset)

        return self.trn_dataset, self.val_dataset, self.tst_dataset

    def build_data_loaders(self):
        if not hasattr(self, "trn_dataset"):
            self.build_datasets()

        print('neg_type:', self.neg_type)
        data_loader_args = {
            'batch_size': self.bsz,
            'num_workers': 4,
            'collate_fn': TwoTowerTrainCollator(self.trn_dataset, neg_type=self.neg_type, num_neg_samples=self.num_neg_samples, num_pos_samples=self.num_pos_samples, mixing_alpha=self.mixing_alpha, mixing_beta=self.mixing_beta, use_fifo_queue=self.use_fifo_queue, queue_size=self.fifo_queue_size),
            'shuffle': True,
            'pin_memory': True
        }

        self.trn_loader = torch.utils.data.DataLoader(self.trn_dataset, **data_loader_args)

        data_loader_args['shuffle'] = False
        data_loader_args['collate_fn'] = None
        data_loader_args['batch_size'] = 2*self.bsz
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, **data_loader_args)
        self.tst_loader = torch.utils.data.DataLoader(self.tst_dataset, **data_loader_args)

        return self.trn_loader, self.val_loader, self.tst_loader

class CrossDataManager(XMCDataManager):
    def __init__(self, args):
        super().__init__(args)
        self.transpose_trn_dataset = args.transpose_trn_dataset
        self.num_neg_samples = args.num_neg_samples if hasattr(args, 'num_neg_samples') else 1
        self.trn_shorty = sp.load_npz(args.trn_shorty) if hasattr(args, 'trn_shorty') and os.path.exists(args.trn_shorty) else None
        self.tst_shorty = sp.load_npz(args.tst_shorty) if hasattr(args, 'tst_shorty') and os.path.exists(args.tst_shorty) else None
        if self.num_val_points > 0 and self.trn_shorty is not None:
            self.val_shorty = self.trn_shorty[self.val_inds]
            self.trn_shorty = self.trn_shorty[self.trn_inds]
        else:
            self.val_shorty = self.tst_shorty

    def build_datasets(self):
        trnx_dataset, valx_dataset, tstx_dataset = super().build_datasets()
        if self.data_tokenization == 'offline':
            self.lbl_dataset = OfflineBertDataset(f'{self.DATA_DIR}/raw/Y.{self.tf_token_type}_{self.tf_max_len}.dat', self.trn_X_Y.T.tocsr(), self.tf_max_len, self.tf_token_type, sample=None, filter_mat=None)
        elif self.data_tokenization == 'online':
            Y = [x.strip() for x in open(f'{self.DATA_DIR}/raw/Y.txt').readlines()]
            self.lbl_dataset = OnlineBertDataset(Y, self.trn_X_Y.T.tocsr(), self.tf_max_len, self.tf_token_type, sample=None, filter_mat=None)
        else:
            raise Exception(f"Unrecongnized data_tokenization argument: {self.data_tokenization}")
        
        if self.transpose_trn_dataset:
            self.trn_dataset = CrossDataset(self.lbl_dataset, trnx_dataset, self.lbl_dataset.labels, iterate_over='labels')
            # TODO: make trn_shorty work with transpose
        else:
            self.trn_dataset = CrossDataset(trnx_dataset, self.lbl_dataset, self.trn_shorty, iterate_over='labels')
        self.val_dataset = CrossDataset(valx_dataset, self.lbl_dataset, self.val_shorty + valx_dataset.labels, iterate_over='shorty')
        self.tst_dataset = CrossDataset(tstx_dataset, self.lbl_dataset, self.tst_shorty, iterate_over='shorty')

        return self.trn_dataset, self.val_dataset, self.tst_dataset

    def build_data_loaders(self):
        if not hasattr(self, "trn_dataset"):
            self.build_datasets()

        data_loader_args = {
            'batch_size': self.bsz,
            'num_workers': 4,
            'collate_fn': CrossCollator(self.trn_dataset, num_neg_samples=self.num_neg_samples),
            'shuffle': True,
            'pin_memory': True
        }

        self.trn_loader = torch.utils.data.DataLoader(self.trn_dataset, **data_loader_args)

        data_loader_args['shuffle'] = False
        data_loader_args['batch_size'] = 4*self.bsz
        data_loader_args['collate_fn'] = CrossCollator(self.val_dataset, num_neg_samples=0)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, **data_loader_args)
        data_loader_args['collate_fn'] = CrossCollator(self.tst_dataset, num_neg_samples=0)
        self.tst_loader = torch.utils.data.DataLoader(self.tst_dataset, **data_loader_args)

        return self.trn_loader, self.val_loader, self.tst_loader

class XMCEvaluator:
    def __init__(self, args, data_source, data_manager: XMCDataManager, prefix='default'):
        self.eval_interval = args.eval_interval
        self.num_epochs = args.num_epochs
        self.track_metric = args.track_metric
        self.OUT_DIR = args.OUT_DIR
        self.save = args.save
        self.bsz = args.bsz
        self.eval_topk = args.eval_topk
        self.wandb_id = args.wandb_id if hasattr(args, "wandb_id") else None
        self.prefix = prefix

        self.data_source = data_source
        self.labels = data_source.labels if isinstance(data_source, torch.utils.data.Dataset) else data_source.dataset.labels
        self.filter_mat = data_source.filter_mat if isinstance(data_source, torch.utils.data.Dataset) else data_source.dataset.filter_mat
        self.inv_prop = data_manager.inv_prop
        self.best_score = -99999999

    def predict(self, net):
        score_mat = unwrap(net).predict(self.data_source, K=self.eval_topk, bsz=self.bsz)
        return score_mat

    def eval(self, score_mat, epoch=-1, loss=float('inf')):
        _filter(score_mat, self.filter_mat, copy=False)
        eval_name = f'{self.prefix}' + [f' {epoch}/{self.num_epochs}', ''][epoch < 0]
        metrics = compute_xmc_metrics(score_mat, self.labels, self.inv_prop, K=self.eval_topk, name=eval_name, disp=False)
        metrics.index.names = [self.wandb_id]
        if loss < float('inf'):  metrics['loss'] = ["%.4E"%loss]
        metrics.to_csv(open(f'{self.OUT_DIR}/{self.prefix}_metrics.tsv', 'a+'), sep='\t', header=(epoch <= 0))
        return metrics

    def predict_and_track_eval(self, net, epoch='-', loss=float('inf')):
        if epoch%self.eval_interval == 0 or epoch == (self.num_epochs-1):
            score_mat = self.predict(net)
            return self.track_eval(net, score_mat, epoch, loss)

    def track_eval(self, net, score_mat, epoch='-', loss=float('inf')):
        if score_mat is None: 
            return None
        
        metrics = self.eval(score_mat, epoch, loss)
        if metrics.iloc[0][self.track_metric] > self.best_score:
            self.best_score = metrics.iloc[0][self.track_metric]
            print(_c(f'Found new best model with {self.track_metric}: {"%.2f"%self.best_score}\n', attr='blue'))
            if self.save:
                sp.save_npz(f'{self.OUT_DIR}/{self.prefix}_score_mat.npz', score_mat)
                net.save(f'{self.OUT_DIR}/model.pt')
                net.save_safetensors(f'{self.OUT_DIR}/model.safetensors')
        return metrics

DATA_MANAGERS = {
    'xmc': XMCDataManager,
    'two-tower': TwoTowerDataManager,
    'xmc-embed': XMCEmbedDataManager,
    'cross': CrossDataManager,
}
            