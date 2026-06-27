import pickle
import torch
import numpy as np
import torch.nn as nn
import pdb

import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.optim as optim
import pdb
import torch.nn.functional as F
import math
from itertools import islice
import collections
import torch.nn.utils.rnn as rnn_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def collate_MIL(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    label = torch.LongTensor([item[1] for item in batch])
    return [img, label]

def collate_features(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    coords = np.vstack([item[1] for item in batch])
    return [img, coords]

def collate_MIL_padded(batch):
    """
    Custom collate function to pad 'features' tensors to the same number of patches.

    Args:
        batch: list of dicts with 'features' and 'labels'

    Returns:
        A dict with:
            'features': Padded tensor of shape (batch_size, max_num_patches, feature_dim)
            'labels': Tensor of shape (batch_size,)
            'mask': Tensor of shape (batch_size, max_num_patches) indicating valid patches
    """
    features = [item['features'] for item in batch]
    labels = torch.LongTensor([item['labels'] for item in batch])

    # Find max number of patches
    max_num_patches = max([f.size(0) for f in features])
    feature_dim = features[0].size(1)

    # Initialize padded features tensor and mask
    padded_features = torch.zeros((len(features), max_num_patches, feature_dim), dtype=features[0].dtype)
    mask = torch.zeros((len(features), max_num_patches), dtype=torch.bool)

    for i, f in enumerate(features):
        num_patches = f.size(0)
        padded_features[i, :num_patches, :] = f
        mask[i, :num_patches] = True

    return {'features': padded_features, 'labels': labels, 'mask': mask}

def get_simple_loader(dataset, batch_size=1, num_workers=1, collate_fn=None):
    kwargs = {'num_workers': 4, 'pin_memory': False, 'num_workers': num_workers} if device.type == "cuda" else {}
    if collate_fn is None:
        collate_fn = collate_MIL
    loader = DataLoader(dataset, batch_size=batch_size, sampler=SequentialSampler(dataset), collate_fn=collate_fn, **kwargs)
    return loader 

def get_split_loader(split_dataset, training=False, testing=False, weighted=False, collate_fn=None):
    """
        return either the validation loader or training loader 
    """
    kwargs = {'num_workers': 4} if device.type == "cuda" else {}
    if not testing:
        if training:
            if weighted:
                weights = make_weights_for_balanced_classes_split(split_dataset)
                loader = DataLoader(split_dataset, batch_size=1, sampler=WeightedRandomSampler(weights, len(weights)), collate_fn=collate_fn, **kwargs)    
            else:
                loader = DataLoader(split_dataset, batch_size=1, sampler=RandomSampler(split_dataset), collate_fn=collate_fn, **kwargs)
        else:
            loader = DataLoader(split_dataset, batch_size=1, sampler=SequentialSampler(split_dataset), collate_fn=collate_fn, **kwargs)
    
    else:
        ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace=False)
        loader = DataLoader(split_dataset, batch_size=1, sampler=SubsetSequentialSampler(ids), collate_fn=collate_fn, **kwargs )

    return loader

def get_optim(model, args):
    if args.opt == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    else:
        raise NotImplementedError
    return optimizer

def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)
    
    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n
    
    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)


def generate_split(cls_ids, val_num, test_num, samples, n_splits=5,
    seed=7, label_frac=1.0, custom_test_ids=None):
    indices = np.arange(samples).astype(int)
    
    if custom_test_ids is not None:
        indices = np.setdiff1d(indices, custom_test_ids)

    np.random.seed(seed)
    for i in range(n_splits):
        all_val_ids = []
        all_test_ids = []
        sampled_train_ids = []
        
        if custom_test_ids is not None: # pre-built test split, do not need to sample
            all_test_ids.extend(custom_test_ids)

        for c in range(len(val_num)):
            possible_indices = np.intersect1d(cls_ids[c], indices) #all indices of this class
            val_ids = np.random.choice(possible_indices, val_num[c], replace=False) # validation ids

            remaining_ids = np.setdiff1d(possible_indices, val_ids) #indices of this class left after validation
            all_val_ids.extend(val_ids)

            if custom_test_ids is None: # sample test split

                test_ids = np.random.choice(remaining_ids, test_num[c], replace=False)
                remaining_ids = np.setdiff1d(remaining_ids, test_ids)
                all_test_ids.extend(test_ids)

            if label_frac == 1:
                sampled_train_ids.extend(remaining_ids)
            
            else:
                sample_num  = math.ceil(len(remaining_ids) * label_frac)
                slice_ids = np.arange(sample_num)
                sampled_train_ids.extend(remaining_ids[slice_ids])

        yield sampled_train_ids, all_val_ids, all_test_ids


def nth(iterator, n, default=None):
    if n is None:
        return collections.deque(iterator, maxlen=0)
    else:
        return next(islice(iterator, n, None), default)

def calculate_error(Y_hat, Y):
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

    return error

def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))                                           
    weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
    weight = [0] * int(N)                                           
    for idx in range(len(dataset)):   
        y = dataset.getlabel(idx)                        
        weight[idx] = weight_per_class[y]                                  

    return torch.DoubleTensor(weight)

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
        
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)



def get_loss_function(loss_name):
    if loss_name == 'svm':
        return nn.MultiMarginLoss()
    elif loss_name == 'ce':
        return nn.CrossEntropyLoss()
    elif loss_name == 'None':
        return None
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")
