# import torch
# import os
# import glob
# import h5py
# import logging
# import numpy as np
# import pandas as pd

# from torch.utils.data import Dataset
# from utils.utils import generate_split, nth

# logger = logging.getLogger(__name__)


# def save_splits(split_datasets, column_keys, filename, boolean_style=False):
#     splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
#     if not boolean_style:
#         df = pd.concat(splits, ignore_index=True, axis=1)
#         df.columns = column_keys
#     else:
#         df = pd.concat(splits, ignore_index=True, axis=0)
#         index = df.values.tolist()
#         one_hot = np.eye(len(split_datasets)).astype(bool)
#         bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
#         df = pd.DataFrame(bool_array, index=index, columns=['train', 'val', 'test'])

#     df.to_csv(filename, index=False)
#     logger.info(f"Splits saved to {filename}")


# class Generic_WSI_Classification_Dataset(Dataset):
#     def __init__(
#         self,
#         csv_path='dataset_csv/ccrcc_clean.csv',
#         shuffle=False,
#         seed=7,
#         print_info=True,
#         label_dict={},
#         filter_dict={},
#         ignore=[],
#         patient_strat=False,
#         label_col=None,
#         patient_voting='max'
#     ):
#         self.label_dict = label_dict
#         self.num_classes = len(set(self.label_dict.values()))
#         self.seed = seed
#         self.print_info = print_info
#         self.patient_strat = patient_strat
#         self.train_ids, self.val_ids, self.test_ids = (None, None, None)
#         self.data_dir = None

#         if not label_col:
#             label_col = 'label'
#         self.label_col = label_col

#         if not os.path.isfile(csv_path):
#             logger.error(f"CSV file not found: {csv_path}")
#             raise FileNotFoundError(f"CSV file not found: {csv_path}")

#         slide_data = pd.read_csv(csv_path)
#         slide_data = self.filter_df(slide_data, filter_dict)
#         slide_data = self.df_prep(slide_data, self.label_dict, ignore, self.label_col)

#         if shuffle:
#             np.random.seed(seed)
#             slide_data = slide_data.sample(frac=1).reset_index(drop=True)
#             logger.info(f"Data shuffled with seed {seed}")

#         self.slide_data = slide_data
#         self.patient_data_prep(patient_voting)
#         self.cls_ids_prep()

#         if print_info:
#             self.summarize()

#     def cls_ids_prep(self):
#         self.patient_cls_ids = [[] for _ in range(self.num_classes)]
#         for i in range(self.num_classes):
#             self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

#         self.slide_cls_ids = [[] for _ in range(self.num_classes)]
#         for i in range(self.num_classes):
#             self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

#     def patient_data_prep(self, patient_voting='max'):
#         patient_col = 'case_id' if 'case_id' in self.slide_data.columns else 'slide_id'
#         patients = np.unique(self.slide_data[patient_col].astype(str).values)
#         patient_labels = []

#         for p in patients:
#             locations = self.slide_data[self.slide_data[patient_col].astype(str) == p].index.tolist()
#             assert len(locations) > 0, f"No data found for patient id: {p}"
#             label = self.slide_data.loc[locations, 'label'].values

#             if patient_voting == 'max':
#                 label = label.max()
#             elif patient_voting == 'maj':
#                 values, counts = np.unique(label, return_counts=True)
#                 label = values[np.argmax(counts)]
#             else:
#                 raise NotImplementedError(f"Unsupported patient_voting method: {patient_voting}")

#             patient_labels.append(label)

#         self.patient_data = {
#             patient_col: patients,
#             'label': np.array(patient_labels)
#         }

#     @staticmethod
#     def df_prep(data, label_dict, ignore, label_col):
#         if label_col != 'label':
#             data['label'] = data[label_col].copy()

#         data = data[data['label'] != 'label']
#         mask = data['label'].isin(ignore)
#         data = data[~mask]
#         data.reset_index(drop=True, inplace=True)

#         for i in data.index:
#             key = str(data.loc[i, 'label'])
#             assert key in label_dict, f"Label '{key}' not found in label_dict"
#             data.at[i, 'label'] = label_dict[key]

#         data['label'] = data['label'].astype(int)
#         return data

#     def filter_df(self, df, filter_dict={}):
#         if len(filter_dict) > 0:
#             filter_mask = np.full(len(df), True, bool)
#             for key, val in filter_dict.items():
#                 mask = df[key].isin(val)
#                 filter_mask = np.logical_and(filter_mask, mask)
#             df = df[filter_mask]
#         return df

#     def __len__(self):
#         if self.patient_strat:
#             patient_key = 'case_id' if 'case_id' in self.patient_data else 'slide_id'
#             return len(self.patient_data[patient_key])
#         return len(self.slide_data)

#     def summarize(self):
#         logger.info(f"Label column: {self.label_col}")
#         logger.info(f"Label dictionary: {self.label_dict}")
#         logger.info(f"Number of classes: {self.num_classes}")
#         logger.info(f"Slide-level counts:\n{self.slide_data['label'].value_counts(sort=False)}")
#         for i in range(self.num_classes):
#             logger.info(f'Patient-LVL; Number of samples registered in class {i}: {self.patient_cls_ids[i].shape[0]}')
#             logger.info(f'Slide-LVL; Number of samples registered in class {i}: {self.slide_cls_ids[i].shape[0]}')

#     def create_splits(self, k=3, val_num=(25, 25), test_num=(40, 40), label_frac=1.0, custom_test_ids=None):
#         settings = {
#             'n_splits': k,
#             'val_num': val_num,
#             'test_num': test_num,
#             'label_frac': label_frac,
#             'seed': self.seed,
#             'custom_test_ids': custom_test_ids
#         }

#         if self.patient_strat:
#             patient_key = 'case_id' if 'case_id' in self.patient_data else 'slide_id'
#             settings.update({'cls_ids': self.patient_cls_ids, 'samples': len(self.patient_data[patient_key])})
#         else:
#             settings.update({'cls_ids': self.slide_cls_ids, 'samples': len(self.slide_data)})

#         self.split_gen = generate_split(**settings)

#     def set_splits(self, start_from=None):
#         if start_from:
#             ids = nth(self.split_gen, start_from)
#         else:
#             ids = next(self.split_gen)

#         if self.patient_strat:
#             patient_key = 'case_id' if 'case_id' in self.patient_data else 'slide_id'
#             slide_ids = [[] for _ in range(len(ids))]

#             for split in range(len(ids)):
#                 for idx in ids[split]:
#                     patient_id = self.patient_data[patient_key][idx]
#                     slide_indices = self.slide_data[self.slide_data[patient_key].astype(str) == str(patient_id)].index.tolist()
#                     slide_ids[split].extend(slide_indices)

#             self.train_ids, self.val_ids, self.test_ids = slide_ids[0], slide_ids[1], slide_ids[2]
#         else:
#             self.train_ids, self.val_ids, self.test_ids = ids

#     def get_split_from_df(self, all_splits, split_key='train'):
#         split = all_splits[split_key]
#         split = split.dropna().reset_index(drop=True)

#         if len(split) > 0:
#             mask = self.slide_data['slide_id'].astype(str).isin(split.astype(str).tolist())
#             df_slice = self.slide_data[mask].reset_index(drop=True)
#             split = Generic_Split(
#                 df_slice,
#                 data_dir=self.data_dir,
#                 num_classes=self.num_classes,
#                 cache_in_ram=getattr(self, 'cache_in_ram', False)
#             )
#         else:
#             split = None
#             logger.warning(f"No samples found for split: {split_key}")

#         return split

#     def get_merged_split_from_df(self, all_splits, split_keys=['train']):
#         merged_split = []
#         for split_key in split_keys:
#             split = all_splits[split_key]
#             split = split.dropna().reset_index(drop=True).astype(str).tolist()
#             merged_split.extend(split)

#         if len(merged_split) > 0:
#             mask = self.slide_data['slide_id'].astype(str).isin(merged_split)
#             df_slice = self.slide_data[mask].reset_index(drop=True)
#             split = Generic_Split(
#                 df_slice,
#                 data_dir=self.data_dir,
#                 num_classes=self.num_classes,
#                 cache_in_ram=getattr(self, 'cache_in_ram', False)
#             )
#         else:
#             split = None
#             logger.warning(f"No samples found for merged splits: {split_keys}")

#         return split

#     def return_splits(self, from_id=True, csv_path=None):
#         if from_id:
#             if self.train_ids is not None and len(self.train_ids) > 0:
#                 train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
#                 train_split = Generic_Split(
#                     train_data,
#                     data_dir=self.data_dir,
#                     num_classes=self.num_classes,
#                     cache_in_ram=getattr(self, 'cache_in_ram', False)
#                 )
#             else:
#                 train_split = None

#             if self.val_ids is not None and len(self.val_ids) > 0:
#                 val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
#                 val_split = Generic_Split(
#                     val_data,
#                     data_dir=self.data_dir,
#                     num_classes=self.num_classes,
#                     cache_in_ram=getattr(self, 'cache_in_ram', False)
#                 )
#             else:
#                 val_split = None

#             if self.test_ids is not None and len(self.test_ids) > 0:
#                 test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
#                 test_split = Generic_Split(
#                     test_data,
#                     data_dir=self.data_dir,
#                     num_classes=self.num_classes,
#                     cache_in_ram=getattr(self, 'cache_in_ram', False)
#                 )
#             else:
#                 test_split = None
#         else:
#             assert csv_path, "csv_path must be provided if from_id is False."
#             if not os.path.isfile(csv_path):
#                 logger.error(f"Split CSV file not found: {csv_path}")
#                 raise FileNotFoundError(f"Split file not found: {csv_path}")

#             all_splits = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype)
#             train_split = self.get_split_from_df(all_splits, 'train')
#             val_split = self.get_split_from_df(all_splits, 'val')
#             test_split = self.get_split_from_df(all_splits, 'test')

#         return train_split, val_split, test_split

#     def get_list(self, ids):
#         return self.slide_data['slide_id'][ids]

#     def getlabel(self, ids):
#         return self.slide_data['label'][ids]

#     def __getitem__(self, idx):
#         return None

#     def test_split_gen(self, return_descriptor=False):
#         if return_descriptor:
#             index = [list(self.label_dict.keys())[list(self.label_dict.values()).index(i)] for i in range(self.num_classes)]
#             columns = ['train', 'val', 'test']
#             df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.int32), index=index, columns=columns)

#         count = len(self.train_ids) if self.train_ids is not None else 0
#         logger.info(f'\nNumber of training samples: {count}')
#         if self.train_ids is not None and len(self.train_ids) > 0:
#             labels = self.getlabel(self.train_ids)
#             unique, counts = np.unique(labels, return_counts=True)
#             for u in range(len(unique)):
#                 logger.info(f'Number of samples in class {unique[u]}: {counts[u]}')
#                 if return_descriptor:
#                     df.loc[index[u], 'train'] = counts[u]

#         count = len(self.val_ids) if self.val_ids is not None else 0
#         logger.info(f'\nNumber of validation samples: {count}')
#         if self.val_ids is not None and len(self.val_ids) > 0:
#             labels = self.getlabel(self.val_ids)
#             unique, counts = np.unique(labels, return_counts=True)
#             for u in range(len(unique)):
#                 logger.info(f'Number of samples in class {unique[u]}: {counts[u]}')
#                 if return_descriptor:
#                     df.loc[index[u], 'val'] = counts[u]

#         count = len(self.test_ids) if self.test_ids is not None else 0
#         logger.info(f'\nNumber of test samples: {count}')
#         if self.test_ids is not None and len(self.test_ids) > 0:
#             labels = self.getlabel(self.test_ids)
#             unique, counts = np.unique(labels, return_counts=True)
#             for u in range(len(unique)):
#                 logger.info(f'Number of samples in class {unique[u]}: {counts[u]}')
#                 if return_descriptor:
#                     df.loc[index[u], 'test'] = counts[u]

#         if self.train_ids is not None and self.test_ids is not None:
#             assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
#         if self.train_ids is not None and self.val_ids is not None:
#             assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
#         if self.val_ids is not None and self.test_ids is not None:
#             assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0

#         if return_descriptor:
#             return df

#     def save_split(self, filename):
#         train_split = self.get_list(self.train_ids) if self.train_ids is not None else pd.Series(dtype=str)
#         val_split = self.get_list(self.val_ids) if self.val_ids is not None else pd.Series(dtype=str)
#         test_split = self.get_list(self.test_ids) if self.test_ids is not None else pd.Series(dtype=str)

#         df_tr = pd.DataFrame({'train': train_split})
#         df_v = pd.DataFrame({'val': val_split})
#         df_t = pd.DataFrame({'test': test_split})
#         df = pd.concat([df_tr, df_v, df_t], axis=1)
#         df.to_csv(filename, index=False)
#         logger.info(f"Split saved to {filename}")


# class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
#     def __init__(self, data_dir, cache_in_ram=False, **kwargs):
#         super(Generic_MIL_Dataset, self).__init__(**kwargs)
#         self.data_dir = data_dir
#         self.use_h5 = False
#         self.cache_in_ram = cache_in_ram
#         self.feature_cache = {}
#         self.file_map = self._build_file_map()

#     def load_from_h5(self, toggle):
#         self.use_h5 = toggle

#     def _candidate_keys(self, slide_id, case_id=None):
#         keys = []

#         slide_id = str(slide_id)
#         keys.append(slide_id)

#         if case_id is not None:
#             case_id = str(case_id)
#             if case_id not in keys:
#                 keys.append(case_id)

#         # TCGA patient id fallback
#         slide_tcga12 = slide_id[:12]
#         if slide_tcga12 not in keys:
#             keys.append(slide_tcga12)

#         if case_id is not None:
#             case_tcga12 = str(case_id)[:12]
#             if case_tcga12 not in keys:
#                 keys.append(case_tcga12)

#         return keys

#     def _build_file_map(self):
#         if self.use_h5:
#             return None
#         if isinstance(self.data_dir, dict) or self.data_dir is None:
#             return None

#         file_map = {}

#         for _, row in self.slide_data.iterrows():
#             slide_id = str(row['slide_id'])
#             case_id = str(row['case_id']) if 'case_id' in self.slide_data.columns else None

#             for key in self._candidate_keys(slide_id, case_id):
#                 if key in file_map:
#                     continue
#                 pattern = os.path.join(self.data_dir, f"{key}*.pt")
#                 matched_files = sorted(glob.glob(pattern))
#                 if len(matched_files) > 0:
#                     file_map[key] = matched_files

#         logger.info(f"Indexed {len(file_map)} keys in {self.data_dir}")
#         return file_map

#     def _load_pt_features(self, slide_id, case_id=None):
#         cache_key = str(slide_id)
#         if cache_key in self.feature_cache:
#             return self.feature_cache[cache_key]

#         matched_files = []
#         matched_key = None
#         tried_keys = self._candidate_keys(slide_id, case_id)

#         for key in tried_keys:
#             if self.file_map is not None:
#                 matched_files = self.file_map.get(key, [])
#             else:
#                 pattern = os.path.join(self.data_dir, f"{key}*.pt")
#                 matched_files = sorted(glob.glob(pattern))

#             if len(matched_files) > 0:
#                 matched_key = key
#                 break

#         if len(matched_files) == 0:
#             logger.error(
#                 f"No .pt file found for slide_id: {slide_id}, case_id: {case_id}, tried_keys={tried_keys}"
#             )
#             raise FileNotFoundError(
#                 f"No .pt file found for slide_id: {slide_id}, case_id: {case_id}"
#             )

#         if matched_key != str(slide_id):
#             logger.info(f"Matched slide_id={slide_id} using fallback key={matched_key}")

#         if len(matched_files) > 1:
#             features_list = []
#             for file in matched_files:
#                 features = torch.load(file, map_location='cpu', weights_only=True)
#                 features_list.append(features)
#             features = torch.cat(features_list, dim=0)
#         else:
#             features = torch.load(matched_files[0], map_location='cpu', weights_only=True)

#         if self.cache_in_ram:
#             self.feature_cache[cache_key] = features

#         return features

#     def preload_features(self):
#         if not self.cache_in_ram:
#             return
#         if self.file_map is None:
#             return

#         logger.info(f"Preloading features into RAM for {len(self.slide_data)} slides ...")
#         for _, row in self.slide_data.iterrows():
#             slide_id = str(row['slide_id'])
#             case_id = str(row['case_id']) if 'case_id' in self.slide_data.columns else None
#             _ = self._load_pt_features(slide_id, case_id=case_id)
#         logger.info(f"RAM cache ready: {len(self.feature_cache)} slides")

#     def __getitem__(self, idx):
#         slide_id = str(self.slide_data['slide_id'][idx])
#         label = self.slide_data['label'][idx]
#         case_id = str(self.slide_data['case_id'][idx]) if 'case_id' in self.slide_data.columns else None

#         if isinstance(self.data_dir, dict):
#             source = self.slide_data['source'][idx]
#             data_dir = self.data_dir[source]
#         else:
#             data_dir = self.data_dir

#         if not self.use_h5:
#             if data_dir:
#                 features = self._load_pt_features(slide_id, case_id=case_id)
#                 return {'features': features, 'labels': label}
#             else:
#                 return {'features': slide_id, 'labels': label}
#         else:
#             full_path = os.path.join(data_dir, 'h5_files', f"{slide_id}.h5")
#             with h5py.File(full_path, 'r') as hdf5_file:
#                 features = hdf5_file['features'][:]
#                 coords = hdf5_file['coords'][:]

#             features = torch.from_numpy(features)
#             return {'features': features, 'labels': label, 'coords': coords}


# class Generic_Split(Dataset):
#     def __init__(self, slide_data, data_dir=None, num_classes=2, cache_in_ram=False):
#         self.use_h5 = False
#         self.slide_data = slide_data
#         self.data_dir = data_dir
#         self.num_classes = num_classes
#         self.cache_in_ram = cache_in_ram
#         self.feature_cache = {}

#         self.slide_cls_ids = [[] for _ in range(self.num_classes)]
#         for i in range(self.num_classes):
#             self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

#         self.file_map = self._build_file_map()

#     def _candidate_keys(self, slide_id, case_id=None):
#         keys = []

#         slide_id = str(slide_id)
#         keys.append(slide_id)

#         if case_id is not None:
#             case_id = str(case_id)
#             if case_id not in keys:
#                 keys.append(case_id)

#         slide_tcga12 = slide_id[:12]
#         if slide_tcga12 not in keys:
#             keys.append(slide_tcga12)

#         if case_id is not None:
#             case_tcga12 = str(case_id)[:12]
#             if case_tcga12 not in keys:
#                 keys.append(case_tcga12)

#         return keys

#     def _build_file_map(self):
#         if self.use_h5:
#             return None
#         if isinstance(self.data_dir, dict) or self.data_dir is None:
#             return None

#         file_map = {}

#         for _, row in self.slide_data.iterrows():
#             slide_id = str(row['slide_id'])
#             case_id = str(row['case_id']) if 'case_id' in self.slide_data.columns else None

#             for key in self._candidate_keys(slide_id, case_id):
#                 if key in file_map:
#                     continue
#                 pattern = os.path.join(self.data_dir, f"{key}*.pt")
#                 matched_files = sorted(glob.glob(pattern))
#                 if len(matched_files) > 0:
#                     file_map[key] = matched_files

#         logger.info(f"Indexed {len(file_map)} split keys in {self.data_dir}")
#         return file_map

#     def _load_pt_features(self, slide_id, case_id=None):
#         cache_key = str(slide_id)
#         if cache_key in self.feature_cache:
#             return self.feature_cache[cache_key]

#         matched_files = []
#         matched_key = None
#         tried_keys = self._candidate_keys(slide_id, case_id)

#         for key in tried_keys:
#             if self.file_map is not None:
#                 matched_files = self.file_map.get(key, [])
#             else:
#                 pattern = os.path.join(self.data_dir, f"{key}*.pt")
#                 matched_files = sorted(glob.glob(pattern))

#             if len(matched_files) > 0:
#                 matched_key = key
#                 break

#         if len(matched_files) == 0:
#             logger.error(
#                 f"No .pt file found for slide_id: {slide_id}, case_id: {case_id}, tried_keys={tried_keys}"
#             )
#             raise FileNotFoundError(
#                 f"No .pt file found for slide_id: {slide_id}, case_id: {case_id}"
#             )

#         if matched_key != str(slide_id):
#             logger.info(f"Matched split slide_id={slide_id} using fallback key={matched_key}")

#         if len(matched_files) > 1:
#             features_list = []
#             for file in matched_files:
#                 features = torch.load(file, map_location='cpu', weights_only=True)
#                 features_list.append(features)
#             features = torch.cat(features_list, dim=0)
#         else:
#             features = torch.load(matched_files[0], map_location='cpu', weights_only=True)

#         if self.cache_in_ram:
#             self.feature_cache[cache_key] = features

#         return features

#     def preload_features(self):
#         if not self.cache_in_ram:
#             return
#         if self.file_map is None:
#             return

#         logger.info(f"Preloading split features into RAM for {len(self.slide_data)} slides ...")
#         for _, row in self.slide_data.iterrows():
#             slide_id = str(row['slide_id'])
#             case_id = str(row['case_id']) if 'case_id' in self.slide_data.columns else None
#             _ = self._load_pt_features(slide_id, case_id=case_id)
#         logger.info(f"Split RAM cache ready: {len(self.feature_cache)} slides")

#     def __len__(self):
#         return len(self.slide_data)

#     def __getitem__(self, idx):
#         slide_id = str(self.slide_data['slide_id'][idx])
#         label = self.slide_data['label'][idx]
#         case_id = str(self.slide_data['case_id'][idx]) if 'case_id' in self.slide_data.columns else None

#         if isinstance(self.data_dir, dict):
#             source = self.slide_data['source'][idx]
#             data_dir = self.data_dir[source]
#         else:
#             data_dir = self.data_dir

#         if not self.use_h5:
#             if data_dir:
#                 features = self._load_pt_features(slide_id, case_id=case_id)
#                 return {'features': features, 'labels': label}
#             else:
#                 return {'features': slide_id, 'labels': label}
#         else:
#             full_path = os.path.join(data_dir, 'h5_files', f"{slide_id}.h5")
#             with h5py.File(full_path, 'r') as hdf5_file:
#                 features = hdf5_file['features'][:]
#                 coords = hdf5_file['coords'][:]

#             features = torch.from_numpy(features)
#             return {'features': features, 'labels': label, 'coords': coords}


# dataset_modules/dataset_generic.py
import torch
import os

import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle
from scipy import stats
import glob

from torch.utils.data import Dataset
import h5py

from utils.utils import generate_split, nth
import logging

from collections import defaultdict

# Get logger
logger = logging.getLogger(__name__)

def save_splits(split_datasets, column_keys, filename, boolean_style=False):
    splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
    if not boolean_style:
        df = pd.concat(splits, ignore_index=True, axis=1)
        df.columns = column_keys
    else:
        df = pd.concat(splits, ignore_index=True, axis=0)
        index = df.values.tolist()
        one_hot = np.eye(len(split_datasets)).astype(bool)
        bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
        df = pd.DataFrame(bool_array, index=index, columns=['train', 'val', 'test'])

    df.to_csv(filename, index=False)  # Ensure not to save index column
    logger.info(f"Splits saved to {filename}")

class Generic_WSI_Classification_Dataset(Dataset):
    def __init__(self,
                 csv_path='dataset_csv/ccrcc_clean.csv',
                 shuffle=False,
                 seed=7,
                 print_info=True,
                 label_dict={},
                 filter_dict={},
                 ignore=[],
                 patient_strat=False,
                 label_col=None,
                 patient_voting='max'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            shuffle (boolean): Whether to shuffle
            seed (int): random seed for shuffling the data
            print_info (boolean): Whether to print a summary of the dataset
            label_dict (dict): Dictionary with key, value pairs for converting str labels to int
            ignore (list): List containing class labels to ignore
        """
        self.label_dict = label_dict
        self.num_classes = len(set(self.label_dict.values()))
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids = (None, None, None)
        self.data_dir = None
        if not label_col:
            label_col = 'label'
        self.label_col = label_col

        # Read CSV
        if not os.path.isfile(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        slide_data = pd.read_csv(csv_path)
        slide_data = self.filter_df(slide_data, filter_dict)
        slide_data = self.df_prep(slide_data, self.label_dict, ignore, self.label_col)

        ### Shuffle data
        if shuffle:
            np.random.seed(seed)
            slide_data = slide_data.sample(frac=1).reset_index(drop=True)
            logger.info(f"Data shuffled with seed {seed}")

        self.slide_data = slide_data

        self.patient_data_prep(patient_voting)
        self.cls_ids_prep()

        if print_info:
            self.summarize()

    def cls_ids_prep(self):
        # Store ids corresponding each class at the patient or case level
        self.patient_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

        # Store ids corresponding each class at the slide level
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    # def patient_data_prep(self, patient_voting='max'):
    #     logger.debug(f"Processing patient_voting: {patient_voting}")
    #     patients = np.unique(np.array(self.slide_data['slide_id']))
    #     patient_labels = []

    #     for p in patients:
    #         locations = self.slide_data[self.slide_data['slide_id'] == p].index.tolist()
    #         assert len(locations) > 0, f"No data found for slide_id: {p}"
    #         label = self.slide_data['label'][locations].values
    #         if patient_voting == 'max':
    #             label = label.max()
    #         elif patient_voting == 'maj':
    #             values, counts = np.unique(label, return_counts=True)
    #             label = values[np.argmax(counts)]
    #         else:
    #             raise NotImplementedError(f"Unsupported patient_voting method: {patient_voting}")
    #         patient_labels.append(label)

    #     self.patient_data = {'slide_id': patients, 'label': np.array(patient_labels)}
    
    def patient_data_prep(self, patient_voting='max'):
        patient_col = 'case_id' if 'case_id' in self.slide_data.columns else 'slide_id'
        patients = np.unique(self.slide_data[patient_col].astype(str).values)
        patient_labels = []

        for p in patients:
            locations = self.slide_data[self.slide_data[patient_col].astype(str) == p].index.tolist()
            assert len(locations) > 0, f"No data found for patient id: {p}"
            label = self.slide_data.loc[locations, 'label'].values

            if patient_voting == 'max':
                label = label.max()
            elif patient_voting == 'maj':
                values, counts = np.unique(label, return_counts=True)
                label = values[np.argmax(counts)]
            else:
                raise NotImplementedError

            patient_labels.append(label)

        self.patient_data = {
            patient_col: patients,
            'label': np.array(patient_labels)
        }


    @staticmethod
    def df_prep(data, label_dict, ignore, label_col):
        if label_col != 'label':
            data['label'] = data[label_col].copy()  # Create 'label' column if needed
        # Filter out any rows where label is 'label' to avoid misreading header as data
        data = data[data['label'] != 'label']
        mask = data['label'].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)

        # 在这里将读取到的label转为字符串，以便与label_dict中的键匹配
        for i in data.index:
            key = str(data.loc[i, 'label'])  # 确保key为字符串
            assert key in label_dict, f"Label '{key}' not found in label_dict"
            data.at[i, 'label'] = label_dict[key]

        data['label'] = data['label'].astype(int)
        return data

    def filter_df(self, df, filter_dict={}):
        if len(filter_dict) > 0:
            filter_mask = np.full(len(df), True, bool)
            for key, val in filter_dict.items():
                mask = df[key].isin(val)
                filter_mask = np.logical_and(filter_mask, mask)
            df = df[filter_mask]
        return df

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['slide_id'])
        else:
            return len(self.slide_data)

    def summarize(self):
        logger.info(f"Label column: {self.label_col}")
        logger.info(f"Label dictionary: {self.label_dict}")
        logger.info(f"Number of classes: {self.num_classes}")
        logger.info(f"Slide-level counts:\n{self.slide_data['label'].value_counts(sort=False)}")
        for i in range(self.num_classes):
            logger.info(f'Patient-LVL; Number of samples registered in class {i}: {self.patient_cls_ids[i].shape[0]}')
            logger.info(f'Slide-LVL; Number of samples registered in class {i}: {self.slide_cls_ids[i].shape[0]}')

    def create_splits(self, k=3, val_num=(25, 25), test_num=(40, 40), label_frac=1.0, custom_test_ids=None):
        settings = {
            'n_splits': k,
            'val_num': val_num,
            'test_num': test_num,
            'label_frac': label_frac,
            'seed': self.seed,
            'custom_test_ids': custom_test_ids
        }

        if self.patient_strat:
            settings.update({'cls_ids': self.patient_cls_ids, 'samples': len(self.patient_data['slide_id'])})
        else:
            settings.update({'cls_ids': self.slide_cls_ids, 'samples': len(self.slide_data)})

        self.split_gen = generate_split(**settings)

    def set_splits(self, start_from=None):
        if start_from:
            ids = nth(self.split_gen, start_from)
        else:
            ids = next(self.split_gen)

        if self.patient_strat:
            slide_ids = [[] for i in range(len(ids))]

            for split in range(len(ids)):
                for idx in ids[split]:
                    slide_id = self.patient_data['slide_id'][idx]
                    slide_indices = self.slide_data[self.slide_data['slide_id'] == slide_id].index.tolist()
                    slide_ids[split].extend(slide_indices)

            self.train_ids, self.val_ids, self.test_ids = slide_ids[0], slide_ids[1], slide_ids[2]
        else:
            self.train_ids, self.val_ids, self.test_ids = ids

    def get_split_from_df(self, all_splits, split_key='train'):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].reset_index(drop=True)
            #split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes)
            split = Generic_Split(df_slice,data_dir=self.data_dir,num_classes=self.num_classes,cache_in_ram=getattr(self, 'cache_in_ram', False))
        else:
            split = None
            logger.warning(f"No samples found for split: {split_key}")

        return split

    def get_merged_split_from_df(self, all_splits, split_keys=['train']):
        merged_split = []
        for split_key in split_keys:
            split = all_splits[split_key]
            split = split.dropna().reset_index(drop=True).tolist()
            merged_split.extend(split)

        if len(merged_split) > 0:
            mask = self.slide_data['slide_id'].isin(merged_split)
            df_slice = self.slide_data[mask].reset_index(drop=True)
            #split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes)
            split = Generic_Split(df_slice,data_dir=self.data_dir,num_classes=self.num_classes,cache_in_ram=getattr(self, 'cache_in_ram', False))
        else:
            split = None
            logger.warning(f"No samples found for merged splits: {split_keys}")

        return split

    def return_splits(self, from_id=True, csv_path=None):
        if from_id:
            if self.train_ids is not None and len(self.train_ids) > 0:
                train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
                #train_split = Generic_Split(train_data, data_dir=self.data_dir, num_classes=self.num_classes)
                train_split = Generic_Split(train_data,data_dir=self.data_dir,num_classes=self.num_classes,cache_in_ram=getattr(self, 'cache_in_ram', False))
            else:
                train_split = None

            if self.val_ids is not None and len(self.val_ids) > 0:
                val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
                #val_split = Generic_Split(val_data, data_dir=self.data_dir, num_classes=self.num_classes)
                val_split = Generic_Split(val_data,data_dir=self.data_dir,num_classes=self.num_classes,cache_in_ram=getattr(self, 'cache_in_ram', False))
            else:
                val_split = None

            if self.test_ids is not None and len(self.test_ids) > 0:
                test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
                #test_split = Generic_Split(test_data, data_dir=self.data_dir, num_classes=self.num_classes)
                test_split = Generic_Split(test_data,data_dir=self.data_dir,num_classes=self.num_classes,cache_in_ram=getattr(self, 'cache_in_ram', False))
            else:
                test_split = None
        else:
            assert csv_path, "csv_path must be provided if from_id is False."
            if not os.path.isfile(csv_path):
                logger.error(f"Split CSV file not found: {csv_path}")
                raise FileNotFoundError(f"Split file not found: {csv_path}")
            all_splits = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype)
            train_split = self.get_split_from_df(all_splits, 'train')
            val_split = self.get_split_from_df(all_splits, 'val')
            test_split = self.get_split_from_df(all_splits, 'test')

        return train_split, val_split, test_split

    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def __getitem__(self, idx):
        return None

    def test_split_gen(self, return_descriptor=False):
        if return_descriptor:
            index = [list(self.label_dict.keys())[list(self.label_dict.values()).index(i)] for i in range(self.num_classes)]
            columns = ['train', 'val', 'test']
            df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.int32), index=index, columns=columns)

        count = len(self.train_ids) if self.train_ids is not None else 0
        logger.info(f'\nNumber of training samples: {count}')
        if self.train_ids is not None and len(self.train_ids) > 0:
            labels = self.getlabel(self.train_ids)
            unique, counts = np.unique(labels, return_counts=True)
            for u in range(len(unique)):
                logger.info(f'Number of samples in class {unique[u]}: {counts[u]}')
                if return_descriptor:
                    df.loc[index[u], 'train'] = counts[u]

        count = len(self.val_ids) if self.val_ids is not None else 0
        logger.info(f'\nNumber of validation samples: {count}')
        if self.val_ids is not None and len(self.val_ids) > 0:
            labels = self.getlabel(self.val_ids)
            unique, counts = np.unique(labels, return_counts=True)
            for u in range(len(unique)):
                logger.info(f'Number of samples in class {unique[u]}: {counts[u]}')
                if return_descriptor:
                    df.loc[index[u], 'val'] = counts[u]

        count = len(self.test_ids) if self.test_ids is not None else 0
        logger.info(f'\nNumber of test samples: {count}')
        if self.test_ids is not None and len(self.test_ids) > 0:
            labels = self.getlabel(self.test_ids)
            unique, counts = np.unique(labels, return_counts=True)
            for u in range(len(unique)):
                logger.info(f'Number of samples in class {unique[u]}: {counts[u]}')
                if return_descriptor:
                    df.loc[index[u], 'test'] = counts[u]

        if self.train_ids is not None and self.test_ids is not None:
            assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
        if self.train_ids is not None and self.val_ids is not None:
            assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
        if self.val_ids is not None and self.test_ids is not None:
            assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0

        if return_descriptor:
            return df

    def save_split(self, filename):
        train_split = self.get_list(self.train_ids) if self.train_ids is not None else pd.Series(dtype=str)
        val_split = self.get_list(self.val_ids) if self.val_ids is not None else pd.Series(dtype=str)
        test_split = self.get_list(self.test_ids) if self.test_ids is not None else pd.Series(dtype=str)
        df_tr = pd.DataFrame({'train': train_split})
        df_v = pd.DataFrame({'val': val_split})
        df_t = pd.DataFrame({'test': test_split})
        df = pd.concat([df_tr, df_v, df_t], axis=1)
        df.to_csv(filename, index=False)  # Ensure not to save index column
        logger.info(f"Split saved to {filename}")

# class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
#     def __init__(self, data_dir, **kwargs):
#         super(Generic_MIL_Dataset, self).__init__(**kwargs)
#         self.data_dir = data_dir
#         self.use_h5 = False

#     def load_from_h5(self, toggle):
#         self.use_h5 = toggle

#     def __getitem__(self, idx):
#         slide_id = self.slide_data['slide_id'][idx]
#         label = self.slide_data['label'][idx]
#         if isinstance(self.data_dir, dict):
#             source = self.slide_data['source'][idx]
#             data_dir = self.data_dir[source]
#         else:
#             data_dir = self.data_dir

#         if not self.use_h5:
#             if data_dir:
#                 pattern = os.path.join(data_dir, f"{slide_id}*.pt")
#                 matched_files = glob.glob(pattern)

#                 if len(matched_files) == 0:
#                     logger.error(f"No .pt file found for slide_id: {slide_id}")
#                     raise FileNotFoundError(f"No .pt file found for slide_id: {slide_id}")
#                 elif len(matched_files) > 1:
#                     logger.warning(f"Multiple .pt files found for slide_id: {slide_id}, loading and concatenating them.")
#                     features_list = []
#                     for file in matched_files:
#                         try:
#                             features = torch.load(file, weights_only=True)
#                             assert features.shape[1] in [1024, 512], f"Feature dimension mismatch in {file}"
#                             features_list.append(features)
#                         except Exception as e:
#                             logger.error(f"Error loading file {file}: {e}")
#                             raise e
#                     try:
#                         features = torch.cat(features_list, dim=0)
#                     except Exception as e:
#                         logger.error(f"Error concatenating features for slide_id: {slide_id}: {e}")
#                         raise e
#                 else:
#                     full_path = matched_files[0]
#                     try:
#                         features = torch.load(full_path, weights_only=True)
#                     except Exception as e:
#                         logger.error(f"Error loading file {full_path}: {e}")
#                         raise e

#                 return {'features': features, 'labels': label}
#             else:
#                 return {'features': slide_id, 'labels': label}
#         else:
#             full_path = os.path.join(data_dir, 'h5_files', f"{slide_id}.h5")
#             try:
#                 with h5py.File(full_path, 'r') as hdf5_file:
#                     features = hdf5_file['features'][:]
#                     coords = hdf5_file['coords'][:]
#             except Exception as e:
#                 logger.error(f"Error loading h5 file {full_path}: {e}")
#                 raise e

#             features = torch.from_numpy(features)
#             return {'features': features, 'labels': label, 'coords': coords}

class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
    def __init__(self, data_dir, cache_in_ram=False, **kwargs):
        super(Generic_MIL_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.use_h5 = False
        self.cache_in_ram = cache_in_ram
        self.feature_cache = {}
        self.file_map = self._build_file_map()

    def load_from_h5(self, toggle):
        self.use_h5 = toggle

    def _build_file_map(self):
        if self.use_h5:
            return None
        if isinstance(self.data_dir, dict) or self.data_dir is None:
            return None

        file_map = {}
        unique_slide_ids = self.slide_data['slide_id'].astype(str).unique().tolist()

        for slide_id in unique_slide_ids:
            pattern = os.path.join(self.data_dir, f"{slide_id}*.pt")
            matched_files = sorted(glob.glob(pattern))
            if len(matched_files) > 0:
                file_map[slide_id] = matched_files

        logger.info(f"Indexed {len(file_map)} slide ids in {self.data_dir}")
        return file_map

    def _load_pt_features(self, slide_id):
        if slide_id in self.feature_cache:
            return self.feature_cache[slide_id]

        if self.file_map is not None:
            matched_files = self.file_map.get(slide_id, [])
        else:
            pattern = os.path.join(self.data_dir, f"{slide_id}*.pt")
            matched_files = sorted(glob.glob(pattern))

        if len(matched_files) == 0:
            logger.error(f"No .pt file found for slide_id: {slide_id}")
            raise FileNotFoundError(f"No .pt file found for slide_id: {slide_id}")

        elif len(matched_files) > 1:
            features_list = []
            for file in matched_files:
                features = torch.load(file, map_location='cpu', weights_only=True)
                features_list.append(features)
            features = torch.cat(features_list, dim=0)

        else:
            features = torch.load(matched_files[0], map_location='cpu', weights_only=True)

        if self.cache_in_ram:
            self.feature_cache[slide_id] = features

        return features

    def preload_features(self):
        if not self.cache_in_ram:
            return
        if self.file_map is None:
            return

        logger.info(f"Preloading {len(self.file_map)} slides into RAM ...")
        for slide_id in self.file_map.keys():
            _ = self._load_pt_features(slide_id)
        logger.info(f"RAM cache ready: {len(self.feature_cache)} slides")

    def __getitem__(self, idx):
        slide_id = str(self.slide_data['slide_id'][idx])
        label = self.slide_data['label'][idx]

        if isinstance(self.data_dir, dict):
            source = self.slide_data['source'][idx]
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir

        if not self.use_h5:
            if data_dir:
                features = self._load_pt_features(slide_id)
                return {'features': features, 'labels': label}
            else:
                return {'features': slide_id, 'labels': label}
        else:
            full_path = os.path.join(data_dir, 'h5_files', f"{slide_id}.h5")
            with h5py.File(full_path, 'r') as hdf5_file:
                features = hdf5_file['features'][:]
                coords = hdf5_file['coords'][:]

            features = torch.from_numpy(features)
            return {'features': features, 'labels': label, 'coords': coords}


# class Generic_Split(Dataset):
#     def __init__(self, slide_data, data_dir=None, num_classes=2):
#         self.use_h5 = False
#         self.slide_data = slide_data
#         self.data_dir = data_dir
#         self.num_classes = num_classes
#         self.slide_cls_ids = [[] for i in range(self.num_classes)]
#         for i in range(self.num_classes):
#             self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

#     def __len__(self):
#         return len(self.slide_data)

#     def __getitem__(self, idx):
#         slide_id = self.slide_data['slide_id'][idx]
#         label = self.slide_data['label'][idx]
#         if isinstance(self.data_dir, dict):
#             source = self.slide_data['source'][idx]
#             data_dir = self.data_dir[source]
#         else:
#             data_dir = self.data_dir

#         if not self.use_h5:
#             if data_dir:
#                 pattern = os.path.join(data_dir, f"{slide_id}*.pt")
#                 matched_files = glob.glob(pattern)

#                 if len(matched_files) == 0:
#                     logger.error(f"No .pt file found for slide_id: {slide_id}")
#                     raise FileNotFoundError(f"No .pt file found for slide_id: {slide_id}")
#                 elif len(matched_files) > 1:
#                     logger.warning(f"Multiple .pt files found for slide_id: {slide_id}, loading and concatenating them.")
#                     features_list = []
#                     for file in matched_files:
#                         try:
#                             features = torch.load(file, weights_only=True)
#                             assert features.shape[1] in [1024, 512], f"Feature dimension mismatch in {file}"
#                             features_list.append(features)
#                         except Exception as e:
#                             logger.error(f"Error loading file {file}: {e}")
#                             raise e
#                     try:
#                         features = torch.cat(features_list, dim=0)
#                     except Exception as e:
#                         logger.error(f"Error concatenating features for slide_id: {slide_id}: {e}")
#                         raise e
#                 else:
#                     full_path = matched_files[0]
#                     try:
#                         features = torch.load(full_path, weights_only=True)
#                     except Exception as e:
#                         logger.error(f"Error loading file {full_path}: {e}")
#                         raise e

#                 return {'features': features, 'labels': label}
#             else:
#                 return {'features': slide_id, 'labels': label}
#         else:
#             full_path = os.path.join(data_dir, 'h5_files', f"{slide_id}.h5")
#             try:
#                 with h5py.File(full_path, 'r') as hdf5_file:
#                     features = hdf5_file['features'][:]
#                     coords = hdf5_file['coords'][:]
#             except Exception as e:
#                 logger.error(f"Error loading h5 file {full_path}: {e}")
#                 raise e

#             features = torch.from_numpy(features)
#             return {'features': features, 'labels': label, 'coords': coords}

class Generic_Split(Dataset):
    def __init__(self, slide_data, data_dir=None, num_classes=2, cache_in_ram=False):
        self.use_h5 = False
        self.slide_data = slide_data
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.cache_in_ram = cache_in_ram
        self.feature_cache = {}

        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

        self.file_map = self._build_file_map()

    def _build_file_map(self):
        if self.use_h5:
            return None
        if isinstance(self.data_dir, dict) or self.data_dir is None:
            return None

        file_map = {}
        unique_slide_ids = self.slide_data['slide_id'].astype(str).unique().tolist()

        for slide_id in unique_slide_ids:
            pattern = os.path.join(self.data_dir, f"{slide_id}*.pt")
            matched_files = sorted(glob.glob(pattern))
            if len(matched_files) > 0:
                file_map[slide_id] = matched_files

        logger.info(f"Indexed {len(file_map)} split slide ids in {self.data_dir}")
        return file_map

    def _load_pt_features(self, slide_id):
        if slide_id in self.feature_cache:
            return self.feature_cache[slide_id]

        if self.file_map is not None:
            matched_files = self.file_map.get(slide_id, [])
        else:
            pattern = os.path.join(self.data_dir, f"{slide_id}*.pt")
            matched_files = sorted(glob.glob(pattern))

        if len(matched_files) == 0:
            logger.error(f"No .pt file found for slide_id: {slide_id}")
            raise FileNotFoundError(f"No .pt file found for slide_id: {slide_id}")

        elif len(matched_files) > 1:
            features_list = []
            for file in matched_files:
                features = torch.load(file, map_location='cpu', weights_only=True)
                features_list.append(features)
            features = torch.cat(features_list, dim=0)
        else:
            features = torch.load(matched_files[0], map_location='cpu', weights_only=True)

        if self.cache_in_ram:
            self.feature_cache[slide_id] = features

        return features

    def preload_features(self):
        if not self.cache_in_ram:
            return
        if self.file_map is None:
            return

        logger.info(f"Preloading {len(self.file_map)} split slides into RAM ...")
        for slide_id in self.file_map.keys():
            _ = self._load_pt_features(slide_id)
        logger.info(f"Split RAM cache ready: {len(self.feature_cache)} slides")

    def __len__(self):
        return len(self.slide_data)

    def __getitem__(self, idx):
        slide_id = str(self.slide_data['slide_id'][idx])
        label = self.slide_data['label'][idx]

        if isinstance(self.data_dir, dict):
            source = self.slide_data['source'][idx]
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir

        if not self.use_h5:
            if data_dir:
                features = self._load_pt_features(slide_id)
                return {'features': features, 'labels': label}
            else:
                return {'features': slide_id, 'labels': label}
        else:
            full_path = os.path.join(data_dir, 'h5_files', f"{slide_id}.h5")
            with h5py.File(full_path, 'r') as hdf5_file:
                features = hdf5_file['features'][:]
                coords = hdf5_file['coords'][:]

            features = torch.from_numpy(features)
            return {'features': features, 'labels': label, 'coords': coords}

