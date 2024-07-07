import lmdb
import paddle
import os
import numpy as np

import paddle.nn as nn
from paddle.io import Dataset
import yaml

from ppocr.data.imaug import transform, create_operators

class YiDataSet(Dataset):
    def __init__(self, config, mode, logger, seed=None):
        super(YiDataSet, self).__init__()
        self.logger = logger
        self.mode = mode.lower()

        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']
        batch_size = loader_config['batch_size_per_card']
        data_dir = dataset_config['data_dir']
        self.do_shuffle = loader_config['shuffle']
        self.lmdb_sets = self.load_hierarchical_lmdb_dataset(data_dir)
        # logger.info("Initialize indexs of datasets:%s" % data_dir)
        self.data_idx_order_list = self.dataset_traversal()
        if self.do_shuffle:
            np.random.shuffle(self.data_idx_order_list)
        self.ops = create_operators(dataset_config['transforms'], global_config)
        self.ext_op_transform_idx = dataset_config.get("ext_op_transform_idx",
                                                       1)

        ratio_list = dataset_config.get("ratio_list", [1.0])
        self.need_reset = True in [x < 1 for x in ratio_list]


    def load_hierarchical_lmdb_dataset(self, data_dir):
        lmdb_sets = {}
        dataset_idx = 0
        for dirpath, dirnames, filenames in os.walk(data_dir + '/'):
            if not dirnames:
                env = lmdb.open(
                    dirpath,
                    max_readers=32,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False)
                txn = env.begin(write=False)
                num_samples = int(txn.get('num-samples'.encode()))
                lmdb_sets[dataset_idx] = {"dirpath":dirpath, "env":env, \
                    "txn" : txn, "num_samples":num_samples}
                dataset_idx += 1
        return lmdb_sets

    def dataset_traversal(self):
        lmdb_num = len(self.lmdb_sets)
        total_sample_num = 0
        for lno in range(lmdb_num):
            total_sample_num += self.lmdb_sets[lno]['num_samples']
        data_idx_order_list = np.zeros((total_sample_num, 2))
        beg_idx = 0
        for lno in range(lmdb_num):
            tmp_sample_num = self.lmdb_sets[lno]['num_samples']
            end_idx = beg_idx + tmp_sample_num
            data_idx_order_list[beg_idx:end_idx, 0] = lno
            data_idx_order_list[beg_idx:end_idx, 1] \
                = list(range(tmp_sample_num))
            data_idx_order_list[beg_idx:end_idx, 1] += 1
            beg_idx = beg_idx + tmp_sample_num
        return data_idx_order_list

    def get_lmdb_sample_info(self, txn, index):
        label_key = 'label-%09d'.encode() % index

        label = txn.get(label_key)
        if label is None:
            return None
        label = label.decode('utf-8')
        img_key = 'image-%09d'.encode() % index
        imgbuf = txn.get(img_key)
        seq_key = 'sequence-%09d'.encode() % index
        seq_buf = txn.get(seq_key)
        return imgbuf, label, seq_buf


    def __getitem__(self, idx):
        lmdb_idx, file_idx = self.data_idx_order_list[idx]
        lmdb_idx = int(lmdb_idx)
        file_idx = int(file_idx)
        sample_info = self.get_lmdb_sample_info(self.lmdb_sets[lmdb_idx]['txn'], file_idx)
        if sample_info is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        img_bin, label, seq_bin = sample_info
        seq_array = np.frombuffer(seq_bin, dtype=float)

        img_array = np.frombuffer(img_bin, dtype=float)
        img = img_array.reshape((7, 32, 256))

        seq = seq_array.reshape((400, 4))
        seq = seq.transpose(1, 0)
        data = {'image': img, 'label': label, 'seq': seq}
        outs = transform(data, self.ops)
        return outs
        pass


    def __len__(self):
        return self.data_idx_order_list.shape[0]


def load_config(file_path):
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"
    config = yaml.load(open(file_path, 'rb'), Loader=yaml.Loader)
    return config

if __name__ == '__main__':
    # config = load_config(r'E:\PaddleOCR\myuse\rec_mv3_none_none_ctc.yml')
    # # config = {'Global':None, 'Train': {
    # #     'dataset':{'data_dir': "E:\\yi_dataset\\yi_image_50000\\lmdb\\baseline\\train",
    # #                'transforms':None }, 'loader':{
    # #         'batch_size_per_card':4, 'shuffle':True
    # #     }
    # # }}
    #
    # dataset = YiDataSet(config, mode='Train', logger=None)
    # print(dataset.__getitem__(1))
    pass





