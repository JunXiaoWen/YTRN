import numpy as np
import os
import json
import random
import yaml
import traceback
from paddle.io import Dataset

from ppocr.data.imaug import transform, create_operators

from ppocr.data.utils import (dict2corrds, corrds2dict, calu_path_signature, strokes_to_stroke_4, slantStroke,
                              disturbanceStroke, draw_path_signature, padding_sequence, normalize_lines, draw_text_line,
                              equidistance_sample, normalize_pen_speed, coordToOriginPoint, scale_points)





class YiTextDataSet(Dataset):
    def __init__(self, config, mode, logger, seed=None):
        super(YiTextDataSet, self).__init__()
        self.logger = logger
        self.mode = mode.lower()

        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']

        self.delimiter = dataset_config.get('delimiter', '\t')
        label_file_list = dataset_config.pop('label_file_list')
        data_source_num = len(label_file_list)
        ratio_list = dataset_config.get("ratio_list", 1.0)
        if isinstance(ratio_list, (float, int)):
            ratio_list = [float(ratio_list)] * int(data_source_num)

        assert len(
            ratio_list
        ) == data_source_num, "The length of ratio_list should be the same as the file_list."
        self.data_dir = dataset_config['data_dir']
        self.do_shuffle = loader_config['shuffle']
        self.seed = seed
        # logger.info("Initialize indexs of datasets:%s" % label_file_list)
        self.data_lines = self.get_image_info_list(label_file_list, ratio_list)
        self.data_idx_order_list = list(range(len(self.data_lines)))
        if self.mode == "train" and self.do_shuffle:
            self.shuffle_data_random()
        self.ops = create_operators(dataset_config['transforms'], global_config)
        self.ext_op_transform_idx = dataset_config.get("ext_op_transform_idx",
                                                       2)
        self.need_reset = True in [x < 1 for x in ratio_list]

    def get_image_info_list(self, file_list, ratio_list):
        if isinstance(file_list, str):
            file_list = [file_list]
        data_lines = []
        for idx, file in enumerate(file_list):
            with open(file, "rb") as f:
                lines = f.readlines()
                if self.mode == "train" or ratio_list[idx] < 1.0:
                    random.seed(self.seed)
                    lines = random.sample(lines,
                                          round(len(lines) * ratio_list[idx]))
                data_lines.extend(lines)
        return data_lines

    def shuffle_data_random(self):
        random.seed(self.seed)
        random.shuffle(self.data_lines)
        return

    def _try_parse_filename_list(self, file_name):
        # multiple images -> one gt label
        if len(file_name) > 0 and file_name[0] == "[":
            try:
                info = json.loads(file_name)
                file_name = random.choice(info)
            except:
                pass
        return file_name

    def get_ext_data(self):
        ext_data_num = 0
        for op in self.ops:
            if hasattr(op, 'ext_data_num'):
                ext_data_num = getattr(op, 'ext_data_num')
                break
        load_data_ops = self.ops[:self.ext_op_transform_idx]
        ext_data = []

        while len(ext_data) < ext_data_num:
            file_idx = self.data_idx_order_list[np.random.randint(self.__len__(
            ))]
            data_line = self.data_lines[file_idx]
            data_line = data_line.decode('utf-8')
            substr = data_line.strip("\n").split(self.delimiter)
            file_name = substr[0]
            file_name = self._try_parse_filename_list(file_name)
            label = substr[1]
            img_path = os.path.join(self.data_dir, file_name)
            data = {'img_path': img_path, 'label': label}
            if not os.path.exists(img_path):
                continue
            with open(data['img_path'], 'rb') as f:
                img = f.read()
                data['image'] = img
            data = transform(data, load_data_ops)

            if data is None:
                continue
            if 'polys' in data.keys():
                if data['polys'].shape[1] != 4:
                    continue
            ext_data.append(data)
        return ext_data

    def __getitem__(self, idx):
        file_idx = self.data_idx_order_list[idx]
        data_line = self.data_lines[file_idx]
        data_line = data_line.decode('utf-8')
        substr = data_line.strip("\n").split(self.delimiter)
        sample = json.loads(substr[0])
        strokes = dict2corrds(sample['Strokes'])

        # draw_text_line(strokes)

        if self.mode == 'train' and random.random() < 0.6:
            strokes = disturbanceStroke(strokes)
            strokes = slantStroke(strokes)
        #     draw_text_line(strokes)
        strokes = normalize_pen_speed(strokes, 2)
        # draw_text_line(strokes)
        norm_strokes = normalize_lines(strokes)
         # draw_text_line(norm_strokes)

        seq = strokes_to_stroke_4(norm_strokes)
        seq = padding_sequence(seq)
        dict_strokes = corrds2dict(strokes)
        path_sign = calu_path_signature(dict_strokes, (32, 256))
        # draw_path_signature(path_sign)
        # path_sign = np.asarray([0])


        label = sample['transcription']
        data = {'image': path_sign, 'label': label, 'seq': seq}
        outs = transform(data, self.ops)
        return outs

    def __len__(self):
        return len(self.data_idx_order_list)



def load_config(file_path):
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"
    config = yaml.load(open(file_path, 'rb'), Loader=yaml.Loader)
    return config



if __name__ == '__main__':
    # config = load_config(r'E:\PaddleOCR\myuse\test.yml')
    # dataset = YiTextDataSet(config, mode='Train', logger=None)
    # print(dataset.__getitem__(10))
    pass
