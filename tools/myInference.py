from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import os
import sys
import json
from ppma import cam
import matplotlib.pyplot as plt
import copy
from ppocr.data.utils import draw_path_signature, draw_xys_seq_sample, normalize_pen_speed, normalize_lines, \
    corrds2dict, calu_path_signature, strokes_to_stroke_4

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import paddle
from rapidfuzz.distance import Levenshtein

from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import get_image_file_list
import tools.program as program

def padding_sequence(sequence: np.ndarray, padding_len=512):
    # sequence = sequence.tolist()
    if len(sequence) <= padding_len:
        padding_strokes = sequence
        for i in range(len(sequence), padding_len):
            padding_strokes.append([0, 0, 1, 1])
    else:
        padding_strokes = sequence[0:padding_len]
    assert len(padding_strokes) == padding_len
    return np.array(padding_strokes)



def main():
    global_config = config['Global']

    # build post process
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)

    # build model
    if hasattr(post_process_class, 'character'):
        char_num = len(getattr(post_process_class, 'character'))
        if config['Architecture']["algorithm"] in ["Distillation",
                                                   ]:  # distillation model
            for key in config['Architecture']["Models"]:
                if config['Architecture']['Models'][key]['Head'][
                        'name'] == 'MultiHead':  # for multi head
                    out_channels_list = {}
                    if config['PostProcess'][
                            'name'] == 'DistillationSARLabelDecode':
                        char_num = char_num - 2
                    out_channels_list['CTCLabelDecode'] = char_num
                    out_channels_list['SARLabelDecode'] = char_num + 2
                    config['Architecture']['Models'][key]['Head'][
                        'out_channels_list'] = out_channels_list
                else:
                    config['Architecture']["Models"][key]["Head"][
                        'out_channels'] = char_num
        elif config['Architecture']['Head'][
                'name'] == 'MultiHead':  # for multi head loss
            out_channels_list = {}
            if config['PostProcess']['name'] == 'SARLabelDecode':
                char_num = char_num - 2
            out_channels_list['CTCLabelDecode'] = char_num
            out_channels_list['SARLabelDecode'] = char_num + 2
            config['Architecture']['Head'][
                'out_channels_list'] = out_channels_list
        else:  # base rec model
            config['Architecture']["Head"]['out_channels'] = char_num

    model = build_model(config['Architecture'])

    load_model(config, model)
    model.eval()

    valid_label_path = r"E:\yi_dataset\valid\labels.txt"
    norm_edit_dis = 0.0
    count = 0
    file_name_list = ['022e2ea2-1a8c-4979-8847-a09e6f01aebe', ' 0bddc836-e897-4f8c-9e58-3328e2415151', '0cf5ea45-beb2-413f-ad89-867aaabf88fa',
                      ' 0e09312e-fbfb-42b1-b195-5292733f273f', '128f929d-ef5b-49ee-9e6d-d416e76ed599', '17647d05-1f59-44c9-aca7-bcc4f70ddb7d',
                      '24a991e4-03bb-4d68-a249-b6aa5b749755', '35fa0476-f643-4ce1-859e-948541dc4edf', '3990f55d-aa97-44fd-b5cf-be8362db85ad',
                      '41aa82d4-0061-40fc-ab88-ce0ff399c7ca', '42d6466f-7fd6-403f-a69e-123b05b8c549', '4df7c8d4-6913-4ace-9f11-76563d9f6104',
                      '4fffd034-9129-4843-ab9b-366880691bcf', '5486dfcf-3f8e-4f21-90c3-42d476b9eb58', '5db84423-31e0-435e-ae1a-cf6db58c6e68',
                      '5f864893-03cc-4e07-aeda-e66f22c4c0b9', '671ce873-80b2-4eda-aefb-f4c4657bdfbe', '6c86ee19-3eb8-40b8-bd51-bc303baddbf2',
                      '80295acd-1928-4297-8908-51a0e0955dd5', '82ee689b-1740-49b7-9cf1-04b372457a99', '855ae628-dea2-44e4-a896-3b44f8f2887e',
                      '8ed0fec7-57b7-44f4-aa9a-103be4f30996', '92d979b7-4055-4b8d-88f1-9cee982e0bab', '95492fcb-73b8-4a5f-918a-0e7ca12fde94',
                      '97ed06a1-056d-40f3-aec6-f83f17a1482d', '9de0e77f-dd38-4f9a-8e10-2e508b66b0ff', 'a3a4a9a9-8902-45b5-881d-fee38e479c74',
                      'a9dff7ac-4da7-415b-9590-a0dd90d155be', 'ac585add-1bb6-40b6-91f1-8f3a90b59fe1', 'ada98bae-8928-4d31-82d5-c08f5dba1720',
                      'be6cd66b-5cc7-4a13-b2e7-1ae972a84d5d', 'c0ca527b-72c8-48bd-9f45-8367f06a32ca', 'd64d74ea-aaef-410f-9733-5a427b397360',
                      'd769a038-cd06-4a9f-9569-fd4ddb6d5a16', 'dbf66c1a-1775-4857-8e9a-640ed909f2bb', 'dc5afa96-2de4-4c78-be26-0a737be8b298',
                      'f3f1dcc2-f30b-498a-a4dd-78d7a21d0536']
    selected = ['f3f1dcc2-f30b-498a-a4dd-78d7a21d0536', 'd769a038-cd06-4a9f-9569-fd4ddb6d5a16', 'c0ca527b-72c8-48bd-9f45-8367f06a32ca',
                'be6cd66b-5cc7-4a13-b2e7-1ae972a84d5d', '80295acd-1928-4297-8908-51a0e0955dd5',
                '5db84423-31e0-435e-ae1a-cf6db58c6e68', '4df7c8d4-6913-4ace-9f11-76563d9f6104','3990f55d-aa97-44fd-b5cf-be8362db85ad',
                '24a991e4-03bb-4d68-a249-b6aa5b749755', ]


    seq_path = r'E:\yi_dataset\valid\all_txt'
    for file_name in os.listdir(seq_path):
        f = open(os.path.join(seq_path, file_name), 'r', encoding='utf-8')
        sample = json.loads(f.readline().strip('\n'))
        file_name = file_name.strip('.txt')

        if file_name not in selected:
            continue

        strokes = sample['strokes']
        label = sample['label']

        strokes = normalize_pen_speed(strokes, 2)
        # draw_text_line(strokes)
        norm_strokes = normalize_lines(strokes)
        # draw_text_line(norm_strokes)

        seq = strokes_to_stroke_4(norm_strokes)
        seq = padding_sequence(seq)
        dict_strokes = corrds2dict(strokes)
        path_sign = calu_path_signature(dict_strokes, (32, 256))

        images = paddle.to_tensor(path_sign)
        # draw_xys_seq_sample(seq)
        seq = paddle.to_tensor(seq)  # 400, 4

        images = paddle.unsqueeze(images, 0)
        seq = paddle.unsqueeze(seq, 0)
        # seq = paddle.transpose(seq, [0, 2, 1])
        pred = model([images, seq])
        post_result = post_process_class(pred)
        dis = (1 - Levenshtein.normalized_distance(post_result[0][0], label))
        if dis < 1:
            print(post_result, label, dis, file_name)
            draw_xys_seq_sample(seq.squeeze(0).numpy())
            count += 1
    print(norm_edit_dis / count)

    # with open(valid_label_path, 'r', encoding='utf-8') as f:
    #     for line in f.readlines():
    #         file_name = line.strip('\n').split('\t')[0]
    #         if file_name not in selected:
    #             continue
    #         label = line.strip('\n').split('\t')[1]
    #         image_path = "E:\\yi_dataset\\valid\\image\\" + file_name + '.npy'
    #         seq_path = "E:\\yi_dataset\\valid\\seq\\" + file_name + '.npy'
    #         images = np.load(image_path, allow_pickle=True)
    #         seq = np.load(seq_path, allow_pickle=True)
    #         seq = padding_sequence(seq)
    #         # draw_path_signature(images)
    #
    #         # draw_path_signature(images)
    #         images = paddle.to_tensor(images)
    #         # draw_xys_seq_sample(seq)
    #         seq = paddle.to_tensor(seq)  # 400, 4
    #
    #
    #
    #         images = paddle.unsqueeze(images, 0)
    #         seq = paddle.unsqueeze(seq, 0)
    #         # seq = paddle.transpose(seq, [0, 2, 1])
    #         pred = model([images, seq])
    #         post_result = post_process_class(pred)
    #         dis = (1 - Levenshtein.normalized_distance(post_result[0][0], label))
    #         if dis < 1:
    #             print(post_result, label, dis, file_name)
    #             draw_xys_seq_sample(seq.squeeze(0).numpy())
    #             count += 1
    #     print(norm_edit_dis / count)




if __name__ == '__main__':
    config, device, logger, vdl_writer = program.preprocess()
    main()
