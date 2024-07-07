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

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import paddle

from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import get_image_file_list
import tools.program as program


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


    file_path = "E:\\yi_dataset\\valid\\image\\"
    seq_path = "E:\\yi_dataset\\valid\\seq\\"
    for file_name in os.listdir(file_path)[0:20]:
        image_path = os.path.join(file_path, file_name)
        seq_path = os.path.join(seq_path, file_name)
        target_layer = model.backbone.conv[-2]
        cam_extractor = cam.GradCAMPlusPlus(model, target_layer)
        activation_map = cam_extractor(image_path, seq_path, label=None)
        img = np.load(image_path, allow_pickle=True)
        image = copy.deepcopy(img[0])
        map = np.vstack((activation_map, image))
        plt.imshow(map)
        plt.axis('off')
        plt.show()









if __name__ == '__main__':
    config, device, logger, vdl_writer = program.preprocess()
    main()
