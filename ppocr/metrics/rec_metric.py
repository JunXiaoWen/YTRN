# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from rapidfuzz.distance import Levenshtein
from difflib import SequenceMatcher

import numpy as np
import string


class RecMetric(object):
    def __init__(self,
                 main_indicator='acc',
                 is_filter=False,
                 ignore_space=True,
                 **kwargs):
        self.main_indicator = main_indicator
        self.is_filter = is_filter
        self.ignore_space = ignore_space
        self.eps = 1e-5
        self.reset()

    def _normalize_text(self, text):
        text = ''.join(
            filter(lambda x: x in (string.digits + string.ascii_letters), text))
        return text.lower()

    def edit_distance(self, str1, str2):
        # 初始化矩阵
        dp = [[0] * (len(str2) + 1) for _ in range(len(str1) + 1)]

        # 填充第一行和第一列
        for i in range(len(str1) + 1):
            dp[i][0] = i
        for j in range(len(str2) + 1):
            dp[0][j] = j

        # 动态规划求解
        for i in range(1, len(str1) + 1):
            for j in range(1, len(str2) + 1):
                if str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1])

        # 计算插入、删除及替换次数
        i, j = len(str1), len(str2)
        insert_count, delete_count, replace_count = 0, 0, 0
        while i > 0 or j > 0:
            if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
                i -= 1
                delete_count += 1
            elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
                j -= 1
                insert_count += 1
            else:
                i -= 1
                j -= 1
                if str1[i] != str2[j]:
                    replace_count += 1
        return dp[-1][-1], insert_count, delete_count, replace_count

    def __call__(self, pred_label, *args, **kwargs):
        preds, labels = pred_label
        correct_num = 0
        all_num = 0
        norm_edit_dis = 0.0
        norm_edit_distance = 0.0
        norm_CR = 0.0
        norm_AR = 0.0
        for (pred, pred_conf), (target, _) in zip(preds, labels):
            if self.ignore_space:
                pred = pred.replace(" ", "")
                target = target.replace(" ", "")
            if self.is_filter:
                pred = self._normalize_text(pred)
                target = self._normalize_text(target)
            norm_edit_dis += Levenshtein.normalized_distance(pred, target)
            edit_distance, insert_count, delete_count, replace_count = self.edit_distance(pred, target)
            target_length = len(target)
            norm_edit_distance += edit_distance / target_length
            norm_CR += (len(target) - delete_count - replace_count) / target_length
            norm_AR += (len(target) - delete_count - replace_count - insert_count) / target_length
            if pred == target:
                correct_num += 1
            all_num += 1
        self.correct_num += correct_num
        self.all_num += all_num
        self.norm_edit_dis += norm_edit_dis
        self.norm_edit_distance += norm_edit_distance
        self.norm_CR += norm_CR
        self.norm_AR += norm_AR
        return {
            'acc': correct_num / (all_num + self.eps),
            'norm_edit_dis': 1 - norm_edit_dis / (all_num + self.eps),
            'AR': norm_AR / (all_num + self.eps),
            'CR': norm_CR / (all_num + self.eps),
            'NED': 1 - norm_edit_distance / (all_num + self.eps),
        }

    def get_metric(self):
        """
        return metrics {
                 'acc': 0,
                 'norm_edit_dis': 0,
            }
        """
        acc = 1.0 * self.correct_num / (self.all_num + self.eps)
        norm_edit_dis = 1 - self.norm_edit_dis / (self.all_num + self.eps)
        AR = 1.0 * self.norm_AR / (self.all_num + self.eps)
        CR = 1.0 * self.norm_CR / (self.all_num + self.eps)
        NED = 1 - self.norm_edit_distance / (self.all_num + self.eps)
        self.reset()
        return {'acc': acc, 'norm_edit_dis': norm_edit_dis, 'AR': AR, 'CR':CR, 'NED': NED
                }

    def reset(self):
        self.correct_num = 0
        self.all_num = 0
        self.norm_edit_dis = 0
        self.norm_edit_distance = 0
        self.norm_CR = 0
        self.norm_AR = 0


class CNTMetric(object):
    def __init__(self, main_indicator='acc', **kwargs):
        self.main_indicator = main_indicator
        self.eps = 1e-5
        self.reset()

    def __call__(self, pred_label, *args, **kwargs):
        preds, labels = pred_label
        correct_num = 0
        all_num = 0
        for pred, target in zip(preds, labels):
            if pred == target:
                correct_num += 1
            all_num += 1
        self.correct_num += correct_num
        self.all_num += all_num
        return {'acc': correct_num / (all_num + self.eps), }

    def get_metric(self):
        """
        return metrics {
                 'acc': 0,
            }
        """
        acc = 1.0 * self.correct_num / (self.all_num + self.eps)
        self.reset()
        return {'acc': acc}

    def reset(self):
        self.correct_num = 0
        self.all_num = 0


class CANMetric(object):
    def __init__(self, main_indicator='exp_rate', **kwargs):
        self.main_indicator = main_indicator
        self.word_right = []
        self.exp_right = []
        self.word_total_length = 0
        self.exp_total_num = 0
        self.word_rate = 0
        self.exp_rate = 0
        self.reset()
        self.epoch_reset()

    def __call__(self, preds, batch, **kwargs):
        for k, v in kwargs.items():
            epoch_reset = v
            if epoch_reset:
                self.epoch_reset()
        word_probs = preds
        word_label, word_label_mask = batch
        line_right = 0
        if word_probs is not None:
            word_pred = word_probs.argmax(2)
        word_pred = word_pred.cpu().detach().numpy()
        word_scores = [
            SequenceMatcher(
                None,
                s1[:int(np.sum(s3))],
                s2[:int(np.sum(s3))],
                autojunk=False).ratio() * (
                    len(s1[:int(np.sum(s3))]) + len(s2[:int(np.sum(s3))])) /
            len(s1[:int(np.sum(s3))]) / 2
            for s1, s2, s3 in zip(word_label, word_pred, word_label_mask)
        ]
        batch_size = len(word_scores)
        for i in range(batch_size):
            if word_scores[i] == 1:
                line_right += 1
        self.word_rate = np.mean(word_scores)  #float
        self.exp_rate = line_right / batch_size  #float
        exp_length, word_length = word_label.shape[:2]
        self.word_right.append(self.word_rate * word_length)
        self.exp_right.append(self.exp_rate * exp_length)
        self.word_total_length = self.word_total_length + word_length
        self.exp_total_num = self.exp_total_num + exp_length

    def get_metric(self):
        """
        return {
            'word_rate': 0,
            "exp_rate": 0,
        }
        """
        cur_word_rate = sum(self.word_right) / self.word_total_length
        cur_exp_rate = sum(self.exp_right) / self.exp_total_num
        self.reset()
        return {'word_rate': cur_word_rate, "exp_rate": cur_exp_rate}

    def reset(self):
        self.word_rate = 0
        self.exp_rate = 0

    def epoch_reset(self):
        self.word_right = []
        self.exp_right = []
        self.word_total_length = 0
        self.exp_total_num = 0

if __name__ == '__main__':
    rec = RecMetric()
    str1 = 'sp'
    str2 = 'shp'
    print(Levenshtein.normalized_distance(str1, str2))
    print(rec.edit_distance(str1, str2))