# coding=utf8

import json
import os


def prepare(f_path,label_map):
    texts = list()
    labels = list()
    raw_labels = list()
    with open(f_path) as in_:
        for line in in_:
            line = line.strip()
            if len(line) > 1:
                obj_ = json.loads(line)
                texts.append(obj_['texts'].strip())
                cur_labels = obj_['label']
                raw_labels.append(cur_labels)

    for tmp_labels in raw_labels:
        cur_labels = [label_map[tmp_label] for tmp_label in tmp_labels]
        labels.append(cur_labels)
    return texts, labels