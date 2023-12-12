path = "/home/lab315/jzy/PaperLearning/MyDatasets/zylofor/SteelDataset/WorkPiecesDataTrans/combinResults/combinResults.json"

import os
import json
import numpy as np
import matplotlib.pyplot as plt


def count_width_height(annotations):
    width_height_list = []
    for annotation in annotations['annotations']:
        bbox = annotation['bbox']
        width = bbox[2]
        height = bbox[3]
        width_height_list.append((width, height))

    return width_height_list


def classify_objects(width_height_list):
    small_objs = []
    medium_objs = []
    large_objs = []

    for width, height in width_height_list:
        if width * height <= 32 * 32:
            small_objs.append((width, height))
        elif width * height <= 96 * 96:
            medium_objs.append((width, height))
        else:
            large_objs.append((width, height))

    return small_objs, medium_objs, large_objs


def draw_width_height(width_height_list):
    x, y = zip(*width_height_list)


    fig, ax = plt.subplots()

    small_objs, medium_objs, large_objs = classify_objects(width_height_list)

    ax.scatter(*zip(*small_objs), s=10,alpha=0.5,marker='v', color='r',  label='Small Objects (area <= 32*32)')
    ax.scatter(*zip(*medium_objs), s=10,alpha=0.5,marker='s', color='g', label='Medium Objects (32*32 < area <= 96*96)')
    ax.scatter(*zip(*large_objs), s=10,alpha=0.5,marker='o', color='b',  label='Large Objects (area > 96*96)')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')

    plt.legend(loc='upper right')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('width_height.jpg', dpi=2048, bbox_inches='tight')

    print('Number of small objects:', len(small_objs))
    print('Number of medium objects:', len(medium_objs))
    print('Number of large objects:', len(large_objs))

def count_and_draw_width_height(annotations_file_path):
    with open(annotations_file_path) as f:
        annotations = json.load(f)

    width_height_list = count_width_height(annotations)
    draw_width_height(width_height_list)

# 测试
count_and_draw_width_height(path)
