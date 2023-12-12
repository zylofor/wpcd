path = "../WorkPiecesDataTrans/combinResults/combinResults.json"

import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt


def count_categories(file_path):
    with open(file_path) as f:
        annotations = json.load(f)

    category_count = {}
    for annotation in annotations['annotations']:
        category_id = annotation['category_id']
        if category_id not in category_count:
            category_count[category_id] = 0
        category_count[category_id] += 1

    return category_count


def random_color():
    r = random.random()
    g = random.random()
    b = random.random()
    return (r, g, b)


def draw_hist_and_cdf(category_count):
    categories = []
    counts = []
    for k, v in category_count.items():
        categories.append(k)
        counts.append(v)

  
    categories = [categories[i] for i in np.argsort(counts)]
    counts = [counts[i] for i in np.argsort(counts)]

  
    fig, ax1 = plt.subplots()
    ax1.bar(range(len(counts)), counts, color=[random_color() for _ in range(len(counts))], width=2.0, align = 'edge')
    ax1.set_xlim(-1,353)
    ax1.set_xlabel('Workpiece Categories')
    ax1.set_ylabel('Numbers')
    ax1.set_xticks(range(0, len(categories), 10))
    ax1.set_xticklabels([categories[i] for i in range(0, len(categories), 10)], rotation=90)

    
    cum_counts = np.cumsum(counts)
    cum_counts_norm = cum_counts / cum_counts[-1]
    ax2 = ax1.twinx()
    ax2.plot(range(len(cum_counts_norm)), cum_counts_norm, color='r', linestyle='--', linewidth=2)
    ax2.set_ylabel('Cumulative Percentage')
    ax2.set_ylim(0, 1)

    # plt.title('Category Counts and Cumulative Distribution Function')
    plt.tight_layout()

    plt.savefig('category_counts_and_cdf.pdf')

# 测试
category_count = count_categories(path)
draw_hist_and_cdf(category_count)
