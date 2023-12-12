import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker


methods = ['KMeans++', 'GMM']
attributes = ['PCA', 'PCA_pre', 'UMAP', 'UMAP_pre', 'TSNE', 'TSNE_pre']


colors = ['#1f77b4', '#f0e68c', '#ff4f0e', '#f0e68c','#2ca02c','#f0e68c']


accuracies = np.array([[81.14, 42.78, 83.21, 83.21, 87.15, 81.79],
                       [78.16, 57.24, 83.85, 83.85, 83.32, 83.32],])


fig, ax = plt.subplots()


bar_width = 0.160
for i in range(len(attributes)):
    x = np.arange(len(methods)) + i*bar_width
    ax.bar(x, accuracies[:, i], bar_width, color=colors[i], label=attributes[i])
    for j, acc in enumerate(accuracies[:,i]):
        ax.text(x[j], acc+0.02, '{:.2f}'.format(acc), ha='center', va='bottom', fontsize=10)


ax.set_xticks(np.arange(len(methods)) + 2*bar_width)
ax.set_xticklabels(methods)
ax.set_xlabel('Clustering Methods',fontsize=15)
ax.set_ylabel('Clustering Accuracy(%)',fontsize=15)
ax.set_title('Resize+LBP')


ax.set_yticks(np.arange(0,101,10))

fig.tight_layout()
ax.set_xlim(left=-0.55*bar_width, right=(len(methods)+1.5)*bar_width*3.37)
plt.savefig("output_cluster_results_2d.jpg",dpi=600)


plt.show()
