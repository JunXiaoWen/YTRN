from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import pandas as pd

color_map = ['r', 'y', 'g', 'm', 'c', 'peru', 'pink', 'teal',  'darkred', 'gray', 'darkcyan',
              'darkorchid', 'lightblue', 'hotpink', 'violet', 'ivory', 'tomato', 'brown',
             'gold', 'navy', 'aqua', 'skyblue', 'seashell']
def plot_embedding(data, label, title):
    # x_min, x_max = np.min(data, 0), np.max(data, 0)
    # data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    for i in range(data.shape[0]):
        plt.plot(data[i, 0], data[i, 1], marker='o', markersize=2, color=color_map[label[i]])
    return fig


if __name__ == '__main__':
    all_data = np.load('data.npy', allow_pickle=True)
    all_label = np.load('label.npy', allow_pickle=True)
    data = []
    label = []
    select = [1, 10, 17, 28, 31, 43, 49, 56, 61, 62,  78, 79, 80, 82, 84]
    # select = [83, ]
    for index, l in enumerate(all_label):
        if l in select:
            label.append(l)
            data.append(all_data[index])
    data = np.asarray(data)
    tsne = TSNE(n_components=2, perplexity=30.0, init='pca', random_state=2)
    result = tsne.fit_transform(data)
    x_min, x_max = np.min(result, 0), np.max(result, 0)
    result = (result - x_min) / (x_max - x_min)


    

    # 初始化LabelEncoder对象
    label_encoder = LabelEncoder()

    # 调用fit_transform方法进行标签编码
    encode_label = label_encoder.fit_transform(label)
    # new_label2 = label_encoder.fit_transform(new_label)

    fig = plot_embedding(result, encode_label, '11')
    fig.show()

    data = {'x': [item[0] for item in result], 'y': [item[1] for item in result], 'label': [i for i in encode_label]}
    df = pd.DataFrame(data)
    df.to_excel('./all.xlsx')




