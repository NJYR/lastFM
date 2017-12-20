import gc
import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py

def heatmapplot(train,rowName='userID',colName='artistID',valueName='weight'):
    pt = train.pivot_table(index=rowName, columns=colName, values=valueName)
    pt = pt.fillna(0)
    f, ax = plt.subplots(figsize=(10, 4))
    #cmap = sns.cubehelix_palette(start=1, rot=3, gamma=0.8, as_cmap=True)
    sns.heatmap(pt[:100,:1000], cmap='rainbow', linewidths=0.05, ax=ax)
    ax.set_title('Amounts per kind and region')
    ax.set_xlabel('region')
    ax.set_ylabel('kind')
    f.savefig('sns_heatmap_normal.jpg', bbox_inches='tight')


if __name__ == "__main__":
    filePlace = "C:\\Users\\22560\\PycharmProjects\\lastFM\\networkData\\"

    gc.collect()
    # read train data
    os.chdir("C:\\Users\\22560\\PycharmProjects\\lastFM\\hetrec2011-lastfm-2k")
    train = pd.read_csv("user_artists.csv")
    heatmapplot(train,'userID','artistID','weight')

    h5f = h5py.File("C:\\Users\\22560\\PycharmProjects\\lastFM\\networkData\\yPrepare.h5", 'r')
    y = h5f['/yData/y'].value
    plt.imshow(np.log(y[:1000,:1000]), aspect='auto')
    plt.show()

    vmin = 0.0001
    plt.matshow(pt, aspect='auto', vmin=vmin)

    import plotly
    plotly.tools.set_credentials_file(username='anxu5829', api_key='JcQfwVNcBp9emYn5ifwl')

    import plotly.plotly as py
    import plotly.graph_objs as go
    y[y>1] = np.log(y[y>1])


    trace = go.Heatmap(z=y[:1000,:1000])
    data = [trace]
    py.iplot(data, filename='heatmapFordenseY1000')








