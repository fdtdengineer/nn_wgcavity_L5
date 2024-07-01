#%%
if True:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from torchvision.datasets import MNIST
    from torch.utils.data import DataLoader
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    from matplotlib import rc
    rc('text', usetex=False)

    fs = 18
    plt.rcParams['font.family']= 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams["font.size"] = fs 
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    import nntorch
    filepath_npy = "npy\\"
    filepath_output = "output\\"
    filepath_figure = "figure\\"

if __name__ == "__main__":
    model = nntorch.CNN()
    model.load_state_dict(torch.load(filepath_output + "cnn.pkl"))

    df_out = pd.read_csv(filepath_npy + "x_out.csv", header=None)
    x = df_out.values
    x = x.reshape(-1, 1, 13, 2)
    x = torch.from_numpy(x).float()

    prediction = model(x)
    prediction = prediction.detach().numpy()
    prediction = prediction.reshape(1, 1)

    print(prediction)
    print(10**prediction[0, 0])

    mat_x = x[0][0].detach().numpy()

    # mat_x の先頭に 0 を追加
    mat_x = np.insert(mat_x, 0, 0, axis=0)
    mat_x = mat_x.transpose(1,0).reshape(2,-1,7)
    grid_x, grid_y = np.meshgrid(np.arange(7), np.arange(2))


    # x を 2次元ベクトルとして表示

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.quiver( grid_x, grid_y, mat_x[:,0], mat_x[:,1], angles='xy', scale_units='xy', scale=1)
    plt.xlabel("Re[$x$]")
    plt.ylabel("Im[$x$]")
    
    plt.tight_layout()
    plt.show()
