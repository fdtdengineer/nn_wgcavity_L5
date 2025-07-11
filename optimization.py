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
    # Load trained model "cnn.pkl"
    model = nntorch.CNN()
    model.load_state_dict(torch.load(filepath_output + "cnn.pkl"))

    t_target = 5 # target: Q=1e5

    # Load test data
    test_data = np.load(filepath_npy + "test_data.npy")
    test_label = np.load(filepath_npy + "test_label.npy")
    x0 = np.zeros(test_data[0].shape)
    x0 = x0.reshape(-1)
    x = torch.from_numpy(x0).float()
    x = x.view(-1, 1, 14, 2)
    x.requires_grad = True
    t = torch.from_numpy(np.array(t_target)).float()
    t = t.view(1, 1)  

    optimizer = torch.optim.SGD([x], lr=2)
    #criterion = torch.nn.MSELoss()
    criterion = nntorch.MSEwithL2(model, lambda_reg=1)
    # Optimization
    num_iter = int(5e3)
    loss_history = np.zeros(num_iter)

    for i in range(num_iter):
        optimizer.zero_grad()
        prediction = model(x)
        loss = criterion(prediction, t)
        loss.backward()
        optimizer.step()
        loss_history[i] = loss.item()

    np.save(filepath_npy + "loss_history.npy", loss_history)
    x_out = x.detach().numpy()
    print(x_out.flatten())
    # Plot loss history
    plt.figure(figsize=(5,4))
    plt.plot(loss_history)
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(filepath_figure + "loss_history_opt.svg", transparent=True)
    plt.show()

    # Prediction
    x = x.reshape(-1, 1, 14, 2)
    prediction = model(x)
    prediction = prediction.detach().numpy()
    prediction = prediction.reshape(1, 1)

    print("x = ", x)
    print("prediction = ", prediction)

    # to_save_x
    x_out = x_out.reshape(14,2)

    df_x_out = pd.DataFrame(x_out)
    df_x_out.to_csv(filepath_npy + "x_out1.csv", index=False, header=False)

    print("end")
# %%
