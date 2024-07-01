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
    plt.rcParams["font.size"] = fs # 全体のフォントサイズが変更されます。
    plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
    plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')

    import nntorch
    filepath_npy = "npy\\"
    filepath_output = "output\\"
    filepath_figure = "figure\\"


if __name__ == "__main__":
    # Load trained model "cnn.pkl"
    model = nntorch.CNN()
    model.load_state_dict(torch.load(filepath_output + "cnn.pkl"))

    # Load test data
    test_data = np.load(filepath_npy + "test_data.npy")
    test_label = np.load(filepath_npy + "test_label.npy")

    # drop
    #idx = test_label.argmax()
    #test_data = np.delete(test_data, idx, axis=0)
    #test_label = np.delete(test_label, idx, axis=0)

    # Get prediction
    test_data = torch.from_numpy(test_data).float()
    test_label = torch.from_numpy(test_label).float()
    test_data = test_data.view(-1, 1, 13, 2)
    #test_data = test_data.view(-1, 2, 5, 13)
    test_label = test_label.view(-1, 1)
    prediction = model(test_data)
    prediction = prediction.detach().numpy()
    prediction = prediction.reshape(-1, 1)

    # linear regression without intercept
    from sklearn import linear_model
    reg = linear_model.LinearRegression(fit_intercept=False)
    reg.fit(test_label, prediction)
    test_fit = reg.predict(test_label)


    # log-log plot
    plt.figure(figsize=(4.4, 4))
    plt.scatter(test_label, prediction, color="black")
    plt.plot(test_label, test_fit, color="blue")
    coef = reg.coef_
    coef = coef.reshape(-1)
    coef = coef.tolist()
    coef = [round(x, 3) for x in coef]
    score = reg.score(test_label, prediction)
    score = round(score, 3)
    plt.text(0.05, 0.78, "$\mathrm{R}^2 = $" + str(score), transform=plt.gca().transAxes)
    plt.text(0.05, 0.9, "$Q_{\mathrm{NN}} = $" + str(coef[0]) + "$Q_{\mathrm{FEM}}$", transform=plt.gca().transAxes)
    
    plt.xlabel("$\log_{10}Q_{\mathrm{FEM}}$")
    plt.ylabel("$\log_{10}Q_{\mathrm{NN}}$")
    plt.savefig(filepath_figure + "prediction.svg", transparent=True)
    plt.tight_layout()
    plt.show()

    # linear-linear plot
    test_label_liniar = 10**test_label
    test_prediction_linear = 10**prediction
    test_fit_linear = 10**test_fit

    plt.figure(figsize=(4.4, 4))
    plt.scatter(test_label_liniar, test_prediction_linear, color="black")
    plt.plot(test_label_liniar, test_fit_linear, color="blue")
    
    reg_linear = linear_model.LinearRegression(fit_intercept=False)
    reg_linear.fit(test_label_liniar, test_prediction_linear)
    coef_linear = reg_linear.coef_
    coef_linear = coef_linear.reshape(-1)
    coef_linear = coef_linear.tolist()
    coef_linear = [round(x, 3) for x in coef_linear]
    score_linear = reg_linear.score(test_label_liniar, test_prediction_linear)
    score_linear = round(score_linear, 3)
    plt.text(0.05, 0.78, "$\mathrm{R}^2 = $" + str(score_linear), transform=plt.gca().transAxes)
    plt.text(0.05, 0.9, "$Q_{\mathrm{NN}} = $" + str(coef_linear[0]) + "$Q_{\mathrm{FEM}}$", transform=plt.gca().transAxes)

    plt.xlabel("$Q_{\mathrm{FEM}}$")
    plt.ylabel("$Q_{\mathrm{NN}}$")
    plt.savefig(filepath_figure + "prediction_linear.svg", transparent=True)
    plt.tight_layout()
    plt.show()

    print("Done.")