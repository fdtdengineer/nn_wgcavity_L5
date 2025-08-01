#%%
if True:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    filepath_npy = "npy\\"
    filepath_output = "output\\"
    filepath_figure = "figure\\"

# To represent the symmetry about x- and y-axes
class SymX(nn.Module):
    def __init__(self, list_idxzero_x=[8], list_idxzero_y=[0,1,2], xabsmax=20):
        super(SymX, self).__init__()
        self.list_idxzero_x = list_idxzero_x
        self.list_idxzero_y = list_idxzero_y
        self.xabsmax = xabsmax

    def forward(self, x):
        num_data = x.size(0)

        # should be zero according to the symmetry
        x = x.clone()
        x[:, 0, self.list_idxzero_x, 0] = 0
        x[:, 0, self.list_idxzero_y, 1] = 0
        #xabsmax
        x[x > self.xabsmax] = self.xabsmax
        x[x < -self.xabsmax] = -self.xabsmax

        xq = torch.cat([torch.zeros(num_data, 1, 3, 2), x], dim=2)
        # dim=2 の10行目と11行目の間に0を挿入（2列目調節用）
        xql = xq[:, :, :6, :]
        xqr = xq[:, :, 6:, :]
        xq = torch.cat([xql, torch.zeros(num_data, 1, 1, 2), xqr], dim=2)

        xq2 = xq.view(num_data, 3, -1, 2).permute(0, 3, 1, 2)
        x_half = torch.cat([xq2.flip(dims=[3])[:,:,:,:-1], xq2], dim=3)
        x_all = torch.cat([x_half.flip(dims=[2])[:,:,:-1,:], x_half], dim=2)
        return x_all

# MSE with L2 regularization
class MSEwithL2(nn.Module):
    def __init__(self, model, lambda_reg=1):
        super(MSEwithL2, self).__init__()
        self.lambda_reg = lambda_reg
        self.model = model

    def forward(self, outputs, target):
        mse_loss = nn.MSELoss()(outputs, target)

        regularization = torch.tensor(0.0)
        num_params = torch.tensor(0)
        for param in self.model.parameters():
            regularization += torch.sum(param**2)
            num_params += param.numel()

        loss = mse_loss + self.lambda_reg * regularization / num_params

        return loss

# CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        padding=0
        self.symx = SymX()
        self.conv1 = nn.Conv2d(2, 50, kernel_size=(3,3), padding=padding)
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(50*3*9, 200) ## (5-4)*(13-4)
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(200, 50)
        self.relu3 = nn.ReLU()
        self.fc_out = nn.Linear(50, 1)

    def forward(self, x):
        x = self.symx(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu3(x)
        x = self.fc_out(x)
        return x


def train(train_loader):
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss = running_loss / len(train_loader)
    return train_loss


def valid(test_loader):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            predicted = outputs#.max(1, keepdim=True)[1]

            # (outputs - labels) の標準偏差を計算したい
            #std = torch.std(outputs - labels)
            std = torch.std(outputs - labels).cpu().numpy()
            accuracy = (10**std - 10**(-std)) / 2

            labels = labels.view_as(predicted)
            correct += accuracy #predicted.eq(labels).sum().item()
            total += labels.size(0)
    val_loss = running_loss / len(test_loader)
    val_acc = correct / len(test_loader) # total
    return val_loss, val_acc



if __name__ == "__main__":
    # Config
    filepath_coordinates = 'input\\coordinates\\'
    filepath_q_fem = 'input\\q_fem\\'
    filename_coordinates = 'randn_mat.csv'
    filename_q_fem = 'q_fem.csv'
    filepath_npy = 'npy\\'

    # params
    num_params = 14
    num_dims = 2
    num_samples = 10000 #10000
    EPOCH = 300
    num_batch = 100
    num_test = 100
    seed = 12345678
    lr=0.01
    lambda_reg=2
    # Data Loadings
    df_q_fem = pd.read_csv(filepath_q_fem + filename_q_fem)[0:num_samples]
    npr_q_fem = df_q_fem.to_numpy()
    npr_q_fem = np.log10(npr_q_fem)

    # load_npy
    x = np.load(filepath_npy + 'dataset.npy')
    x = x[:num_samples]
    x = torch.tensor(x, dtype=torch.float32)
    t = torch.tensor(npr_q_fem, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(x, t)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [num_samples-num_test, num_test], generator=torch.Generator().manual_seed(seed))
    #train_dataset = torch.utils.data.Subset(dataset, range(num_samples-num_test))
    #test_dataset = torch.utils.data.Subset(dataset, range(num_samples-num_test, num_samples))

    # save test dataset
    np.save(filepath_npy+'test_data.npy', test_dataset[:][0])
    np.save(filepath_npy+'test_label.npy', test_dataset[:][1])

    # Preparing for training
    train_loader = DataLoader(dataset=train_dataset, batch_size=num_batch, shuffle=False)
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=num_batch, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN().to(device)
    #criterion = nn.MSELoss()
    criterion = MSEwithL2(model, lambda_reg=lambda_reg)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training
    loss_list = []
    val_loss_list = []
    val_acc_list = []
    for epoch in range(EPOCH):
        loss = train(train_loader)
        val_loss, val_acc = valid(test_loader)
        print('epoch %d, loss: %.4f val_loss: %.4f val_acc: %.4f' % (epoch, loss, val_loss, val_acc))
        loss_list.append(loss)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

    np.save(filepath_npy+'loss_list.npy', np.array(loss_list))
    np.save(filepath_npy+'val_loss_list.npy', np.array(val_loss_list))
    np.save(filepath_npy+'val_acc_list.npy', np.array(val_acc_list))
    torch.save(model.state_dict(), filepath_output+'cnn.pkl')


    #plot
    plt.plot(range(EPOCH), loss_list, 'r-', label='train_loss')
    plt.plot(range(EPOCH), val_loss_list, 'b-', label='test_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.savefig(filepath_figure + "loss.svg", transparent=True)

    plt.figure()
    plt.plot(range(EPOCH), val_acc_list, 'g-', label='val_acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.ylim(0,1)
    plt.savefig(filepath_figure + "accuracy.svg", transparent=True)
    plt.close()
    print('Accuracy',val_acc_list[-1]*100, '%')


    print("Done.")