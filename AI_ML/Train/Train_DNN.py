import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import torch.nn as nn
import matplotlib.pyplot as plt


# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device:{device}')
Train_log_dir = '../Log'


class NoseqDataset(Dataset):

    def __init__(self):
        # data loading
        x_np = np.loadtxt('../Datasets/features.csv',
                           delimiter=",", dtype=np.float32, skiprows=0)
        y_np = np.loadtxt('../Datasets/tags.csv',
                           delimiter=",", dtype=np.float32, skiprows=0)
        self.x = torch.from_numpy(x_np[:, ])
        self.y = torch.from_numpy(y_np[:, ])
        if x_np.shape[0] != y_np.shape[0]:
            print('error')
        self.n_samples = x_np.shape[0]

    def __getitem__(self, index):
        # dataset[index]
        return self.x[index], self.y[index]

    def __len__(self):
        # len(dataset)
        return self.n_samples


class DNN(nn.Module):
    def __init__(self, input_layer, _num_classes):
        super(DNN, self).__init__()
        self.l1 = nn.Linear(input_layer, 1024)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(1024, 512)
        self.l3 = nn.Linear(512, 256)
        self.l4 = nn.Linear(256, _num_classes)
        # design DNN here

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l4(out)
        return out


def train_DNN_3_layer(train_batch_size=8, learning_rate=0.005):
    # train_batch_size = 8
    input_size = 53
    # hidden_size = 1024
    num_classes = 1
    # learning_rate = 0.005
    num_epochs = 50

    dataset = NoseqDataset()

    first_data = dataset[0]

    train_size = int(len(dataset) * 0.7)
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=True)

    total_samples = len(dataset)
    n_iterations = math.ceil(total_samples/4)
    print(total_samples, n_iterations)


    model = DNN(input_size, num_classes).to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    val_loss_list = []
    min_val_loss = 100
    train_loss_list = []
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        epoch_loss_list = []
        for i, (features, tags) in enumerate(train_loader):
            features = features.reshape(-1, input_size).to(device)  # if gpu is working, push to it
            tags = tags.to(device)

            # forward
            outputs = model(features)
            loss = criterion(outputs, tags)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_total_steps}, loss={loss:.5f}')
        total = 0
        for loss in epoch_loss_list:
            total += loss
        if len(epoch_loss_list) != 0:
            train_loss_list.append(total / len(epoch_loss_list))

        total_val_loss = 0
        i = 0
        with torch.no_grad():
            for i, (features, tags) in enumerate(val_loader):
                # print(val_y)
                features = features.to(device)
                tags = tags.to(device)
                # Forward pass
                val_outputs = model(features)
                # print(val_outputs)
                # print(val_y)
                val_loss = criterion(val_outputs, tags)
                total_val_loss += val_loss.item()
                i += 1
        final_val_loss = total_val_loss / i
        val_loss_list.append(final_val_loss)
        if final_val_loss < min_val_loss:
            min_val_loss = final_val_loss
        print(f'Final Valid Loss: {final_val_loss:.4f}')
    print(f'Min Valid Loss: {min_val_loss:.4f}')
    return min_val_loss, train_loss_list, val_loss_list


def save_train_loss_plt(min_val_loss, train_loss_list, val_loss_list, filename):
    plt.title('Loss Chart, min val loss:' + str(min_val_loss))

    plt.plot(train_loss_list, color='skyblue', label='train')
    plt.plot(val_loss_list, color='yellow', label='valid')
    plt.legend()

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(f'{Train_log_dir}/{filename}.png')
    plt.close()


if __name__ == '__main__':
    for lr in [0.001, 0.005, 0.01, 0.05, 0.1]:
        for layer in [3]:
            for batch_size in [4, 8, 16, 32]:
                print(f'lr: {lr}, layer_num: {layer}, train_batch_size: {batch_size}')
                cur_min_val_loss, cur_train_loss_list, cur_val_loss_list = \
                    train_DNN_3_layer(train_batch_size=batch_size, learning_rate=lr)
                cur_filename = f'lr_{lr}_layer_num_{layer}_' \
                               f'train_batch_size_{batch_size}_minValLoss_{cur_min_val_loss}'
                save_train_loss_plt(cur_min_val_loss, cur_train_loss_list, cur_val_loss_list, cur_filename)
                f = open(Train_log_dir + '/model_min_val_loss.txt', 'a')
                f.write(f'lr: {lr}, layer_num: {layer}, train_batch_size: {batch_size}'
                        f' - min val loss: {cur_min_val_loss}\n')
                f.close()
