import csv
import torch
import torch.nn.utils.rnn as rnn_utils
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
Train_log_dir = '../Log'


def read_f_t_csv():
    with open('../Datasets/features.csv') as csvfile:
        x_reader = csv.reader(csvfile)
        x_tensor_rows_3D = []
        temp_tensor_row_2d = []
        for row in x_reader:
            temp_tensor_row = []
            for i in range(1, len(row)+1):
                temp_tensor_row.append(float(row[i-1]))
                if i % 53 == 0:
                    temp_tensor_row_2d.append(temp_tensor_row)
                    temp_tensor_row = []
            x_tensor_rows_3D.append(torch.tensor(temp_tensor_row_2d))
            temp_tensor_row_2d = []
    with open('../Datasets/tags.csv') as csvfile:
        y_reader = csv.reader(csvfile)
        y_tensor_rows = []
        for row in y_reader:
            temp_tensor_row = []
            for i in range(len(row)):
                temp_tensor_row.append(float(row[i]))
            y_tensor_rows.append(torch.tensor(temp_tensor_row))
    return x_tensor_rows_3D, y_tensor_rows


class MyData(data.Dataset):
    def __init__(self, data_seq, y):
        self.data_seq = data_seq
        self.y = y

    def __len__(self):
        return len(self.data_seq)

    def __getitem__(self, idx):
        tuple_ = (self.data_seq[idx], self.y[idx])
        return tuple_


def collate_fn(data_tuple):
    data_tuple.sort(key=lambda x: len(x[0]), reverse=True)
    data = [sq[0] for sq in data_tuple]
    label = [sq[1] for sq in data_tuple]
    data_length = [len(q) for q in data]
    label_length = [len(q) for q in label]
    data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0.0)
    label = rnn_utils.pad_sequence(label, batch_first=True, padding_value=0.0)
    return data, label.unsqueeze(-1), data_length, label_length


# Fully connected neural network with one hidden layer
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.rnn(x, h0)

        # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out)

        return out


def lstm_train_param(learning_rate=0.001, num_epochs=50, lstm_hidden_size=32, lstm_num_layers=2, train_batch_size=4):
    min_val_loss = 100

    x, y = read_f_t_csv()
    # print(x)
    # print(y)
    mydata = MyData(x, y)
    train_size = int(len(mydata) * 0.7)
    test_size = len(mydata) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(mydata, [train_size, test_size])

    model = RNN(input_size=53, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers).to(device)

    # Loss and optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=True, collate_fn=collate_fn)
    n_total_steps = len(train_loader)
    train_loss_list = []
    val_loss_list = []
    for epoch in range(num_epochs):
        epoch_loss_list = []
        for i, (x, y, x_len, y_len) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            # Forward pass
            outputs = model(x)

            # print('------------')
            # print(x.shape)
            # print(y.shape)
            # print(outputs.shape)
            # print('------------')
            loss = criterion(outputs, y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 1000 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}],'
                      f' Train Loss: {loss.item():.4f}')
                epoch_loss_list.append(loss.item())
        total = 0
        for loss in epoch_loss_list:
            total += loss
        train_loss_list.append(total/len(epoch_loss_list))

        total_val_loss = 0
        i = 0
        with torch.no_grad():
            for i, (val_x, val_y, val_x_len, val_y_len) in enumerate(val_loader):
                # print(val_y)
                val_x = val_x.to(device)
                val_y = val_y.to(device)
                # Forward pass
                val_outputs = model(val_x)
                # print(val_outputs)
                # print(val_y)
                val_loss = criterion(val_outputs, val_y)
                total_val_loss += val_loss.item()
                i += 1
        final_val_loss = total_val_loss / i
        val_loss_list.append(final_val_loss)
        if final_val_loss < min_val_loss:
            min_val_loss = final_val_loss
        print(f'Min Valid Loss: {final_val_loss:.4f}')
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
    for lr in [0.001, 0.005, 0.0005, 0.0001, 0.00001]:
        for hid_size in [32, 64, 128, 256, 512]:
            for layer in [1, 2, 3, 4]:
                for batch_size in [4, 8, 16, 32]:
                    print(f'lr: {lr}, hidden_size: {hid_size}, layer_num: {layer}, train_batch_size: {batch_size}')
                    cur_min_val_loss, cur_train_loss_list, cur_val_loss_list = \
                        lstm_train_param(learning_rate=lr, lstm_hidden_size=hid_size,
                                         lstm_num_layers=layer, train_batch_size=batch_size)
                    cur_filename = f'lr_{lr}_hidden_size_{hid_size}_layer_num_{layer}_' \
                                   f'train_batch_size_{batch_size}_minValLoss_{cur_min_val_loss}'
                    save_train_loss_plt(cur_min_val_loss, cur_train_loss_list, cur_val_loss_list, cur_filename)
                    f = open(Train_log_dir + '/model_min_val_loss.txt', 'a')
                    f.write(f'lr: {lr}, hidden_size: {hid_size}, layer_num: {layer}, train_batch_size: {batch_size}'
                            f' - min val loss: {cur_min_val_loss}\n')
                    f.close()
