#Rated_Capacity = 1.1
fig, ax = plt.subplots(1, figsize=(12, 8))
color_list = ['b:', 'g--', 'r-.', 'c.']
for name,color in zip(Battary_list, color_list):
    df_result = Battery[name]
    ax.plot(df_result['cycle'], df_result['capacity'], color, label='Battery_'+name)
ax.set(xlabel='Discharge cycles', ylabel='Capacity (Ah)', 
       title='Capacity degradation at ambient temperature of 1°C')
plt.legend()


def build_sequences(text, window_size):
    #text:list of capacity
    x, y = [],[]
    for i in range(len(text) - window_size):
        sequence = text[i:i+window_size]
        target = text[i+1:i+1+window_size]

        x.append(sequence)
        y.append(target)

    return np.array(x), np.array(y)

# 留一评估：一组数据为测试集，其他所有数据全部拿来训练
def get_train_test(data_dict, name, window_size=8, train_ratio=0.):
    data_sequence=data_dict[name][1]
    train_data, test_data = data_sequence[:window_size+1], data_sequence[window_size+1:]
    train_x, train_y = build_sequences(text=train_data, window_size=window_size)
    for k, v in data_dict.items():
        if k != name:
            data_x, data_y = build_sequences(text=v[1], window_size=window_size)
            train_x, train_y = np.r_[train_x, data_x], np.r_[train_y, data_y]

    return train_x, train_y, list(train_data), list(test_data)



class Net(nn.Module):
    def __init__(self, feature_size=8, hidden_size=[16, 8]):
        super(Net, self).__init__()
        self.feature_size, self.hidden_size = feature_size, hidden_size
        self.layer0 = nn.Linear(self.feature_size, self.hidden_size[0])
        self.layers = [nn.Sequential(
            nn.Linear(self.hidden_size[i], self.hidden_size[i+1]), nn.ReLU()) 
                       for i in range(len(self.hidden_size) - 1)]
        self.linear = nn.Linear(self.hidden_size[-1], 1)

    def forward(self, x):
        out = self.layer0(x)
        for layer in self.layers:
            out = layer(out)
        out = self.linear(out) 
        return out

def tain(LR=0.01, feature_size=8, hidden_size=[16,8], weight_decay=0.0, 
         window_size=8, EPOCH=1000, seed=0):
    mae_list, rmse_list, re_list = [], [], []
    result_list = []
    for i in range(4):
        name = Battary_list[i]
        train_x, train_y, train_data, test_data = get_train_test(Battery, name, window_size)
        train_size = len(train_x)
        print('sample size: {}'.format(train_size))

        setup_seed(seed)
        model = Net(feature_size=feature_size, hidden_size=hidden_size)
        if torch.cuda.is_available():
            model = model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        test_x = train_data.copy()
        loss_list, y_ = [0], []
        for epoch in range(EPOCH):
            X = np.reshape(train_x/Rated_Capacity, (-1, feature_size)).astype(np.float32)
            y = np.reshape(train_y[:,-1]/Rated_Capacity,(-1,1)).astype(np.float32)

            X, y = torch.from_numpy(X), torch.from_numpy(y)
            output= model(X)
            loss = criterion(output, y)
            optimizer.zero_grad()              # clear gradients for this training step
            loss.backward()                    # backpropagation, compute gradients
            optimizer.step()                   # apply gradients

            if (epoch + 1)%100 == 0:
                test_x = train_data.copy() #每100次重新预测一次
                point_list = []
                while (len(test_x) - len(train_data)) < len(test_data):
                    x = np.reshape(np.array(test_x[-feature_size:])/Rated_Capacity, 
                                   (-1, feature_size)).astype(np.float32)
                    x = torch.from_numpy(x)
                    pred = model(x) 
                    next_point = pred.data.numpy()[0,0] * Rated_Capacity
                    test_x.append(next_point)#测试值加入原来序列用来继续预测下一个点
                    point_list.append(next_point)#保存输出序列最后一个点的预测值
                y_.append(point_list)#保存本次预测所有的预测值
                loss_list.append(loss)
                mae, rmse = evaluation(y_test=test_data, y_predict=y_[-1])
                re = relative_error(
                    y_test=test_data, y_predict=y_[-1], threshold=Rated_Capacity*0.7)
                print('epoch:{:<2d} | loss:{:<6.4f} | MAE:{:<6.4f} | RMSE:{:<6.4f} | \
                RE:{:<6.4f}'.format(epoch, loss, mae, rmse, re))
            if (len(loss_list) > 1) and (abs(loss_list[-2] - loss_list[-1]) < 1e-5):
                break

        mae, rmse = evaluation(y_test=test_data, y_predict=y_[-1])
        re = relative_error(
            y_test=test_data, y_predict=y_[-1], threshold=Rated_Capacity*0.7)
        mae_list.append(mae)
        rmse_list.append(rmse)
        re_list.append(re)
        result_list.append(y_[-1])
    return re_list, mae_list, rmse_list, result_list


window_size = 8
EPOCH = 1000
LR = 0.01    # learning rate
feature_size = window_size
hidden_size = [32,16]
weight_decay = 0.0
Rated_Capacity = 1.1

MAE, RMSE, RE = [], [], []
for seed in range(10):
    re_list, mae_list, rmse_list, _ = tain(LR, feature_size, hidden_size, weight_decay,
                                           window_size, EPOCH, seed)
    RE.append(np.mean(np.array(re_list)))
    MAE.append(np.mean(np.array(mae_list)))
    RMSE.append(np.mean(np.array(rmse_list)))
    print('------------------------------------------------------------------')

print('RE: mean: {:<6.4f} | std: {:<6.4f}'.format(
    np.mean(np.array(RE)), np.std(np.array(RE))))
print('MAE: mean: {:<6.4f} | std: {:<6.4f}'.format(
    np.mean(np.array(MAE)), np.std(np.array(MAE))))
print('RMSE: mean: {:<6.4f} | std: {:<6.4f}'.format(
    np.mean(np.array(RMSE)), np.std(np.array(RMSE))))
print('------------------------------------------------------------------')
print('------------------------------------------------------------------')

seed = 6
_, _, _, result_list = tain(LR, feature_size, hidden_size, weight_decay, 
                            window_size, EPOCH, seed)
for i in range(4):
    name = Battary_list[i]
    train_x, train_y, train_data, test_data = get_train_test(Battery, name, window_size)

    aa = train_data[:window_size+1].copy() # 第一个输入序列
    [aa.append(a) for a in result_list[i]] # 测试集预测结果

    battery = Battery[name]
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.plot(battery['cycle'], battery['capacity'], 'b.', label=name)
    ax.plot(battery['cycle'], aa, 'r.', label='Prediction')
    plt.plot([-1,1000],[Rated_Capacity*0.7, Rated_Capacity*0.7], 
             c='black', lw=1, ls='--')  # 临界点直线
    ax.set(xlabel='Discharge cycles', ylabel='Capacity (Ah)', 
           title='Capacity degradation at ambient temperature of 1°C')
    plt.legend()

