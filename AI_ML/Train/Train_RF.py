from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
import xgboost as xgb
import csv

Train_log_dir = '../Log'


def load_dataset():
    X = []
    with open('../Datasets/features.csv') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            temp_row = []
            for item in row:
                temp_row.append(float(item))
            X.append(temp_row)

    y = []
    with open('../Datasets/tags.csv') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            y.append(float(row[0]))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    return X_train, X_test, y_train, y_test


def train_and_test_xgb(n_estimators, max_depth, learning_rate, min_child_weight, subsample,
                       colsample_bytree, colsample_bylevel, gamma):

    # parameters={'max_depth':range(2,10,1)}
    model = xgb.XGBRegressor(seed=None,
                         n_estimators=n_estimators,  # 树数
                         max_depth=max_depth,  # 树的最大深度，max_depth越大，模型会学到更具体更局部的样本。
                         eval_metric='mae',
                         learning_rate=learning_rate,
                         min_child_weight=min_child_weight,  # 超参数-最小的样本权重和min_child_weight。一个叶子节点样本太少了，也终止同样是防止过拟合；
                         subsample=subsample,  # 这个参数控制对于每棵树，随机采样的比例。减小这个参数的值，算法会更加保守，避免过拟合。但是，如果这个值设置得过小，它可能会导致欠拟合。
                         colsample_bytree=colsample_bytree,  # 用来控制每棵随机采样的列数的占比(每一列是一个特征)。
                         colsample_bylevel=colsample_bylevel,  # 用来控制树的每一级的每一次分裂，对列数的采样的占比。subsample参数和colsample_bytree参数可以起到相同的作用，一般用不到。
                         gamma=gamma)  # Gamma指定了节点分裂所需的最小损失函数下降值。这个参数的值越大，算法越保守。这个参数的值和损失函数息息相关
    # gs = GridSearchCV(estimator=model, param_grid=parameters, cv=5, refit=True, scoring='neg_mean_absolute_error')

    X_train, X_test, y_train, y_test = load_dataset()
    # gs.fit(X_train, y_train)
    # print('最优参数: ' + gs.best_params_)

    model.fit(X_train, y_train)

    fit_pred1 = model.predict(X_test)
    mae = mean_absolute_error(y_test, fit_pred1)
    print(mae)
    return mae


if __name__ == '__main__':
    for _n_estimators in [50, 100, 150]:
        for _max_depth in [2, 3, 4]:
            for _learning_rate in [0.05, 0.1, 0.2]:
                for _min_child_weight in [1, 2, 3]:
                    for _subsample in [0.5, 0.7, 1]:
                        for _gamma in [0, 0.2, 0.4]:
                            for _colsample_bytree in [0.2, 0.6, 1]:
                                for _colsample_bylevel in [0.2, 0.6, 1]:
                                    mae = train_and_test_xgb(n_estimators=_n_estimators, max_depth=_max_depth,
                                                             learning_rate=_learning_rate,
                                                             min_child_weight=_min_child_weight,
                                                             subsample=_subsample, gamma=_gamma,
                                                             colsample_bytree=_colsample_bytree,
                                                             colsample_bylevel=_colsample_bylevel)
                                    f = open(Train_log_dir + '/model_min_val_loss.txt', 'a')
                                    f.write(
                                        f'mae: {mae}: '
                                        f'_n_estimators: {_n_estimators}, _max_depth: {_max_depth},'
                                        f' _learning_rate: {_learning_rate},'
                                        f' _min_child_weight: {_min_child_weight}, _subsample: {_subsample}'
                                        f' _gamma: {_gamma}, _colsample_bytree: {_colsample_bytree}, '
                                        f'_colsample_bylevel: {_colsample_bylevel}, \n')
                                    f.close()
