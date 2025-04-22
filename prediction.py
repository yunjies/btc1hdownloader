import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# 数据准备与特征工程
def prepare_data(data):
    # 提取特征
    features = data[['open', 'high', 'low', 'close', 'volume']]
    # 计算简单的技术指标作为额外特征，如移动平均线
    features['ma_5'] = features['close'].rolling(window=5).mean()
    features['ma_20'] = features['close'].rolling(window=20).mean()
    features = features.dropna()

    # 目标变量：预测下一个时间步的最高价、最低价和收盘价
    target_high = features['high'].shift(-1).dropna()
    target_low = features['low'].shift(-1).dropna()
    target_close = features['close'].shift(-1).dropna()

    # 确保特征和目标变量长度一致
    features = features[:len(target_high)]

    # 数据归一化
    scaler_features = MinMaxScaler()
    scaler_high = MinMaxScaler()
    scaler_low = MinMaxScaler()
    scaler_close = MinMaxScaler()

    scaled_features = scaler_features.fit_transform(features)
    scaled_target_high = scaler_high.fit_transform(target_high.values.reshape(-1, 1))
    scaled_target_low = scaler_low.fit_transform(target_low.values.reshape(-1, 1))
    scaled_target_close = scaler_close.fit_transform(target_close.values.reshape(-1, 1))

    # 划分训练集和测试集
    X_train, X_test, y_train_high, y_test_high, y_train_low, y_test_low, y_train_close, y_test_close = train_test_split(
        scaled_features, scaled_target_high, scaled_target_low, scaled_target_close, test_size=0.2, shuffle=False
    )

    # 调整输入数据形状以适应 LSTM 模型
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    return X_train, X_test, y_train_high, y_test_high, y_train_low, y_test_low, y_train_close, y_test_close, scaler_high, scaler_low, scaler_close


# 构建 LSTM 模型
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# 预测与结果反归一化
def predict_and_inverse_transform(model, X_test, scaler):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    return predictions


# 区间交易法
def range_trading_strategy(data):
    # 假设已经有了历史的最高价和最低价数据
    historical_high = data['high'].rolling(window=20).max()
    historical_low = data['low'].rolling(window=20).min()

    # 确定区间
    upper_bound = historical_high
    lower_bound = historical_low

    # 生成交易信号
    data['signal'] = 0
    data.loc[data['close'] > upper_bound, 'signal'] = -1  # 卖出信号
    data.loc[data['close'] < lower_bound, 'signal'] = 1  # 买入信号

    # 计算持仓
    data['position'] = data['signal'].diff()

    # 回测交易策略
    initial_capital = 10000
    positions = pd.DataFrame(index=data.index).fillna(0.0)
    positions['BTC'] = data['signal']

    portfolio = positions.multiply(data['close'], axis=0)
    pos_diff = positions.diff()

    portfolio['holdings'] = (positions.multiply(data['close'], axis=0)).sum(axis=1)
    portfolio['cash'] = initial_capital - (pos_diff.multiply(data['close'], axis=0)).sum(axis=1).cumsum()
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']
    portfolio['returns'] = portfolio['total'].pct_change()

    # 计算胜率和盈亏比
    trades = portfolio[portfolio['position'] != 0]
    wins = trades[trades['returns'] > 0]
    losses = trades[trades['returns'] < 0]

    win_rate = len(wins) / len(trades) if len(trades) > 0 else 0
    risk_reward_ratio = wins['returns'].mean() / abs(losses['returns'].mean()) if len(losses) > 0 else 0

    print(f"胜率: {win_rate * 100:.2f}%")
    print(f"盈亏比: {risk_reward_ratio:.2f}")


def do_prediction(data):
    # 数据准备
    X_train, X_test, y_train_high, y_test_high, y_train_low, y_test_low, y_train_close, y_test_close, scaler_high, scaler_low, scaler_close = prepare_data(
        data)

    # 构建并训练预测最高价的模型
    model_high = build_lstm_model(X_train.shape[1:])
    model_high.fit(X_train, y_train_high, epochs=50, batch_size=32, validation_split=0.1)

    # 构建并训练预测最低价的模型
    model_low = build_lstm_model(X_train.shape[1:])
    model_low.fit(X_train, y_train_low, epochs=50, batch_size=32, validation_split=0.1)

    # 构建并训练预测收盘价的模型
    model_close = build_lstm_model(X_train.shape[1:])
    model_close.fit(X_train, y_train_close, epochs=50, batch_size=32, validation_split=0.1)

    # 进行预测
    predictions_high = predict_and_inverse_transform(model_high, X_test, scaler_high)
    predictions_low = predict_and_inverse_transform(model_low, X_test, scaler_low)
    predictions_close = predict_and_inverse_transform(model_close, X_test, scaler_close)

    # 计算回撤量
    with np.errstate(divide='ignore', invalid='ignore'):
        drawdowns = (predictions_high - predictions_low) / predictions_high

    return f"预测的最高价: {predictions_high}\n\
    预测的最低价: {predictions_low}\n\
    预测的收盘价: {predictions_close}\n\
    回撤量: {drawdowns}"

    # # 执行区间交易策略
    # range_trading_strategy(data)
    