import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import ProgbarLogger


# 禁用 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 计算 CVD（累积成交量 delta）
def calculate_cvd(data):
    # 将字符串类型的 volume 和 taker_buy_base_asset_volume 转换为浮点数
    data['volume'] = data['volume'].astype(float)
    data['taker_buy_base_asset_volume'] = data['taker_buy_base_asset_volume'].astype(float)

    # 主动买盘（Taker Buy）
    taker_buy = data['taker_buy_base_asset_volume']
    # 主动卖盘 = 总成交量 - 主动买盘
    taker_sell = data['volume'] - taker_buy
    # 单周期 Volume Delta = 主动买盘 - 主动卖盘
    data['volume_delta'] = taker_buy - taker_sell
    # 累积 CVD
    data['cvd'] = data['volume_delta'].cumsum()
    return data


def prepare_data(data, future_hours=1):
    # 计算 CVD 指标
    data = calculate_cvd(data)

    # 要转换为数值类型的列
    columns_to_convert = ['open', 'high', 'low', 'close', 'volume', 'cvd']
    for col in columns_to_convert:
        data[col] = data[col].astype(float)

    # 提取特征（包含 CVD），使用 .copy() 确保是独立的 DataFrame
    features = data[columns_to_convert].copy()

    # 计算其他技术指标（MA、RSI 等，可选）
    features['ma_5'] = features['close'].rolling(window=5).mean()
    features['ma_20'] = features['close'].rolling(window=20).mean()

    # 计算 RSI（示例，可选）
    delta = features['close'].diff()
    up = delta.where(delta > 0, 0)
    down = -delta.where(delta < 0, 0)
    avg_up = up.rolling(14).mean()
    avg_down = down.rolling(14).mean()
    rs = avg_up / avg_down
    features['rsi'] = 100 - (100 / (1 + rs))

    features = features.dropna()

    # 目标变量：预测未来 N 小时的收盘价
    target_close = features['close'].shift(-future_hours).dropna()
    features = features[:len(target_close)]

    # 数据归一化（包含 CVD 等新特征）
    scaler_features = MinMaxScaler()
    scaler_close = MinMaxScaler()

    scaled_features = scaler_features.fit_transform(features)
    scaled_target_close = scaler_close.fit_transform(target_close.values.reshape(-1, 1))

    # 划分训练集和测试集
    X_train, X_test, y_train_close, y_test_close = train_test_split(
        scaled_features, scaled_target_close, test_size=0.2, shuffle=False
    )

    # 调整输入形状：(样本数, 时间步, 特征数)，这里时间步=1（单步预测）
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    return X_train, X_test, y_train_close, y_test_close, scaler_features, scaler_close

# 构建 LSTM 模型
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(units=128, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=64))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 预测与结果反归一化
def predict_and_inverse_transform(model, X_test, scaler):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    return predictions

# 新增 predict_close_prices 函数
def predict_close_prices(data, future_hours=1, force_retrain=False):
    # 数据准备
    X_train, X_test, y_train_close, y_test_close, scaler_features, scaler_close = prepare_data(data, future_hours)

    if force_retrain:
        # 构建并训练预测收盘价的模型
        model_close = build_lstm_model(X_train.shape[1:])
        model_close.save('./data/close_price_model.keras')
        model_close.fit(X_train, y_train_close, epochs=50, batch_size=32, validation_split=0.1)
    else:
        # 这里可以添加加载已有模型的逻辑，如果存在的话
        # 示例：model_close = load_model('close_price_model.keras')
        # 假设没有已有模型，暂时还是重新训练
        model_close = load_model('./data/close_price_model.keras')
        # 添加 ProgbarLogger 回调函数
        model_close.fit(X_train, y_train_close, epochs=50, batch_size=32, validation_split=0.1)

    # 进行预测
    predictions_close = predict_and_inverse_transform(model_close, X_test, scaler_close)
    return predictions_close

def do_prediction(data, future_hours=1):
    # 数据准备
    X_train, X_test, y_train_close, y_test_close, scaler_features, scaler_close = prepare_data(data, future_hours)

    # 构建并训练预测收盘价的模型
    model_close = build_lstm_model(X_train.shape[1:])
    model_close.fit(X_train, y_train_close, epochs=50, batch_size=32, validation_split=0.1)

    # 进行预测
    predictions_close = predict_and_inverse_transform(model_close, X_test, scaler_close)

    return f"预测未来 {future_hours} 小时的收盘价: {predictions_close}"