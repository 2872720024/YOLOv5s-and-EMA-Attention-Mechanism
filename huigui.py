import numpy as np


def linear_regression(x, y):
    # 添加偏置项，将x转换为矩阵形式
    x = np.c_[np.ones((x.shape[0], 1)), x]

    # 计算最优参数
    theta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)

    return theta


# 样本数据
x = np.array([257.19,416.01,964.06,894.50,529.82])
y = np.array([0, -90,90 , 80,-80,])

# 调用线性回归函数
LR = linear_regression(x, y)

print("最优参数：", LR)
