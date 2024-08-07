import torch
import math
import numpy as np

dtype = torch.float
device = torch.device("cpu")

# Create random input and output data
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# Randomly initialize weights
a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights using gradient descent
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')

llis = [1, 2, 2, 3, 4, 6, 7]
print(llis[3])
print(llis[3:4])

# 定义原始数组
arr = [
    ['a', 'b', 'c'],
    ['d', 'e', 'f'],
    ['g', 'h', 'i']
]

# 使用 zip 和列表推导式进行转置
transposed_arr = [list(row) for row in zip(*arr)]

# 输出结果
print(transposed_arr)

arr1 = [1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1,
        0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]
print(arr1.count(1))
print(arr1.count(0))
print(len(arr1))

import random

# 假设arr1和arr2是已经定义好的列表
arr1 = ['用户数据1', '用户数据2', '用户数据3']
arr2 = [101, 102, 103]

# 步骤1: 创建一个包含(arr2[i], arr1[i])对的列表
paired_list = list(zip(arr2, arr1))

# 步骤2: 随机打乱paired_list
random.shuffle(paired_list)

# 步骤3: 根据打乱后的paired_list重新构建arr1和arr2
arr2_shuffled, arr1_shuffled = zip(*paired_list)  # 解压成两个列表
print(arr2_shuffled)
# 将zip对象转换回列表
arr1 = list(arr1_shuffled)
arr2 = list(arr2_shuffled)

# 打印结果，查看arr1和arr2是否同步变化
print("调整后的arr2:", arr2)
print("调整后的arr1:", arr1)
