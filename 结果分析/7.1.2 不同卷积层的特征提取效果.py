import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# 加载 CIFAR-10 数据集
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# 归一化处理
x_test = x_test / 255.0

# 定义 CNN 模型
def build_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# 创建模型
model = build_cnn()

# 加载预训练的权重（假设已经训练好并保存了模型权重）
# model.load_weights('path_to_your_model_weights.h5')

# 选择要可视化的卷积层
layer_outputs = [layer.output for layer in model.layers[:6]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

# 选择一张测试图像
img = x_test[0]
img_tensor = np.expand_dims(img, axis=0)

# 获取卷积层的激活输出
activations = activation_model.predict(img_tensor)

# 可视化第一层卷积层的特征图
first_layer_activation = activations[0]
plt.figure(figsize=(15, 15))
for i in range(32):
    plt.subplot(8, 8, i + 1)
    plt.imshow(first_layer_activation[0, :, :, i], cmap='viridis')
    plt.axis('off')
plt.show()

# 可视化第二层卷积层的特征图
second_layer_activation = activations[2]
plt.figure(figsize=(15, 15))
for i in range(64):
    plt.subplot(8, 8, i + 1)
    plt.imshow(second_layer_activation[0, :, :, i], cmap='viridis')
    plt.axis('off')
plt.show()
