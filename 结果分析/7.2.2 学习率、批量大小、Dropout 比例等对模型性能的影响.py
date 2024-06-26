import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# 加载 CIFAR-10 数据集
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# 归一化处理
x_train, x_test = x_train / 255.0, x_test / 255.0


# 定义 CNN 模型
def build_cnn(dropout_rate=0.5):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(10, activation='softmax')
    ])
    return model


# 超参数设置
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [32, 64, 128]
dropout_rates = [0.3, 0.5, 0.7]

# 存储实验结果
results = {}

for lr in learning_rates:
    for batch_size in batch_sizes:
        for dropout_rate in dropout_rates:
            model = build_cnn(dropout_rate)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            history = model.fit(x_train, y_train, epochs=10, batch_size=batch_size,
                                validation_data=(x_test, y_test), verbose=0)

            key = f'lr={lr}_batch={batch_size}_dropout={dropout_rate}'
            results[key] = history.history

# 可视化结果
fig, axs = plt.subplots(3, 1, figsize=(12, 18))

for key, history in results.items():
    axs[0].plot(history['accuracy'], label=key)
    axs[1].plot(history['val_accuracy'], label=key)
    axs[2].plot(history['val_loss'], label=key)

axs[0].set_title('Training Accuracy')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Accuracy')
axs[0].legend(loc='best')

axs[1].set_title('Validation Accuracy')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
axs[1].legend(loc='best')

axs[2].set_title('Validation Loss')
axs[2].set_xlabel('Epoch')
axs[2].set_ylabel('Loss')
axs[2].legend(loc='best')

plt.show()
