import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# 1. 数据预处理
# 加载 CIFAR-10 数据集
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# 归一化处理：将像素值从 0-255 缩放到 0-1 之间
x_train, x_test = x_train / 255.0, x_test / 255.0

# 数据增强
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

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

# 编译模型
model = build_cnn()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型（不使用数据增强）
history_no_aug = model.fit(x_train, y_train, epochs=20,
                           validation_data=(x_test, y_test))

# 重置模型
model = build_cnn()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型（使用数据增强）
history_with_aug = model.fit(datagen.flow(x_train, y_train, batch_size=64),
                             steps_per_epoch=len(x_train) / 64, epochs=20,
                             validation_data=(x_test, y_test))

# 绘制训练结果
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 绘制准确率曲线
ax1.plot(history_with_aug.history['accuracy'], label='Train Accuracy without Augmentation')
ax1.plot(history_with_aug.history['val_accuracy'], label='Validation Accuracy without Augmentation')
ax1.plot(history_no_aug.history['accuracy'], label='Train Accuracy with Augmentation')
ax1.plot(history_no_aug.history['val_accuracy'], label='Validation Accuracy with Augmentation')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.set_title('Training and Validation Accuracy')
ax1.legend(loc='center left', bbox_to_anchor=(0.5, 0.5))

# 绘制损失曲线
ax2.plot(history_with_aug.history['loss'], label='Train Loss without Augmentation')
ax2.plot(history_with_aug.history['val_loss'], label='Validation Loss without Augmentation')
ax2.plot(history_no_aug.history['loss'], label='Train Loss with Augmentation')
ax2.plot(history_no_aug.history['val_loss'], label='Validation Loss with Augmentation')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Training and Validation Loss')
ax2.legend(loc='center left', bbox_to_anchor=(0.5, 0.5))

plt.show()

