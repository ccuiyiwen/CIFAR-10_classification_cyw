import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 数据预处理
# 加载 CIFAR-10 数据集
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# 归一化处理：将像素值从 0-255 缩放到 0-1 之间
x_train, x_test = x_train / 255.0, x_test / 255.0

# 将标签扁平化
y_train = y_train.flatten()
y_test = y_test.flatten()

# 自定义 Keras 分类器以便在 scikit-learn 中使用
class KerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, build_fn, epochs=20, batch_size=64, verbose=0):
        self.build_fn = build_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None
        self.classes_ = None
        self.history = None  # 用于存储训练历史记录
    def fit(self, X, y, sample_weight=None):
        self.model = self.build_fn()
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        if sample_weight is not None:
            sample_weight = sample_weight.astype(np.float32)
            self.history = self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose,
                           sample_weight=sample_weight)
        else:
            self.history = self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        self.classes_ = np.unique(y)
        return self
    def predict(self, X):
        y_pred = self.model.predict(X)
        return np.argmax(y_pred, axis=1)
    def predict_proba(self, X):
        return self.model.predict(X)

# 定义全连接前馈神经网络
def build_fcnn():
    model = models.Sequential([
        layers.Flatten(input_shape=(32, 32, 3)),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# 使用自定义的 Keras 分类器
keras_clf = KerasClassifier(build_fn=build_fcnn, epochs=20, batch_size=64, verbose=1)

# 使用 AdaBoostClassifier
ada_clf = AdaBoostClassifier(base_estimator=keras_clf, n_estimators=10, algorithm='SAMME')

# 训练模型
ada_clf.fit(x_train, y_train)

# 获取 Keras 模型的训练历史记录
history = keras_clf.history

# 可视化训练过程中的损失曲线和准确率曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.show()

# 预测
y_pred = ada_clf.predict(x_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Test accuracy: {accuracy}')

# 混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# 样本预测结果展示
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def plot_sample_predictions(x, y_true, y_pred_classes, class_names, num_samples=5):
    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(x[i])
        plt.title(f'True: {class_names[y_true[i]]}\nPred: {class_names[y_pred_classes[i]]}')
        plt.axis('off')
    plt.show()

plot_sample_predictions(x_test, y_test, y_pred, class_names)
