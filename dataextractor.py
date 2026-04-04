import struct
import numpy as np
import pandas as pd

def load_images(filename):
    # Используем обычный open() вместо gzip.open()
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        return images.reshape(num, rows * cols)

def load_labels(filename):
    # Используем обычный open() вместо gzip.open()
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

# Указываем пути к файлам в папке data (берем те, что с точкой .idx3/.idx1)
X_train = load_images('data/train-images.idx3-ubyte')
y_train = load_labels('data/train-labels.idx1-ubyte')

print(f"Размерность X_train: {X_train.shape}")
print(f"Размерность y_train: {y_train.shape}")

# --- Дальше можно сразу делать CSV ---
y_col = y_train.reshape(-1, 1)
dataset = np.hstack((y_col, X_train))
column_names = ['label'] + [f'pixel{i}' for i in range(1, 785)]

df = pd.DataFrame(dataset, columns=column_names)
print("Сохраняем в CSV...")
df.to_csv('data/mnist_train.csv', index=False)
print("Файл mnist_train.csv успешно создан в папке data!")