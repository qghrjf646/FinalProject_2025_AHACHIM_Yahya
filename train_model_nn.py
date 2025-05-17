import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

if __name__ == '__main__':
    X = pd.read_csv('content_features_X.csv', index_col=0).values.astype(np.float32)
    y = pd.read_csv('content_features_y.csv', index_col=0).values.ravel().astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=256, validation_split=0.1)
    loss, acc = model.evaluate(X_test, y_test)
    print(f'Neural Network Test Accuracy: {acc:.4f}')
    model.save('nn_content_model.h5')
    print('Neural network content-based model saved as nn_content_model.h5')
