import tensorflow as tf
from tensorflow.keras import layers, models, Input

# Tek seferlik veri yükleme
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Önişleme
x_train = x_train.reshape(-1,28,28,1).astype("float32")/255.0
x_test  = x_test .reshape(-1,28,28,1).astype("float32")/255.0

# Model
model = models.Sequential([
    Input(shape=(28,28,1)),                         # ← buraya ekledik
    layers.Conv2D(32, (3,3), padding="same", activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), padding="same", activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test doğruluk: {test_acc:.4f}")
