import tensorflow as tf


def create_model(config, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=config.max_words, output_dim=64, input_length=config.max_len),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model


def train_model(model, x_train, y_train, x_test, y_test, epochs=25, learning_rate=0.0002):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))

    return model
