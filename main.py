from model import create_model, train_model
from config import Config
from datapreparation import preprocess, load_glove_embeddings


def main():
    config = Config()
    tokenizer, x_train, y_train, x_test, y_test = preprocess(config)

    embedding_matrix = load_glove_embeddings('glove.6B.50d.txt', tokenizer, config.embedding_dim)

    model = create_model(config, len(set(y_train)), embedding_matrix)
    model = train_model(model, x_train, y_train, x_test, y_test)

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    model.save('bbc_trained_model.h5')
    print("Model saved successfully.")


if __name__ == "__main__":
    main()
