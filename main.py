import os
from pathlib import Path
from typing import Tuple
import tensorflow as tf
import numpy as np


def load_data(path: [str, Path] = 'data') -> str:
    path = Path(path)
    text = ''
    for file in os.listdir(path):
        with open(path / file) as f:
            text += f.read().replace('\n', ' ').replace('(', ' ').replace(')', ' ') + ' '
    return text


def word_embedding(text: str) -> Tuple[np.ndarray, dict, np.ndarray]:
    vocab = np.array(sorted(set(text)))
    coder = {u: i for i, u in enumerate(vocab)}
    decoder = np.array(vocab)
    text_int = np.array([coder[c] for c in text])
    return text_int, coder, decoder


def split_input_target(chunk):
    input_TEXT = chunk[:-1]
    target_TEXT = chunk[1:]
    return input_TEXT, target_TEXT


def build_model(vocab_size, embedding_dim=256, rnn_units=1024, batch_size=10):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def generate_text(model, start_string, t, coder, decoder):
    # Evaluation step (generating text using the learned model)
    # Number of characters to generate
    num_generate = 100
    # Converting our start string to numbers (vectorizing)
    input_eval = [coder[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    # Empty string to store our results
    text_generated = []
    # Low temperature results in more predictable text.
    # Higher temperature results in more surprising text.
    # Experiment to find the best setting.
    temperature = t
    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)
        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        # Pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(decoder[predicted_id])
    return start_string + ''.join(text_generated)


if __name__ == '__main__':
    text = load_data()
    data, coder, decoder = word_embedding(text)
    seq_length = 100
    examples_per_epoch = len(text)//(seq_length+1)
    vocab_size = len(decoder)
    # Create training examples / targets
    char_dataset = tf.data.Dataset.from_tensor_slices(data)
    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
    dataset = sequences.map(split_input_target)
    BATCH_SIZE = 10
    EPOCHS = 30
    BUFFER_SIZE = 10000  # to shuffle in memory
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    model = build_model(vocab_size, batch_size=BATCH_SIZE)
    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

    example_batch_loss = loss(target_example_batch, example_batch_predictions)
    model.compile(optimizer='adam', loss=loss)
    checkpoint_dir = Path('training_checkpoints')
    checkpoint_prefix = checkpoint_dir / "ckpt_{epoch}"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

    model = build_model(vocab_size, batch_size=1)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))

    print(generate_text(model, u"After the ", 0.1, coder, decoder))
