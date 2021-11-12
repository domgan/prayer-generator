import os
from pathlib import Path
import tensorflow as tf
import numpy as np
import argparse
from pprint import pprint


class PrayerGenerator:
    def __init__(self):
        self.text = ''
        self.coder = None
        self.decoder = None
        self.to_replace = ('\n', '(', ')')

        self.model = None
        self.model_checkpoint_dir = None

    def load_data(self, path: [str, Path] = 'data'):
        path = Path(path)
        for file in os.listdir(path):
            with open(path / file) as f:
                line = f.read()
                for char in self.to_replace:
                    line = line.replace(char, ' ')
                self.text += f'{line} '
        self._word_embedding()

    def _word_embedding(self):
        vocab = np.array(sorted(set(self.text)))
        self.coder = {u: i for i, u in enumerate(vocab)}
        self.decoder = np.array(vocab)
        self.data = np.array([self.coder[c] for c in self.text])

    def load_model(self, generator_model):
        self.model_checkpoint_dir = generator_model.checkpoint_dir
        self.model = generator_model.model
        # todo o co B?
        if self.model:
            self.model.reset_states()
        self.model = generator_model.build_model(generator_model.vocab_size, batch_size=1)
        self.model.load_weights(tf.train.latest_checkpoint(self.model_checkpoint_dir))
        self.model.build(tf.TensorShape([1, None]))

    def generate_text(self, start_string, num_generate=100, t=0.1):
        start_string = f'{start_string} '
        # Evaluation step (generating text using the learned model)
        # Converting our start string to numbers (vectorizing)
        input_eval = [self.coder[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)
        # Empty string to store our results
        text_generated = []
        # Low temperature results in more predictable text.
        # Higher temperature results in more surprising text.
        # Experiment to find the best setting.
        temperature = t
        # Here batch size == 1
        for i in range(num_generate):
            predictions = self.model(input_eval)
            # remove the batch dimension
            predictions = tf.squeeze(predictions, 0)
            # using a categorical distribution to predict the character returned by the model
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
            # Pass the predicted character as the next input to the model
            # along with the previous hidden state
            input_eval = tf.expand_dims([predicted_id], 0)
            text_generated.append(self.decoder[predicted_id])
        return start_string + ''.join(text_generated)


class GeneratorModel:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.seq_length = 100
        self.examples_per_epoch = len(self.data) // (self.seq_length + 1)
        self.vocab_size = len(set(self.data))
        self.model = None
        self.checkpoint_dir = Path('training_checkpoints')

    def build_model(self, embedding_dim=256, rnn_units=1024, batch_size=10):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, embedding_dim,
                                      batch_input_shape=[batch_size, None]),
            tf.keras.layers.GRU(rnn_units,
                                return_sequences=True,
                                stateful=True,
                                recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(self.vocab_size)
        ])
        return model

    @staticmethod
    def _loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    @staticmethod
    def _split_input_target(chunk):
        input_TEXT = chunk[:-1]
        target_TEXT = chunk[1:]
        return input_TEXT, target_TEXT

    def train(self):
        # Create training examples / targets
        char_dataset = tf.data.Dataset.from_tensor_slices(self.data)
        sequences = char_dataset.batch(self.seq_length + 1, drop_remainder=True)
        dataset = sequences.map(self._split_input_target)
        BATCH_SIZE = 10
        EPOCHS = 30
        BUFFER_SIZE = 10000  # to shuffle in memory
        dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
        model = self.build_model(self.vocab_size, batch_size=BATCH_SIZE)
        # todo o co B?
        for input_example_batch, target_example_batch in dataset.take(1):
            example_batch_predictions = model(input_example_batch)
            print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
        # example_batch_loss = self._loss(target_example_batch, example_batch_predictions)
        model.compile(optimizer='adam', loss=self._loss)
        checkpoint_prefix = self.checkpoint_dir / "ckpt_{epoch}"
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)
        history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
        self.model = model
        return history


def _parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('start_str', help='Starting string.', type=str)
    parser.add_argument('words_num', help='Number of words to generate.', nargs='?', default=100, type=int)
    parser.add_argument('temp', help='Temperature. Values 0-1. Higher value means more inventiveness.', nargs='?',
                        default=0.1, type=float)
    parser.add_argument('--train', '-t', help='Pass to train model.', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    ARGS = _parse_command_line()
    generator = PrayerGenerator()
    generator.load_data()
    gen_model = GeneratorModel(generator.data)
    if ARGS.train:
        gen_model.train()
    generator.load_model(gen_model)
    pprint(generator.generate_text(ARGS.start_str, ARGS.words_num, ARGS.temp))
