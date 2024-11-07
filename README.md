import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# Load the data
data = pd.read_csv('English_Hindi_Clean_New.csv', encoding='utf-8')

# Get English and Hindi vocabularies
all_eng_words = set()
for eng in data['English']:
    for word in eng.split():
        all_eng_words.add(word)

all_hin_words = set()
for hin in data['Hindi']:
    for word in hin.split():
        all_hin_words.add(word)

# Create sentence length columns
data['len_eng_sen'] = data['English'].apply(lambda x: len(x.split(" ")))
data['len_hin_sen'] = data['Hindi'].apply(lambda x: len(x.split(" ")))

# Filter sentences by max length (limit both Hindi and English to 20 tokens)
data = data[data['len_eng_sen'] <= 20]
data = data[data['len_hin_sen'] <= 20]

# Get the maximum length of the sentences
max_len_src = max(data['len_eng_sen'])
max_len_tar = max(data['len_hin_sen'])

# Prepare vocabulary and token index mappings
inp_words = sorted(list(all_eng_words))
tar_words = sorted(list(all_hin_words))

num_enc_toks = len(inp_words) + 1  # +1 for padding index (0)
num_dec_toks = len(tar_words) + 1  # +1 for zero padding

# Create token-to-index and index-to-token mappings
inp_tok_idx = dict((word, i + 1) for i, word in enumerate(inp_words))  # Input word -> index
tar_tok_idx = dict((word, i + 1) for i, word in enumerate(tar_words))  # Target word -> index
rev_inp_tok_idx = dict((i, word) for word, i in inp_tok_idx.items())    # Index -> Input word
rev_tar_tok_idx = dict((i, word) for word, i in tar_tok_idx.items())    # Index -> Target word

# Split the data into train and test sets
X, y = data['English'], data['Hindi']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define batch size
batch_size = 128

# Padding sequences to ensure all sentences have the same length
X_train_padded = pad_sequences(X_train.apply(lambda x: [inp_tok_idx.get(word, 0) for word in x.split()]), maxlen=max_len_src, padding='post')
y_train_padded = pad_sequences(y_train.apply(lambda x: [tar_tok_idx.get(word, 0) for word in x.split()]), maxlen=max_len_tar, padding='post')

X_test_padded = pad_sequences(X_test.apply(lambda x: [inp_tok_idx.get(word, 0) for word in x.split()]), maxlen=max_len_src, padding='post')
y_test_padded = pad_sequences(y_test.apply(lambda x: [tar_tok_idx.get(word, 0) for word in x.split()]), maxlen=max_len_tar, padding='post')

# Create a TensorFlow dataset using from_generator
def generate_batch(X, y, batch_size=batch_size):
    while True:
        for j in range(0, len(X), batch_size):
            enc_inp_data = np.zeros((batch_size, max_len_src), dtype='float32')
            dec_inp_data = np.zeros((batch_size, max_len_tar), dtype='float32')
            dec_tar_data = np.zeros((batch_size, max_len_tar, num_dec_toks), dtype='float32')

            for i, (inp_seq, tar_seq) in enumerate(zip(X[j:j + batch_size], y[j:j + batch_size])):
                enc_inp_data[i, :len(inp_seq)] = inp_seq  # Encoder input data
                for t, tok in enumerate(tar_seq):
                    if t < len(tar_seq) - 1:
                        dec_inp_data[i, t] = tok  # Decoder input (shifted target sequence)
                    if t > 0:
                        dec_tar_data[i, t - 1, tok] = 1.0  # Decoder target (one-hot encoded)

            # Convert to tensors to match the output signature
            yield (tf.convert_to_tensor(enc_inp_data, dtype=tf.float32),
                   tf.convert_to_tensor(dec_inp_data, dtype=tf.float32)), tf.convert_to_tensor(dec_tar_data, dtype=tf.float32)

# Use tf.data.Dataset to wrap the generator
train_dataset = tf.data.Dataset.from_generator(
    lambda: generate_batch(X_train_padded, y_train_padded, batch_size=batch_size),
    output_signature=(
        (tf.TensorSpec(shape=(None, max_len_src), dtype=tf.float32),
         tf.TensorSpec(shape=(None, max_len_tar), dtype=tf.float32)),
        tf.TensorSpec(shape=(None, max_len_tar, num_dec_toks), dtype=tf.float32)
    )
)

val_dataset = tf.data.Dataset.from_generator(
    lambda: generate_batch(X_test_padded, y_test_padded, batch_size=batch_size),
    output_signature=(
        (tf.TensorSpec(shape=(None, max_len_src), dtype=tf.float32),
         tf.TensorSpec(shape=(None, max_len_tar), dtype=tf.float32)),
        tf.TensorSpec(shape=(None, max_len_tar, num_dec_toks), dtype=tf.float32)
    )
)

# Model Architecture
latent_dim = 250

# Encoder
enc_inps = Input(shape=(None,))
enc_emb = Embedding(num_enc_toks, latent_dim, mask_zero=True)(enc_inps)
enc_lstm = LSTM(latent_dim, return_state=True)
enc_outputs, st_h, st_c = enc_lstm(enc_emb)
enc_states = [st_h, st_c]

# Decoder
dec_inps = Input(shape=(None,))
dec_emb_layer = Embedding(num_dec_toks, latent_dim, mask_zero=True)
dec_emb = dec_emb_layer(dec_inps)
dec_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
dec_outputs, _, _ = dec_lstm(dec_emb, initial_state=enc_states)
dec_dense = Dense(num_dec_toks, activation='softmax')
dec_outputs = dec_dense(dec_outputs)

# Define the model
model = Model([enc_inps, dec_inps], dec_outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Train the model with a larger batch size using tf.data.Dataset
model.fit(
    train_dataset,
    steps_per_epoch=len(X_train_padded) // batch_size,
    epochs=5,
    validation_data=val_dataset,
    validation_steps=len(X_test_padded) // batch_size
)

# Inference Setup: Encoder
enc_model = Model(enc_inps, enc_states)

# Inference Setup: Decoder
dec_st_inp_h = Input(shape=(latent_dim,))
dec_st_inp_c = Input(shape=(latent_dim,))
dec_states_inps = [dec_st_inp_h, dec_st_inp_c]

dec_emb2 = dec_emb_layer(dec_inps)
dec_outputs2, st_h2, st_c2 = dec_lstm(dec_emb2, initial_state=dec_states_inps)
dec_states2 = [st_h2, st_c2]
dec_outputs2 = dec_dense(dec_outputs2)

# Final decoder model for inference
dec_model = Model([dec_inps] + dec_states_inps, [dec_outputs2] + dec_states2)

# Translate function for inference
def translate(inp_seq):
    states_value = enc_model.predict(inp_seq)
    tar_seq = np.zeros((1, 1))
    tar_seq[0, 0] = tar_tok_idx.get('START_', 0)  # 'START_' token

    stop_cond = False
    dec_sen = ''
    while not stop_cond:
        output_toks, h, c = dec_model.predict([tar_seq] + states_value)
        sampled_tok_idx = np.argmax(output_toks[0, -1, :])
        sampled_char = rev_tar_tok_idx.get(sampled_tok_idx, '')

        if sampled_char == '_END' or len(dec_sen.split()) > max_len_tar:
            stop_cond = True
        else:
            dec_sen += ' ' + sampled_char

        tar_seq = np.zeros((1, 1))
        tar_seq[0, 0] = sampled_tok_idx
        states_value = [h, c]

    return dec_sen.strip()

# Testing the model on a training example
train_gen = generate_batch(X_train_padded, y_train_padded, batch_size=1)
(inp_seq, actual_output), _ = next(train_gen)
hin_sen = translate(inp_seq)

k = 0
print(f'''Input English sentence: {X_train.iloc[k]}\nPredicted Hindi translation: {hin_sen}\nActual Hindi sentence: {y_train.iloc[k]}''')
