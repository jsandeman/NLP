import numpy as np
import pandas as pd

sentence = "Thomas Jefferson began building Monticello at the age of 26."

token_sequence = sentence.split()
vocab = sorted(set(token_sequence))
print(', '.join(vocab))

num_tokens = len(token_sequence)
vocab_size = len(vocab)

onehot_vectors = np.zeros((num_tokens,
  vocab_size), int)
print(onehot_vectors)

for i, word in enumerate(token_sequence):
  onehot_vectors[i, vocab.index(word)] = 1

print(onehot_vectors)

onehot_df = pd.DataFrame(onehot_vectors, columns=vocab)
print(onehot_df)


