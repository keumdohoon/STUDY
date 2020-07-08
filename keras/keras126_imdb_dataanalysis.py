#데이터 분석

from keras.datasets import reuters, imdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

word_index = imdb.get_word_index()

word_index = {k : (v+3) for k,v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


print(decode_review(reverse_word_index))
print(len(reverse_word_index[0]))
# (x_train, y_train), (x_test, y_test) =reverse_word_index(num_words=5000)


