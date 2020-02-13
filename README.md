#### Cyclical Learning rate (CLR)

Implémentation du Cyclical Learning Rate (CLR) proposé par Leslie N. Smith
https://arxiv.org/pdf/1506.01186.pdf

Avantages:
* Gain de temps sur la partie relative à l'optimisation du learning rate
* Permet de sortir rapidement des minimas locaux

```python
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Bidirectional, Flatten, Dense
from Utils.selfAttention import SelfAttentionLayer
from Utils.CyclicalLR import CLR

model = Sequential()
model.add(Embedding(len(tokenizer.word_index), 300, input_length = word_seq_train.shape[1], weights = [embedding_matrix]))
model.add(Bidirectional(LSTM(64, return_sequences = True)))
model.add(SelfAttention(300, 15, attention_regularizer_weights = 0.5, return_attention = False))
model.add(Flatten())
model.add(Dense(2))

CLR = CyclicalLearningRate(min_lr = 0.0001, max_lr = 0.001, stepsize = 4000, cyclical_type = "triangular")

model.compile(optimizer = "Adam", loss = "binary_crossentropy", metrics = ['accuracy'])

model.fit(word_seq_train, y_train, batch_size = 128, epochs = 3, validation_data = (word_seq_test, y_test), callbacks = [CLR])

CLR.lr_history()
```
