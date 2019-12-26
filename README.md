#### Cyclical Learning rate (CLR)

Cyclical Learning Rate implementation for Keras

Benefits:
  * No need for tuning LR
  * CLR methods require no additional computation (vs Adaptive Learning Rates)
  * CLR methods allows to get out of the saddles points plateaus

  https://arxiv.org/pdf/1506.01186.pdf
  
  Author: Leslie N. Smith

```python
CLR = CyclicalLearningRate(min_lr=0.0001, max_lr=0.001, stepsize=200, cyclical_type="triangular")

Model.compile(optimizer = 'Adam', loss = "binary_crossentopy", metrics = ['accuracy'])
Model.fit(X_train, y_train, batch_size = 128, epochs = 3, callbacks = [CLR])

CLR.lr_history()

```
