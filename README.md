#### Cyclical Learning rate: Keras implementation

```python
CLR = CyclicalLearningRate(min_lr=0.0001, max_lr=0.001, stepsize=200, cyclical_type="triangular")

Model.compile(optimizer = 'Adam', loss = 'binary_crossentopy", metrics = ['accuracy'])
Model.fit(X_train, y_train, batch_size = 128, epochs = 3, callbacks = [CLR])

CLR.lr_history()

```
