# Keras functional API
## New Sequential Model Definition
```python
from keras.layers import Input, Dense
from keras.models import Model

inputs = Input(shape=(784,))

x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
preds = Dense(10, activation='softmax')(x)

model = Model(input=inputs, output=preds)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(data, labels)
```

whenever `model` is defined with inputs and outputs,
it can be called as a layer and re-uses its weights

for single input and single output, the `model` can still
be easily defined with `.add` method

```python
from keras.models import Sequential

model = Sequential()
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))

model.compile ....
model.fit ...
```

## Turn image classification model to video classification
```python
from keras.layers import TimeDistributed

input_sequences = Input(shape=(20, 784))

# assumed `model` already defined as above
processed_sequences = TimeDistributed(model)(input_sequences)
```

## Multiple Inputs and Multiple Outputs Model
assumed main_input, aux_input, main_output, aux_output
```python
from keras.layers import Input, Embedding LSTM, Dense, merge
from keras.models import Model

# receive sequences of 100 integers, between 1 and 10000
main_input = Input(shape=(100,), dtype='int32', name='main_input')

# encode input sequene into a sequence of dense 512-dimensional vectors
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)

# transfrom vector sequence into single vector
lstm_out = LSTM(32)(x)

aux_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)

aux_input = Input(shape=(5,), name='aux_input')
x = merge([lstm_out, aux_input], mode='concat')

x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

main_output =  Dense(1, activation='sigmoid', name='main_output')(x)

# model definition
model = Model(input=[main_input, aux_input],
              output=[main_output, aux_output])

model.compile(optimizer='rmsprop',
              loss={
                    'main_output': 'binary_crossentropy',
                    'aux_output': 'binary_crossentropy'
                   },
              loss_weights={
                            'main_output': 1.,
                            'aux_output': 0.2
                           }
             )

model.fit(
          {'main_input': headline_data,
           'aux_input': additional_data},
          {'main_output':labels,
           'aux_output': labels},
          nb_epoch=50, batch_size=32
         )
```

## Shared Layers
```python
# this layer can take as input a matrix
# and will return a vector of size 64
shared_lstm = LSTM(64)

# when we reuse the same layer instance
# multiple times, the weights of the layer
# are also being reused
# (it is effectively *the same* layer)
encoded_a = shared_lstm(tweet_a)
encoded_b = shared_lstm(tweet_b)

# we can then concatenate the two vectors:
merged_vector = merge([encoded_a, encoded_b], mode='concat', concat_axis=-1)

# and add a logistic regression on top
predictions = Dense(1, activation='sigmoid')(merged_vector)

# we define a trainable model linking the
# tweet inputs to the predictions
model = Model(input=[tweet_a, tweet_b], output=predictions)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit([data_a, data_b], labels, nb_epoch=10)
```


