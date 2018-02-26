## Training LSTM Model for Sentiment Analysis with Keras

This project is based on the [Trains an LSTM model on the IMDB sentiment classification task with Keras](https://github.com/keras-team/keras/blob/master/examples/imdb_lstm.py)


To train LSTM Model using IMDB review dataset, run train_lstm_with_imdb_review.py through command line:

```bash
$ python3 train_lstm_with_imdb_review.py -bs 32 -ep 15
/usr/lib/python3.6/importlib/_bootstrap.py:219: RuntimeWarning: compiletime version 3.5 of module 'tensorflow.python.framework.fast_tensor_util' does not match runtime version 3.6
  return f(*args, **kwds)
/usr/local/lib/python3.6/dist-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.
Loading imdb review data ...

Number of training sequences is 25000 

Number of testing sequences is 25000 

Padding the sequences ...

Training Sequence Shape: (25000, 80) 

Testing Sequence Shape: (25000, 80) 

Start training...

Train on 25000 samples, validate on 25000 samples
Epoch 1/15
2018-02-26 12:07:15.451947: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
25000/25000 [==============================] - 488s 20ms/step - loss: 0.5187 - acc: 0.7449 - val_loss: 0.4058 - val_acc: 0.8205
Epoch 2/15
25000/25000 [==============================] - 501s 20ms/step - loss: 0.3685 - acc: 0.8450 - val_loss: 0.3852 - val_acc: 0.8320
Epoch 3/15
25000/25000 [==============================] - 491s 20ms/step - loss: 0.2998 - acc: 0.8746 - val_loss: 0.4200 - val_acc: 0.8303
Epoch 4/15
25000/25000 [==============================] - 462s 18ms/step - loss: 0.2440 - acc: 0.9060 - val_loss: 0.4286 - val_acc: 0.8359
Epoch 5/15
25000/25000 [==============================] - 414s 17ms/step - loss: 0.1969 - acc: 0.9237 - val_loss: 0.4931 - val_acc: 0.8234
Epoch 6/15
25000/25000 [==============================] - 437s 17ms/step - loss: 0.1636 - acc: 0.9384 - val_loss: 0.5104 - val_acc: 0.8274
Epoch 7/15
25000/25000 [==============================] - 456s 18ms/step - loss: 0.1354 - acc: 0.9509 - val_loss: 0.5200 - val_acc: 0.8201
Epoch 8/15
25000/25000 [==============================] - 532s 21ms/step - loss: 0.1065 - acc: 0.9619 - val_loss: 0.6019 - val_acc: 0.8238
Epoch 9/15
25000/25000 [==============================] - 507s 20ms/step - loss: 0.0826 - acc: 0.9708 - val_loss: 0.6763 - val_acc: 0.8140
Epoch 10/15
25000/25000 [==============================] - 469s 19ms/step - loss: 0.0757 - acc: 0.9734 - val_loss: 0.7057 - val_acc: 0.8170
Epoch 11/15
25000/25000 [==============================] - 492s 20ms/step - loss: 0.0553 - acc: 0.9811 - val_loss: 0.7484 - val_acc: 0.8165
Epoch 12/15
25000/25000 [==============================] - 464s 19ms/step - loss: 0.0457 - acc: 0.9849 - val_loss: 0.8810 - val_acc: 0.8067
Epoch 13/15
25000/25000 [==============================] - 551s 22ms/step - loss: 0.0368 - acc: 0.9876 - val_loss: 0.8788 - val_acc: 0.8088
Epoch 14/15
25000/25000 [==============================] - 533s 21ms/step - loss: 0.0372 - acc: 0.9882 - val_loss: 0.9012 - val_acc: 0.8058
Epoch 15/15
25000/25000 [==============================] - 509s 20ms/step - loss: 0.0304 - acc: 0.9899 - val_loss: 0.9434 - val_acc: 0.8092
25000/25000 [==============================] - 79s 3ms/step
Test score: 0.9433712914848328
Test accuracy: 0.8092
```
