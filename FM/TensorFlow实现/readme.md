输出结果
```
Train on 455 samples, validate on 114 samples
Epoch 1/5
455/455 [==============================] - 0s 743us/sample - loss: 25681.2843 - binary_accuracy: 0.3736 - val_loss: 18785.1864 - val_binary_accuracy: 0.3684
Epoch 2/5
455/455 [==============================] - 0s 39us/sample - loss: 17160.1412 - binary_accuracy: 0.3736 - val_loss: 10740.5933 - val_binary_accuracy: 0.3772
Epoch 3/5
455/455 [==============================] - 0s 36us/sample - loss: 7916.0365 - binary_accuracy: 0.4198 - val_loss: 2434.8274 - val_binary_accuracy: 0.5965
Epoch 4/5
455/455 [==============================] - 0s 36us/sample - loss: 1223.8649 - binary_accuracy: 0.7802 - val_loss: 1040.1083 - val_binary_accuracy: 0.8772
Epoch 5/5
455/455 [==============================] - 0s 39us/sample - loss: 932.9886 - binary_accuracy: 0.8308 - val_loss: 673.0605 - val_binary_accuracy: 0.8596
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 30)]         0                                            
__________________________________________________________________________________________________
dense (Dense)                   (None, 1)            31          input_1[0][0]                    
__________________________________________________________________________________________________
fm_layer (FMLayer)              (None, 1)            120         input_1[0][0]                    
__________________________________________________________________________________________________
add (Add)                       (None, 1)            0           dense[0][0]                      
                                                                 fm_layer[0][0]                   
__________________________________________________________________________________________________
activation (Activation)         (None, 1)            0           add[0][0]                        
==================================================================================================
Total params: 151
Trainable params: 151
Non-trainable params: 0
__________________________________________________________________________________________________
```