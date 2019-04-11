# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
fore_step = 3
time_step = 60
train_samples = 1144
test_samples = 20
feature_size = 2
volume_downsize = 10000
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values
volume_set = dataset_train.iloc[:, 5:6].values
volume_set = volume_set / volume_downsize
#epochs = 180
#batch_size = 32

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
volume_set_scaled = sc.fit_transform(volume_set)

# Creating a data structure with 60 timesteps and 1 output
V_train = []
X_train = []
y_train = []
#for i in range(60, samples):

#try two day ahead so it's 60~1256, shift two row
for i in range(time_step, train_samples-fore_step):
    X_train.append(training_set_scaled[i-time_step:i, 0])
    V_train.append(volume_set_scaled[i-time_step:i, 0])
    y_train.append(training_set_scaled[i+fore_step, 0])
y_train = np.array(y_train)

#X_train = np.array(X_train)
#V_train = np.array(V_train)

mer_train = [None]*(len(X_train)+len(V_train))
mer_train[::2] = X_train
mer_train[1::2] = V_train
X_train = np.array(mer_train)
# Reshaping
#X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_train = np.reshape(X_train, (train_samples-time_step-fore_step, time_step, feature_size))



# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import load_model
from pathlib import Path
#import os  

##os.path.isfile('regressor.h5')

my_file = Path("regressor.h5")
if my_file.is_file():

    # returns a compiled model
    # identical to the previous one
    regressor = load_model('regressor.h5')
else:
    # Initialising the RNN
    regressor = Sequential()

    # Adding the first LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50, return_sequences = True,
                       input_shape = (X_train.shape[1], feature_size)))
    regressor.add(Dropout(0.2))

    # Adding a second LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))

    # Adding a third LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))

    # Adding a fourth LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50))
    regressor.add(Dropout(0.2))

    # Adding the output layer
    regressor.add(Dense(units = 1))

    # Compiling the RNN
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

    # Fitting the RNN to the Training set
    regressor.fit(X_train, y_train, epochs=180, batch_size=32)
    regressor.save('regressor.h5')  # creates a HDF5 file 'my_model.h5'
    #del model  # deletes the existing model



# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
volume_total = pd.concat((dataset_train['Volume'], dataset_test['Volume']), axis = 0)
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)

volume_inputs = volume_total[len(volume_total) - len(dataset_test) - time_step :].values
volume_inputs = volume_inputs / volume_downsize
volume_inputs = volume_inputs.reshape(-1,1)
volume_inputs = sc.transform(volume_inputs)

inputs = dataset_total[len(dataset_total) - len(dataset_test) - time_step :].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)


V_test = []
X_test = []
for i in range(time_step, time_step+test_samples):
    V_test.append(volume_inputs[i-time_step:i, 0])
    X_test.append(inputs[i-time_step:i, 0])
mer_train = [None]*(len(X_test)+len(V_test))
mer_train[::2] = X_test
mer_train[1::2] = V_test
X_test = np.array(mer_train)
#X_test = np.array(X_test)
#X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
print(X_test.shape[0])
X_test = np.reshape(X_test, (test_samples, time_step, feature_size))


#output 3 day ahead
predicted_stock_price = regressor.predict(X_test)
#want to extract the first 2 rows and first 3 columns A_NEW = A[0:2,0:3]
filler_test = predicted_stock_price[0:3, :]
#filler_test = np.reshape(filler_test, (fore_step,1))
#predicted_stock_price = np.vstack((filler_test, predicted_stock_price))
#print(type(predicted_stock_price))  #'numpy.ndarray'
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
#print(predicted_stock_price.shape)  #(20,1)

# Visualising the results
plt.grid(linestyle='--',  color='black')
#plt.grid(color='gray', linestyle='dashed',  linewidth='0.5')
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
