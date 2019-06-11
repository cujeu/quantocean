# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def listsum(numList):
	theSum = 0
	for i in numList:
		theSum = theSum + i
	return theSum

def create_features(stock, symbol):

    open_ = stock.columns[0]
    high = stock.columns[1]
    low =  stock.columns[2]
    volume =  stock.columns[4]
    adj = stock.columns[5]
    return_ =  stock.columns[6]

    '''
         All Technical Indicators
    '''
    MA6 = symbol + "_MA6"
    MA14 = symbol + "_MA14"
    EMA6 = symbol + "_EMA6"
    EMA14 = symbol + "_EMA14"
    MACD = symbol + "_MACD"
    MoM6 = symbol + "_MoM6"
    MoM14 = symbol + "_MoM14"
    K_per6 = symbol + "_K%6"
    WIILR6 = symbol + "_WIILR6"
    K_per14 = symbol + "_K%14"
    WIILR14 = symbol + "_WIILR14"
    RoC6 = symbol + "_RoC6"
    RoC14 = symbol + "_RoC14"
    RSI6 = symbol + "_RSI6"
    RSI14 = symbol + "_RSI14"
    OBV = symbol + "_OBV"

    stock[EMA6] = np.nan
    stock[EMA14] = np.nan
    stock[MoM6] = np.nan
    stock[MoM14] = np.nan
    stock[WIILR6] = np.nan
    stock[WIILR14] = np.nan
    stock[K_per6] = np.nan
    stock[K_per14] = np.nan
    stock[OBV] = np.nan
    stock[RSI6]=np.nan
    stock[RSI14]=np.nan
    stock["change"] = np.nan
    stock["gain"] = float(0)
    stock["loss"] = float(0)
    stock["avgGain6"] = float(0)
    stock["avgLoss6"] = float(0)
    stock["avgGain14"] = float(0)
    stock["avgLoss14"] = float(0)
    stock[EMA6][0] = stock[adj][0]
    stock[EMA14][0] = stock[adj][0]
    stock[OBV][0] = stock[volume][0]

    multi_6 = 2/7
    multi_14 = 2/15

    '''
        Rate of Change, 6 and 14 days
    '''
    stock[RoC6] = stock[adj].pct_change(6)
    stock[RoC14] = stock[adj].pct_change(14)

    # Moving Average, 6 and 14 days
    stock[MA6] = pd.rolling_mean(stock[adj], 6)
    stock[MA14] = pd.rolling_mean(stock[adj], 14)

    shape_stock = stock.shape

    '''
        Relative Strength Index (RSI)
    '''
    for i in range(shape_stock[0]-1):
        stock[EMA6][i+1] = stock[EMA6][i] + multi_6 * (stock[adj][i] - stock[EMA6][i])
        stock[EMA14][i+1] = stock[EMA14][i] + multi_14 * (stock[adj][i] - stock[EMA14][i])

        if stock[adj][i+1] > stock[adj][i]:
            stock[OBV][i+1] = stock[OBV][i] + stock[volume][i+1]
        if stock[adj][i+1] < stock[adj][i]:
            stock[OBV][i+1] = stock[OBV][i] - stock[volume][i+1]
        if stock[adj][i+1] == stock[adj][i]:
            stock[OBV][i+1] = stock[OBV][i]

        stock["change"][i+1] = stock[adj][i+1] - stock[adj][i]
        if stock["change"][i+1] > 0:
            stock["gain"][i+1] = float(stock["change"][i+1])
        else:
            stock["loss"][i+1] = float(stock["change"][i+1])*(-1)

    for i in range(shape_stock[0]-6):
        max_ = max(stock[high][i:i+6])
        min_ = min(stock[low][i:i+6])
        stock[MoM6][i+6] = stock[adj][i+6] - stock[adj][i]
        stock[WIILR6][i+6] = (max_ - stock[adj][i+6])/(max_ - min_)
        stock[K_per6][i+6] = (stock[adj][i+6] - min_)/(max_ - min_)
        stock["avgGain6"][i+6] = listsum(stock["gain"][i:i+6])/6
        stock["avgLoss6"][i+6] = listsum(stock["loss"][i:i+6])/6
        stock[RSI6][i+6] = stock["avgGain6"][i+6]/(stock["avgGain6"][i+6] + stock["avgLoss6"][i+6])

    for i in range(shape_stock[0]-14):
        stock[MoM14][i+14] = stock[adj][i+14] - stock[adj][i]
        stock[WIILR14][i+14] = (max(stock[high][i:i+14]) - stock[adj][i+14])/(max(stock[high][i:i+14]) - min(stock[low][i:i+14]))
        stock[K_per14][i+14] = (stock[adj][i+14] - min(stock[low][i:i+14]))/(max(stock[high][i:i+14]) - min(stock[low][i:i+14]))
        stock["avgGain14"][i+14] = listsum(stock["gain"][i:i+14])/14
        stock["avgLoss14"][i+14] = listsum(stock["loss"][i:i+14])/14
        stock[RSI14][i+14] = stock["avgGain14"][i+14]/(stock["avgGain14"][i+14] + stock["avgLoss14"][i+14])

    stock = stock.drop("change", 1)
    stock = stock.drop("gain", 1)
    stock = stock.drop("loss", 1)
    stock = stock.drop("avgGain6", 1)
    stock = stock.drop("avgLoss6", 1)
    stock = stock.drop("avgGain14", 1)
    stock = stock.drop("avgLoss14", 1)

    stock = stock.fillna(stock.mean())

    name = symbol + "_with_features.csv"
    stock.to_csv(name, sep='\t', encoding='utf-8')

    csv_dir = './csv_with_features/'
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    shutil.move(name, csv_dir)
    print ("Done... ", name)

def create_features_rsi(stock, period):

    """
    Date,Open,High,Low,Close,Volume
    1/3/2017,778.81,789.63,775.8,786.14,1657300
    1/4/2017,788.36,791.34,783.16,786.9,1073000
    """
    ## open_ = stock.columns[1]
    ## high_ = stock.columns[2]
    ## low_ =  stock.columns[3]
    close_ = stock.columns[4]
    
    stock["avgGain14"] = float(0)
    stock["avgLoss14"] = float(0)
    stock["change"] = np.nan
    stock["gain"] = float(0)
    stock["loss"] = float(0)
    stock["avgGain14"] = float(0)
    stock["avgLoss14"] = float(0)
    RSI = "_RSI"
    stock[RSI]=np.zeros

    # Moving Average, 6 and 14 days
    #stock[MA6] = pd.rolling_mean(stock[adj], 6)
    #stock[MA14] = pd.rolling_mean(stock[adj], 14)

    shape_stock = stock.shape

    '''
        Relative Strength Index (RSI)
    '''
    for i in range(shape_stock[0]-1):
        stock["change"][i+1] = stock[close_][i+1] - stock[close_][i]
        if stock["change"][i+1] > 0:
            stock["gain"][i+1] = float(stock["change"][i+1])
        else:
            stock["loss"][i+1] = 0-float(stock["change"][i+1])

    for i in range(shape_stock[0]-period):
        stock["avgGain14"][i+period] = listsum(stock["gain"][i:i+period])/period
        stock["avgLoss14"][i+period] = listsum(stock["loss"][i:i+period])/period
        stock[RSI][i+period] = stock["avgGain14"][i+period]/(stock["avgGain14"][i+period] + stock["avgLoss14"][i+period])

    stock = stock.drop("change", 1)
    stock = stock.drop("gain", 1)
    stock = stock.drop("loss", 1)
    stock = stock.drop("avgGain14", 1)
    stock = stock.drop("avgLoss14", 1)

    return stock

def run_rnn():

    # Importing the training set
    fore_step = 3
    time_step = 60
    train_samples = 1144
    test_samples = 20
    feature_size = 2
    dataset_train_pd = pd.read_csv('Google_Stock_Price.csv')
    dataset_train_pd = create_features_rsi(dataset_train_pd, 14)
    
    training_set = dataset_train_pd.iloc[:, 1:2].values
    #epochs = 180
    #batch_size = 32
    
    # Feature Scaling
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    
    # Creating a data structure with 60 timesteps and 1 output
    X_train = []
    y_train = []
    #for i in range(60, samples):
    
    #try two day ahead so it's 60~1256, shift two row
    for i in range(time_step, train_samples-fore_step):
        X_train.append(training_set_scaled[i-time_step:i, 0])
        y_train.append(training_set_scaled[i+fore_step, 0])
    y_train = np.array(y_train)
    
    #X_train = np.array(X_train)
    #V_train = np.array(V_train)
    
    mer_train = [None]*(len(X_train)+len(V_train))
    mer_train[::2] = X_train
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
    volume_total = pd.concat((dataset_train_pd['Volume'], dataset_test['Volume']), axis = 0)
    dataset_total = pd.concat((dataset_train_pd['Open'], dataset_test['Open']), axis = 0)
    
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

if __name__ == "__main__":
    run_rnn()
