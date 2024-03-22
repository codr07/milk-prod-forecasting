
<img src="https://github.com/codr07/milk-prod-forecasting/blob/e02226aee3e1f203e8bc3d0126c555bdb8208afb/PRD%20Header.jpg" width="100%" />

# The Whole process is stated here :
<hr/>

### 🔏The Platform Used in here 
##

[![Kaggle](https://img.shields.io/badge/Kaggle-blue?logo=kaggle&logoColor=orange)](https://www.kaggle.com)

##

<details>
  <summary><h4>Click to begine the process</h4></summary>
<details>
  <summary> <h3>Step 1: Importing Libraries</h3> </summary>
  
## Run this commands for importing all the libraries we need

```matlab
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
```
</details>
  
<details>
  <summary><h3> Step 2: Import Data </h3></summary>
  
## Now, we would import the data into the code segment. The data source must be a reliable one, in order to avoid abnormal results

```matlab
df=pd.read_csv("monthly-milk-production.csv",index_col='Month',parse_dates=True)
df.index.freq='MS'
df.head(168)
```
<details>
  <summary> Should give an output as follows : </summary>

<img src="https://github.com/codr07/milk-prod-forecasting/blob/aff75170fc80f01cfc0029b8f3f11337c00d09e8/Sample%20data%20output.png"/>
</details>
</details>


<details>
  <summary><h3> Step 3: Plot Data</h3> </summary>
  
## Now, we would plot the data we have imported

```matlab
df.plot(figsize=(12,6))
```
<details>
  <summary> <h3> Output</h3>  </summary>

<img src="https://github.com/codr07/milk-prod-forecasting/blob/aff75170fc80f01cfc0029b8f3f11337c00d09e8/main%20data%20plot.png" />

</details>
</details>

<details>
  <summary><h3>Step 4: Decomposition of Data</h3> </summary>
  
## Now, we would find the seasonal decompose. Seasonal decomposition methods can be useful for various purposes, including forecasting, anomaly detection, and understanding the underlying dynamics of time series data. One common technique for seasonal decomposition is the seasonal decomposition of time series (STL) algorithm, which separates these components effectively.

```matlab
results = seasonal_decompose(df['Milk Production'])
results.plot();
```
<details>
  <summary>Possible Output </summary>

<img src="https://github.com/codr07/milk-prod-forecasting/blob/aff75170fc80f01cfc0029b8f3f11337c00d09e8/sasonal%20decompose.png" />

</details>
</details>


<details>
  <summary><h3>Step 5: Train and Test Set</h3></summary>
  
## Now, we would divide the data into train and test sets. This step is called Feature Scaling.

```matlab
print("total values :",len(df))
train = df.iloc[:156]
test = df.iloc[156:]
```
</details>

<details>
  <summary><h3>Step 6: Normalising </h3></summary>
  
## Now, we would Normalize the data. Normalizing the data is very important, as it reduces data abnormality and eliminates redundant data.

```matlab
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)
```
```matlab
scaled_train[:10]
```
```matlab
n_input = 12
n_features = 1
generator = TimeseriesGenerator(scaled_train,
                                scaled_train,
                                length=n_input,
                                batch_size=1)
```
```matlab
X,y = generator[0]
print(f'Given the Array: \n{X.flatten()}')
print(f'Predict this y: \n {y}')
```
### Shaping X
```matlab
X.shape
```
</details>

<details>
  <summary><h3>Step 7: Data Modeling </h3></summary>
  
## Here, we declare the LSTM. LSTM is a kind of RNN, that makes the forecasting quicker than usual.

```matlab
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
```
### Summarising 

```matlab
model.summary()
```
</details>

<details>
  <summary><h3>Step 8: Epoching</h3></summary>
  
## Now, we bwgin the LSTM process
### This takes time

```matlab
model.fit(generator,epochs=50)
```
</details>

<details>
  <summary><h3>Step 9: Visualizing Epoch</h3></summary>
  
## Now, we would try to visualize the loss per epoch with a graph plot

```matlab
loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)),loss_per_epoch)
```
<details>
  <summary>Possible Output</summary>
  
<img src="https://github.com/codr07/milk-prod-forecasting/blob/aff75170fc80f01cfc0029b8f3f11337c00d09e8/loss%20per%20epoch.png" />

</details>
</details>

<details>
  <summary><h3>Step 10: Train and Test </h3></summary>
  
## Now, we would so train and test the sets

```matlab
last_train_batch = scaled_train[-12:]
```
- Next
```matlab
last_train_batch = last_train_batch.reshape((1, n_input, n_features))
```
- Next
- 
```matlab
model.predict(last_train_batch)
```
- [ Output : ]

```
1/1 [==============================] - 0s 348ms/step
array([[0.6611159]], dtype=float32)
```

- Next

```matlab
scaled_test[0]
```
- [ Output : ]

```
array([0.67548077])
```
</details>

<details>
  <summary><h3>Step 11: Prediction</h3></summary>
  
## Initializing the prediction

```matlab
test_predictions = []

first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test)):

    # get the prediction value for the first batch
    current_pred = model.predict(current_batch)[0]

    # append the prediction into the array
    test_predictions.append(current_pred)

    # use the prediction to update the batch and remove the first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
```
- [ Output : (This is a random process may differ to get the output) ]
```
1/1 [==============================] - 0s 25ms/step
1/1 [==============================] - 0s 38ms/step
1/1 [==============================] - 0s 26ms/step
1/1 [==============================] - 0s 32ms/step
1/1 [==============================] - 0s 25ms/step
1/1 [==============================] - 0s 24ms/step
1/1 [==============================] - 0s 26ms/step
1/1 [==============================] - 0s 37ms/step
1/1 [==============================] - 0s 38ms/step
1/1 [==============================] - 0s 31ms/step
1/1 [==============================] - 0s 31ms/step
1/1 [==============================] - 0s 35ms/step
```
- Next : Print Predictions

```matlab
test_predictions
```
- [ Output : (This is a random process may differ to get the output) ]

```
[array([0.6611159], dtype=float32),
 array([0.6293224], dtype=float32),
 array([0.8113305], dtype=float32),
 array([0.87288034], dtype=float32),
 array([0.97425973], dtype=float32),
 array([0.95229566], dtype=float32),
 array([0.8819771], dtype=float32),
 array([0.79158455], dtype=float32),
 array([0.6838793], dtype=float32),
 array([0.64869756], dtype=float32),
 array([0.5978747], dtype=float32),
 array([0.640924], dtype=float32)]
```
- Next

```matlab
test.head()
```
- Next
```matlab
true_predictions = scaler.inverse_transform(test_predictions)
```
- Next
```matlab
true_predictions = scaler.inverse_transform(test_predictions)
```
- [ Output : ]
```
<ipython-input-66-920b79c3c314>:1: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  test['Predictions'] = true_predictions
```
</details>

<details>
  <summary><h3>Step 12: Plotting the Predictions</h3></summary>
  
## Now, we would plot the Predicted Data against the Original Data

```matlab
test.plot(figsize=(14,5))
```
<details>
  <summary>Possible Output</summary>
  
<img src="https://github.com/codr07/milk-prod-forecasting/blob/aff75170fc80f01cfc0029b8f3f11337c00d09e8/original%20data%20vs%20predicted%20data.png" />


</details>
</details>

<details>
  <summary><h3>Step 13: Calculating RMSE </h3></summary>
  
## Now, we would calculate Root Mean Square Deviation(RMSE)

```matlab
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse=sqrt(mean_squared_error(test['Milk Production'],test['Predictions']))
print(rmse)
```

- [ Output : (This is a random process may differ to get the output) ]

```
17.465366039114766
```

</details>






</details>

<img src="https://github.com/codr07/milk-prod-forecasting/blob/81922cd1e36cdd86e380ecbc17fd5d85a63b31f3/bottom_header.svg" width="100%" />
