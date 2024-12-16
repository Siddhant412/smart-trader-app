import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras_tuner.tuners import Hyperband
from sklearn.metrics import mean_squared_error

def create_sequences(data, sequence_length):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length - 5):  # -1 to ensure room for next day's target
        seq = data[i:i + sequence_length]  # Features: past `sequence_length` days
        # Use iloc to select the row and columns by their position
        #target = data.iloc[i + sequence_length, :4].values  # Target: Next day's Open, High, Low, Close
        target1 = data.iloc[i + sequence_length, :4].values  # Target: Next day's Open, High, Low, Close
        target2 = data.iloc[i + sequence_length + 1, :4].values
        target3 = data.iloc[i + sequence_length + 2, :4].values
        target4 = data.iloc[i + sequence_length + 3, :4].values
        target5 = data.iloc[i + sequence_length + 4, :4].values
        target = np.concatenate([target1, target2, target3, target4, target5])  # Concatenate all targets
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)


nvda = yf.Ticker("NVDA")
nvda_data = nvda.history(period="5y")
nvda_data = nvda_data.dropna()
nvda_data = nvda_data[['Open', 'High', 'Low', 'Close', 'Volume']]

print("Full data shape for NVDA:", nvda_data.shape)

train_size = int(len(nvda_data) * 0.8)
val_size = int(len(nvda_data) * 0.1)
test_size = len(nvda_data) - train_size - val_size
print("Train size:", train_size)
print("Validation size:", val_size)
print("Test size:", test_size)

nvda_train = nvda_data[:train_size]
nvda_val = nvda_data[train_size:train_size + val_size]
nvda_test = nvda_data[train_size + val_size:]

scaler_nvda = MinMaxScaler()
scaler_nvda.fit(nvda_train)  # Fit scaler only on the training set
with open('nvda_scaler.pkl', 'wb') as f:
    pickle.dump(scaler_nvda, f)

nvda_train_scaled = pd.DataFrame(scaler_nvda.transform(nvda_train), columns=nvda_data.columns)
nvda_val_scaled = pd.DataFrame(scaler_nvda.transform(nvda_val), columns=nvda_data.columns)
nvda_test_scaled = pd.DataFrame(scaler_nvda.transform(nvda_test), columns=nvda_data.columns)

# Create sequences for NVDA
nvda_X_train, nvda_y_train = create_sequences(nvda_train_scaled, sequence_length=60)
nvda_X_val, nvda_y_val = create_sequences(nvda_val_scaled, sequence_length=60)
nvda_X_test, nvda_y_test = create_sequences(nvda_test_scaled, sequence_length=60)

print("Training sequence shape:", nvda_X_train.shape)  # (samples, sequence_length, features)
print("Training target shape:", nvda_y_train.shape)  # (samples, 4)
print("Validation sequence shape:", nvda_X_val.shape)  # (samples, sequence_length, features)
print("Validation target shape:", nvda_y_val.shape)  # (samples, 4)
print("Testing sequence shape:", nvda_X_test.shape)  # (samples, sequence_length, features)
print("Testing target shape:", nvda_y_test.shape)  # (samples, 4)

sequence_length = 60

# Define the model builder function
def build_model(hp):
    model = Sequential()

    # Add the first LSTM layer
    model.add(LSTM(
        units=hp.Int('units', min_value=32, max_value=128, step=16),
        activation='relu',
        return_sequences=True,  # Always return sequences for intermediate layers
        input_shape=(sequence_length, nvda_train_scaled.shape[1])
    ))

    # Add Dropout after the first LSTM layer
    model.add(Dropout(hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)))

    # Add additional LSTM layers based on num_layers
    num_layers = hp.Int('num_layers', min_value=1, max_value=3)
    for i in range(num_layers):
        model.add(LSTM(
            units=hp.Int(f'units_{i}', min_value=32, max_value=128, step=16),
            activation='relu',
            return_sequences=i < num_layers-1  # Return sequences for all but the last layer
        ))
        # Add Dropout after each additional LSTM layer
        model.add(Dropout(hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1)))

    # Add Dense output layer
    model.add(Dense(20))

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        ),
        loss='mean_squared_error'
    )

    return model

# Initialize the Hyperband tuner
tuner = Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=20,
    factor=3,  # Halving rate
    directory='hyperband_tuning1',
    project_name='lstm_hyperband'
)

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Run the hyperparameter search
tuner.search(
    nvda_X_train, nvda_y_train,
    validation_data=(nvda_X_val, nvda_y_val),
    epochs=20,
    callbacks=[early_stopping],
    batch_size=32
)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best hyperparameters: {best_hps.values}")

best_model = tuner.hypermodel.build(best_hps)

# Train the best model on train + validation data
history = best_model.fit(
    np.concatenate((nvda_X_train, nvda_X_val), axis=0),
    np.concatenate((nvda_y_train, nvda_y_val), axis=0),
    epochs=20,
    batch_size=32,
    callbacks=[early_stopping]
)

# Evaluate the model on the test set
test_loss = best_model.evaluate(nvda_X_test, nvda_y_test)
print(f"Test Loss: {test_loss}")

# # Define LSTM model
# model = Sequential([
#     LSTM(128, activation='relu', return_sequences=True, input_shape=(sequence_length, nvda_train_scaled.shape[1])),
#     LSTM(64, activation='relu', return_sequences=True),
#     LSTM(64, activation='relu'),
#     Dense(20)  # Output layer with 4 neurons (Open, High, Low, Close)
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error')

# # Train the model
# history = model.fit(
#     nvda_X_train, nvda_y_train,
#     validation_data=(nvda_X_val, nvda_y_val),
#     epochs=20,
#     batch_size=32
# )

# # Evaluate on test set
# test_loss = model.evaluate(nvda_X_test, nvda_y_test)
# print("Test Loss:", test_loss)

best_model.save('nvda_model.keras')