import os
import flwr as fl
import tensorflow as tf

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Re-import randomly splitted data frames (equal sources for various tests)
import pandas as pd
df_part1 = pd.read_csv('df_part2.csv')

#df_part1.head()

# Drop unneccesary colums
df_part1 = df_part1.drop(['Failure Type', 'Target', 'Type', 'Product ID', 'UDI', 'Unnamed: 0'], axis=1)
#df_part1.head()

# Split dataset into train and testdata
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df_part1, train_size=0.8)
#df_train.shape

# Define feature and target names
feature_names = ['Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]']
target_names = ['Failure Type Cat']

# Convert features for training to tensor
features_train = df_train[feature_names]
tf.convert_to_tensor(features_train)

# Convert features for testing to tensor
features_test = df_test[feature_names]
tf.convert_to_tensor(features_test)

# Convert target for training to tensor
target_train = df_train[target_names]
tf.convert_to_tensor(target_train)

# Convert target for testing to tensor
target_test = df_test[target_names]
tf.convert_to_tensor(target_test)

# Create a model and compile it
from tensorflow.keras.layers import Dense,BatchNormalization,Dropout,Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.regularizers import l2

def create_model():
    # 5 Features
    inputs=Input(5)
    # Layer with 500 neurons, rectified linear unit is simple and effective dealing with non-linearities,
    # kernel regularizer to prevent overfitting (good on training data, problem with new data)
    x=Dense(500,activation='relu',kernel_regularizer=l2(0.01))(inputs)
    x=Dense(250,activation='relu')(x)
    # Batch normalization to stabilize and speed up training
    x=BatchNormalization()(x)
    x=Dense(125,activation='relu')(x)
    # Softmax activation for multiclass (1, 2, 3...) for 1 output node
    x=Dense(1,activation='softmax')(x)
    model=Model(inputs,x)
    # CategoricalCrossentropy for multiclass, metrics accuracy, Adam = optimization algorithm providing:
    # adaptive learning, use of momentum and bias correction
    model.compile(loss=CategoricalCrossentropy(),metrics=['accuracy'],optimizer=Adam(0.01))
    return model

model=create_model()

# Define Flower client
class FLClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(features_train, target_train, epochs=20, batch_size=100)
        model.save("model1a.keras")
        return model.get_weights(), len(features_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(features_test, target_test)
        return loss, len(features_test), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FLClient())

#%%
