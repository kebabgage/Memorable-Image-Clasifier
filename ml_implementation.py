import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.impute import SimpleImputer
from tensorflow.keras.callbacks import EarlyStopping
from libtlda.flda import FeatureLevelDomainAdaptiveClassifier
from matplotlib import pyplot

# Comment out the model you wish to employ
def create_model():
    """
    Creates a keras.Sequential() model, compiles it and returns it.

    Low functionality, just comment-in the model you want to use.
    """

    # Model 1
    new_model = keras.Sequential(
        [
            layers.Dense(4, activation="sigmoid"),
            layers.Dense(1, activation="sigmoid")
        ]
    )

    # # Model 2
    # new_model = keras.Sequential(
    #     [
    #         layers.Dense(4, activation="relu"),
    #         layers.Dense(1, activation="sigmoid")
    #     ]
    # )

    # # Model 3
    # new_model = keras.Sequential(
    #     [
    #         layers.Dense(8, activation="relu"),
    #         layers.Dense(4, activation="relu"),
    #         layers.Dense(1, activation="sigmoid")
    #     ]
    # )

    # # Model 4
    # new_model = keras.Sequential(
    #     [
    #         layers.Dense(16, activation="relu"),
    #         layers.Dense(8, activation="relu"),
    #         layers.Dense(4, activation="relu"),
    #         layers.Dense(1, activation="sigmoid")
    #     ]
    # )

    # # Model 5
    # new_model = keras.Sequential(
    #     [
    #         layers.Dense(500, activation="relu"),  #
    #         layers.Dense(250, activation="relu"),
    #         layers.Dense(250, activation="relu"),  #
    #         layers.Dense(1, activation="sigmoid")
    #     ]
    # )
    # Compile model
    new_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

    return new_model


# DATA LOADING ------------------------------------------------------------------------------------
training = pd.read_csv("training.csv")
training = training.set_index("ID", drop=True)

# y_train, X_train are the training datasets
y_train = training.pop('prediction').to_numpy()
X_train = training.to_numpy()

# X_test is the testing dataset
X_test_final = pd.read_csv("testing.csv").set_index("ID", drop=True).to_numpy()

# Additional Data loading (has NaNs)
add_data = pd.read_csv("additional_training.csv").set_index("ID", drop=True)
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
add_data_trans = imputer.fit_transform(add_data)

# Split the data column with the output
y_data_add = add_data_trans[:, 4608]
X_data_add = add_data_trans[:, :4608]

y_train = np.concatenate((y_train, y_data_add))  # Concat all the y data
X_train = np.concatenate((X_train, X_data_add))  # Concatenate all the X data

# Train the scaler, standardize all features to have mean=0 and unit variance
sc = StandardScaler()
sc.fit(X_train)

X_train = sc.transform(X_train)  # Apply the scaler to the X training data
X_final = sc.transform(X_test_final)  # Apply the scaler to the X final test data


# CROSS VALIDATION --------------------------------------------------------------------------------
random_seed = 7
# Create CV instance
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_seed)
# Lists to hold the accuracies and losses for each cross validation
cvscores = []
cvsloss = []
histories = []
scores = []
for train, test in kfold.split(X_train, y_train):
    # Set Early Stopping Monitor so the model stops training when it won't improve anymore
    early_stopping_monitor = EarlyStopping(patience=15)

    # Create a Sequential classifier
    model = create_model()
    # Fit the model
    histories.append(model.fit(X_train[train], y_train[train], epochs=150, batch_size=10, 
                                verbose=0,
                               callbacks=[early_stopping_monitor],
                               validation_data=(X_train[test], y_train[test])
                               )
                     )
    # Evaluate the model
    scores = model.evaluate(X_train[test], y_train[test], verbose=0)
    cvscores.append(scores[1] * 100)
    cvsloss.append(scores[0] * 100)

# Ouput the metrics of each cross validation
print(cvscores)
print(cvsloss)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

# Create a Sequential classifier -- same that was used for cross validation 
model = create_model()
history = model.fit(X_train, y_train, epochs=150, batch_size=10, verbose=0)


# PREDICTION --------------------------------------------------------------------------------------
y_predict = model.predict(X_final) # Predict the classses given current trained model 

# Assign the binary classes to outputs 
y_predict[y_predict < 0.5] = 0 
y_predict[y_predict > 0.5] = 1

class0 = len(y_predict[y_predict == 0])
class1 = len(y_predict[y_predict == 1])
classtotal = class0 + class1

# Ouput the ratio of classes 
print("0 class: ", class0 / classtotal)
print("1 class: ", class1 / classtotal)

actual_class1 = 0.3848
actual_class0 = 0.6152

# Save the data 
y_pred_df = pd.DataFrame(y_predict).astype(int)
y_pred_df.index += 1
pd.DataFrame(y_pred_df).to_csv("predictions_data.csv", header=["prediction"], index=True, index_label="ID")
