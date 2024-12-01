import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# Load the dataset
df = pd.read_csv('indian_crop_recommendation_dataset.csv')

# Label encode categorical columns
le_soil = LabelEncoder()
df['Soil Type'] = le_soil.fit_transform(df['Soil Type'])

le_month = LabelEncoder()
df['Month'] = le_month.fit_transform(df['Month'])

le_crop = LabelEncoder()
df['Recommended Crop'] = le_crop.fit_transform(df['Recommended Crop'])

# Features and target variable
X = df[['Latitude', 'Longitude', 'Soil Type', 'Temperature (°C)', 'Humidity (%)', 'Month']]
y = df['Recommended Crop']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(le_crop.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save encoders, scaler, and model in a .pkl file
with open("preprocessing_and_model.pkl", "wb") as f:
    pickle.dump({
        "label_encoder_soil": le_soil,
        "label_encoder_month": le_month,
        "label_encoder_crop": le_crop,
        "scaler": scaler,
        "model": model
    }, f)
