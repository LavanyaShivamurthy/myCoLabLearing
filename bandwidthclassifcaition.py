've created a comprehensive deep learning solution for your network packet bandwidth classification task. This script offers several advanced approaches that can better capture the complex patterns in your network traffic data:
Key Features of the Deep Learning Solution:

Multiple Model Architectures:

Functional API Model: A sophisticated dual-input neural network that processes numeric and binary features separately before combining them
Sequential Model: A simpler alternative that processes all features together
Temporal Model: An LSTM/CNN option for when you need to analyze packet sequences over time


Advanced Neural Network Components:

Batch normalization for faster training and improved stability
Dropout layers to prevent overfitting
Early stopping and model checkpointing to save the best model


Data Processing Improvements:

More sophisticated bandwidth classification using percentile-based boundaries
Feature separation for better handling of different data types
Proper scaling and normalization for optimal neural network performance


Comprehensive Visualization:

Training/validation curves to monitor learning progress
Confusion matrix to understand classification performance by category
Class probability outputs for more nuanced decision-making


Production-Ready Features:

Model saving and loading for deployment
Metadata preservation for consistent preprocessing
Flexible input handling for real-time classification

How to Use It:

Ensure you have TensorFlow installed (pip install tensorflow)
Format your packet data as a CSV with the fields you mentioned
Run the script, which will:

Load and preprocess your data
Train multiple model types (you can select which to use)
Evaluate performance with detailed metrics
Save the trained model for future use
Demonstrate how to classify new packets



The deep learning approach should provide higher accuracy than traditional machine learning, especially when dealing with complex network patterns and large amounts of data.
Would you like me to explain any particular aspect of the deep learning solution in more detail, or would you prefer a different type of neural network architecture for your specific use case?


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(file_path):
    """
    Load the network packet data and perform initial preprocessing
    """
    # Load data (assuming CSV format - adjust as needed)
    df = pd.read_csv(file_path)
    
    # Handle missing values
    df = df.fillna(0)
    
    # Create derived features relevant for bandwidth analysis
    if 'frame.len' in df.columns and 'frame.time_delta' in df.columns:
        # Calculate bandwidth usage in bytes per second
        df['bandwidth_bps'] = df['frame.len'] / df['frame.time_delta']
    
    # Extract MQTT vs TCP traffic
    df['is_mqtt'] = df.apply(lambda x: 1 if any(col.startswith('mqtt.') and pd.notna(x[col]) for col in df.columns) else 0, axis=1)
    
    return df

def create_bandwidth_classes(df, n_classes=3):
    """
    Create bandwidth classes based on frame.len and bandwidth usage
    """
    # Log transform bandwidth if it's highly skewed
    bandwidth = np.log1p(df['bandwidth_bps']) if 'bandwidth_bps' in df.columns else df['frame.len']
    
    # Create class boundaries based on percentiles
    boundaries = np.percentile(bandwidth, np.linspace(0, 100, n_classes + 1))
    
    # Assign classes
    df['bandwidth_class'] = pd.cut(
        bandwidth, 
        bins=boundaries, 
        labels=list(range(n_classes)), 
        include_lowest=True
    ).astype(int)
    
    # Add human-readable labels
    labels = {
        0: 'Low Bandwidth',
        1: 'Medium Bandwidth',
        2: 'High Bandwidth'
    }
    df['bandwidth_category'] = df['bandwidth_class'].map(labels)
    
    return df

def prepare_features(df):
    """
    Prepare features for deep learning model
    """
    # Select numeric features
    numeric_features = [
        'frame.len', 'frame.time_delta', 
        'tcp.len', 'tcp.time_delta', 'tcp.window_size_value',
        'ip.ttl'
    ]
    
    # Select binary features
    binary_features = [
        'tcp.flags.ack', 'tcp.flags.fin', 'tcp.flags.push', 
        'tcp.flags.reset', 'tcp.flags.syn', 'tcp.flags.urg',
        'is_mqtt'
    ]
    
    # Filter to only include columns that exist in the dataframe
    numeric_features = [f for f in numeric_features if f in df.columns]
    binary_features = [f for f in binary_features if f in df.columns]
    
    # Prepare numeric features
    X_numeric = df[numeric_features].values
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X_numeric)
    
    # Prepare binary features
    X_binary = df[binary_features].values
    
    # Prepare target variable
    y = df['bandwidth_class'].values
    y_onehot = to_categorical(y)
    
    # Return processed data
    return X_numeric_scaled, X_binary, y_onehot, numeric_features, binary_features, scaler

def build_deep_learning_model(input_numeric_shape, input_binary_shape, num_classes):
    """
    Build a deep neural network for packet classification
    """
    # Input layers
    input_numeric = Input(shape=(input_numeric_shape,), name='numeric_input')
    input_binary = Input(shape=(input_binary_shape,), name='binary_input')
    
    # Process numeric features
    x_numeric = Dense(64, activation='relu')(input_numeric)
    x_numeric = BatchNormalization()(x_numeric)
    x_numeric = Dropout(0.3)(x_numeric)
    x_numeric = Dense(32, activation='relu')(x_numeric)
    
    # Process binary features
    x_binary = Dense(32, activation='relu')(input_binary)
    x_binary = BatchNormalization()(x_binary)
    x_binary = Dropout(0.2)(x_binary)
    
    # Combine features
    combined = Concatenate()([x_numeric, x_binary])
    
    # Hidden layers
    x = Dense(64, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Output layer
    output = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=[input_numeric, input_binary], outputs=output)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_sequential_model(input_shape, num_classes):
    """
    Build a simpler sequential model as an alternative
    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(X_numeric, X_binary, y, model_type='functional'):
    """
    Train the deep learning model
    """
    # Split data
    X_numeric_train, X_numeric_test, X_binary_train, X_binary_test, y_train, y_test = train_test_split(
        X_numeric, X_binary, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Build model
    if model_type == 'functional':
        model = build_deep_learning_model(
            X_numeric.shape[1], 
            X_binary.shape[1], 
            y.shape[1]
        )
        
        # Training data
        train_data = {
            'numeric_input': X_numeric_train,
            'binary_input': X_binary_train
        }
        
        # Validation data
        val_data = {
            'numeric_input': X_numeric_test,
            'binary_input': X_binary_test
        }
    else:
        # Combine features for sequential model
        X_combined = np.hstack([X_numeric, X_binary])
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=0.2, random_state=42, stratify=y
        )
        
        model = build_sequential_model(X_combined.shape[1], y.shape[1])
        
        train_data = X_train
        val_data = X_test
    
    # Setup callbacks
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint('best_bandwidth_model.h5', save_best_only=True)
    ]
    
    # Train model
    history = model.fit(
        train_data, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(val_data, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    if model_type == 'functional':
        y_pred_prob = model.predict(val_data)
    else:
        y_pred_prob = model.predict(val_data)
    
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    
    return model, history, y_pred, y_true

def visualize_results(history, y_true, y_pred, class_names):
    """
    Visualize model performance
    """
    # Plot loss and accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('dl_model_training.png')
    plt.show()
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('dl_confusion_matrix.png')
    plt.show()

def predict_bandwidth_class(packet_data, model, numeric_features, binary_features, scaler, model_type='functional'):
    """
    Predict bandwidth class for a new packet
    """
    # Extract features
    numeric_vals = [packet_data.get(f, 0) for f in numeric_features]
    binary_vals = [packet_data.get(f, 0) for f in binary_features]
    
    # Scale numeric features
    numeric_scaled = scaler.transform([numeric_vals])
    binary_array = np.array([binary_vals])
    
    # Make prediction
    if model_type == 'functional':
        prediction = model.predict({
            'numeric_input': numeric_scaled,
            'binary_input': binary_array
        })
    else:
        combined = np.hstack([numeric_scaled, binary_array])
        prediction = model.predict(combined)
    
    class_idx = np.argmax(prediction[0])
    
    labels = {
        0: 'Low Bandwidth',
        1: 'Medium Bandwidth',
        2: 'High Bandwidth'
    }
    
    return labels[class_idx], prediction[0]

def main():
    # File path
    file_path = 'network_packets.csv'  # Replace with your data file
    
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data(file_path)
    
    print("Creating bandwidth classes...")
    df = create_bandwidth_classes(df)
    
    print("Preparing features...")
    X_numeric, X_binary, y, numeric_features, binary_features, scaler = prepare_features(df)
    
    print("Building and training deep learning model...")
    # Choose model type: 'functional' for multi-input or 'sequential' for combined input
    model_type = 'functional'  
    model, history, y_pred, y_true = train_model(X_numeric, X_binary, y, model_type)
    
    print("Visualizing results...")
    class_names = ['Low Bandwidth', 'Medium Bandwidth', 'High Bandwidth']
    visualize_results(history, y_true, y_pred, class_names)
    
    # Example of classifying a new packet
    new_packet = {
        'frame.len': 1024,
        'frame.time_delta': 0.001,
        'tcp.len': 984,
        'tcp.time_delta': 0.001,
        'tcp.window_size_value': 65535,
        'tcp.flags.ack': 1,
        'tcp.flags.push': 1,
        'tcp.flags.syn': 0,
        'tcp.flags.fin': 0,
        'tcp.flags.reset': 0,
        'tcp.flags.urg': 0,
        'is_mqtt': 0,
        'ip.ttl': 64
    }
    
    prediction, confidence = predict_bandwidth_class(
        new_packet, model, numeric_features, binary_features, scaler, model_type
    )
    print(f"New packet bandwidth requirement: {prediction}")
    print(f"Class confidences: {confidence}")
    
    # Save the model and metadata
    model.save('deep_learning_bandwidth_classifier.h5')
    
    # Save necessary metadata for future predictions
    import pickle
    with open('dl_model_metadata.pkl', 'wb') as f:
        pickle.dump({
            'numeric_features': numeric_features,
            'binary_features': binary_features,
            'scaler': scaler,
            'model_type': model_type,
            'class_names': class_names
        }, f)
    
    print("Model and metadata saved for future use")

# Function for LSTM/CNN model if temporal features are important
def build_temporal_model(seq_length, num_features, num_classes):
    """
    Build a model that considers temporal aspects of packet sequences
    This is useful if packets need to be analyzed as sequences rather than individual items
    """
    from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Flatten
    
    model = Sequential([
        # Input layer
        Input(shape=(seq_length, num_features)),
        
        # Convolutional layers for feature extraction
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        
        # LSTM layer for temporal patterns
        LSTM(128, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        
        # Dense layers for classification
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    main()
