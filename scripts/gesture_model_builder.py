import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv("hand_landmarks_dataset.csv")
X = df.drop("label", axis=1).values.astype(np.float32)
y = df["label"].values

# Encode string labels into integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

# Print class mapping for verification
print("Class mapping:")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"  {i}: {class_name}")

# One-hot encode for softmax classifier
y_onehot = tf.keras.utils.to_categorical(y_encoded, num_classes)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42, stratify=y_onehot)

# Improved Hailo-compatible MLP model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input

model = tf.keras.Sequential([
    Input(shape=(X.shape[1],)),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    # Softmax activation for multi-class output
    Dense(num_classes, activation='softmax')  
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Choo Choo!
model.fit(X_train, y_train, validation_split=0.1, epochs=30, batch_size=32)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {acc:.4f}")

# Confusion matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Test the model with a few samples to verify outputs
print("\nVerifying model outputs:")
for i in range(min(5, len(X_test))):
    sample = X_test[i:i+1]
    pred = model.predict(sample)
    true_class = np.argmax(y_test[i])
    pred_class = np.argmax(pred)
    print(f"Sample {i}: True class: {label_encoder.classes_[true_class]}, Predicted: {label_encoder.classes_[pred_class]}")
    print(f"  Probabilities: {pred[0]}")

# Save the label encoder
import joblib
joblib.dump(label_encoder, "label_encoder.joblib")

# Save the Keras model
# model.save("gesture_model.h5")

# Convert to TFLite with explicit output format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Ensure the model outputs raw scores (not quantized)
converter.optimizations = []
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
# Explicitly set the output type
converter.inference_output_type = tf.float32
tflite_model = converter.convert()

with open("gesture_model.tflite", "wb") as f:
    f.write(tflite_model)

# Test the TFLite model to verify outputs
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nTFLite model input shape:", input_details[0]['shape'])
print("TFLite model output shape:", output_details[0]['shape'])

# Test with a few samples
print("\nVerifying TFLite model outputs:")
for i in range(min(5, len(X_test))):
    test_input = X_test[i:i+1].astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])
    
    true_class = np.argmax(y_test[i])
    pred_class = np.argmax(tflite_output[0])
    print(f"Sample {i}: True class: {label_encoder.classes_[true_class]}, Predicted: {label_encoder.classes_[pred_class]}")
    print(f"  Probabilities: {tflite_output[0]}")

print("\nSaved gesture_model.tflite and label_encoder.joblib")
print(f"Model outputs {num_classes} class probabilities")