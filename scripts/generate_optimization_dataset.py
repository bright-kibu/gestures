import pandas as pd
import numpy as np
import os

CSV_PATH = "hand_landmarks_dataset.csv"
OUTPUT_DIR = "optimization_data"
NUM_SAMPLES = 300  # You can increase this depending on dataset size

df = pd.read_csv(CSV_PATH)
X = df.drop("label", axis=1).values.astype(np.float32)

os.makedirs(OUTPUT_DIR, exist_ok=True)

for i in range(min(NUM_SAMPLES, len(X))):
    sample = X[i].reshape(1, 1, -1)  # Shape to match model input
    np.save(f"{OUTPUT_DIR}/input_{i:04d}.npy", sample)

print(f"Saved {min(NUM_SAMPLES, len(X))} optimization samples to `{OUTPUT_DIR}/`")