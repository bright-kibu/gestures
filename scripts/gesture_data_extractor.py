import urllib.request
import math
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from tqdm import tqdm
import cv2
import numpy as np
import mediapipe as mp


DATA_DIR = "./images"
CLASS_NAMES = os.listdir(DATA_DIR)
IMG_EXTS = [".jpg", ".jpeg", ".png"]


# This will use the full handlandmark pipeline with palm
# detection.
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# Path to the MediaPipe hand landmark .task model
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task"

# Download the model if it does not exist
def download_model_if_needed(model_path, url):
    if not os.path.exists(model_path):
        print(f"Downloading {model_path} from {url}...")
        urllib.request.urlretrieve(url, model_path)
        print(f"Downloaded {model_path}.")

download_model_if_needed(MODEL_PATH, MODEL_URL)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
MODEL_PATH = "hand_landmarker.task"
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)
detector = HandLandmarker.create_from_options(options)

def process_image(args):
    class_name, img_path = args

    IMG_EXTS = [".jpg", ".jpeg", ".png"]
    if not any(img_path.lower().endswith(ext) for ext in IMG_EXTS):
        return None
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    if h != w:
        size = max(h, w)
        pad_top = (size - h) // 2
        pad_bottom = size - h - pad_top
        pad_left = (size - w) // 2
        pad_right = size - w - pad_left
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0,0,0])
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = detector.detect(mp_image)
    if not result.hand_landmarks or len(result.hand_landmarks) != 1:
        return None
    landmarks = result.hand_landmarks[0]
    row = []
    coords = []
    for lm in landmarks:
        row.extend([lm.x, lm.y, lm.z])
        coords.append([lm.x, lm.y, lm.z])

    # Feature engineered features
    coords_np = np.array(coords)
    # pairwise distances between all landmarks (upper triangle, excluding diagonal)
    dists = []
    for i in range(21):
        for j in range(i+1, 21):
            dists.append(np.linalg.norm(coords_np[i] - coords_np[j]))
    # angles between selected triplets (MCP, PIP, TIP for each finger)
    angles = []
    finger_indices = [
        (0, 2, 4),   # Thumb: wrist, MCP, tip
        (0, 5, 8),   # Index: wrist, MCP, tip
        (0, 9, 12),  # Middle
        (0, 13, 16), # Ring
        (0, 17, 20), # Pinky
    ]
    for a, b, c in finger_indices:
        ba = coords_np[a] - coords_np[b]
        bc = coords_np[c] - coords_np[b]
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        angles.append(angle)

    # Only include confident samples for each class (using just landmark coords):
    mcp_tip_pairs = [(2, 4), (5, 8), (9, 12), (13, 16), (17, 20)]
    mcp_tip_distances = []
    for mcp, tip in mcp_tip_pairs:
        mcp_pt = np.array(coords[mcp])
        tip_pt = np.array(coords[tip])
        dist = np.linalg.norm(tip_pt - mcp_pt)
        mcp_tip_distances.append(dist)
    # Use the extended finger count to increase the quality of
    # each label
    n_extended = sum(d > 0.07 for d in mcp_tip_distances)
    if class_name == 'point':
        index_is_longest = mcp_tip_distances[1] == max(mcp_tip_distances)
        if not (n_extended == 1 and index_is_longest):
            return None
    elif class_name == 'open_hand':
        if n_extended != 5:
            return None
    elif class_name == 'fist':
        if n_extended != 0:
            return None
    # Add features to row
    row.extend(dists)
    row.extend(angles)
    row.append(class_name)
    return row

def main():
    landmark_data = []
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(DATA_DIR, class_name)
        file_list = sorted(os.listdir(class_dir))
        for fname in tqdm(file_list, desc=f"Processing {class_name}"):
            img_path = os.path.join(class_dir, fname)
            row = process_image((class_name, img_path))
            if row is not None:
                landmark_data.append(row)


    # Landmark coordinate columns
    columns = [f"{coord}_{i}" for i in range(21) for coord in ("x", "y", "z")]
    # Feature engineered columns
    dist_columns = [f"dist_{i}_{j}" for i in range(21) for j in range(i+1, 21)]
    angle_columns = [f"angle_{name}" for name in ["thumb", "index", "middle", "ring", "pinky"]]
    columns.extend(dist_columns)
    columns.extend(angle_columns)
    columns.append("label")

    # Balance the dataset so each class has the same number of entries
    df = pd.DataFrame(landmark_data, columns=columns)
    class_counts = df['label'].value_counts()
    min_count = class_counts.min()
    balanced_df = (
        df.groupby('label', group_keys=False)
        .apply(lambda x: x.sample(min_count, random_state=42))
        .reset_index(drop=True)
    )
    balanced_df.to_csv("hand_landmarks_dataset.csv", index=False)

    print("Landmark data saved to hand_landmarks_dataset.csv")


if __name__ == "__main__":
    main()