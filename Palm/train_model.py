
import os
import cv2
import numpy as np
import pandas as pd
import joblib
import shutil
import json
from glob import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


BASE_DIR = os.getcwd()
CSV_PATH = os.path.join(BASE_DIR, "dataset.csv")
IMAGES_DIR = os.path.join(BASE_DIR, "dataset")   # images are directly here
MODEL_DIR = os.path.join(BASE_DIR, "model")      # will hold saved models
TRAINED_DIR = os.path.join(BASE_DIR, "trained_dataset")  # created and populated
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TRAINED_DIR, exist_ok=True)

CNN_SAVE = os.path.join(MODEL_DIR, "cnn_feature_extractor.h5")
RF_SAVE = os.path.join(MODEL_DIR, "rf_model.pkl")
LE_SAVE = os.path.join(MODEL_DIR, "label_encoder.pkl")
FEATURES_CSV = os.path.join(BASE_DIR, "cnn_features.csv")
LABELS_JSON = os.path.join(TRAINED_DIR, "labels.json")

IMG_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 30


def find_image_path(base_dir, image_id):
  
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        p = os.path.join(base_dir, f"{image_id}{ext}")
        if os.path.exists(p):
            return p
 
    matches = glob(os.path.join(base_dir, f"{image_id}.*"))
    return matches[0] if matches else None

def load_and_preprocess(path, img_size=IMG_SIZE):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype("float32") / 255.0
    return img


print("[INFO] Loading CSV:", CSV_PATH)
df = pd.read_csv(CSV_PATH)

images = []
labels = []
image_ids = []
missing = []

for idx, row in df.iterrows():
    img_id = str(row["image_id"])
    p = find_image_path(IMAGES_DIR, img_id)
    if p is None:
        missing.append(img_id)
        continue
    img = load_and_preprocess(p)
    if img is None:
        missing.append(img_id)
        continue
    images.append(img)
    labels.append(row["dominant_line"])
    image_ids.append(img_id)

if missing:
    print("[WARN] Missing images for ids (will be skipped):", missing)
print(f"[INFO] Loaded {len(images)} images.")
if len(images) == 0:
    raise SystemExit("[ERROR] No images found. Fix dataset paths or filenames.")

X = np.array(images)
y = np.array(labels)


le = LabelEncoder()
y_enc = le.fit_transform(y)
joblib.dump(le, LE_SAVE)
print(f"[INFO] Saved LabelEncoder -> {LE_SAVE}")
print("[INFO] Classes:", list(le.classes_))
num_classes = len(le.classes_)

# ---------- BUILD SMALL CNN ----------
def build_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    features = layers.Dense(128, activation="relu", name="feature_vector")(x)
    out = layers.Dense(num_classes, activation="softmax")(features)
    model = models.Model(inputs, out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

model = build_cnn()
model.summary()

# ---------- TRAIN CNN ----------
if num_classes < 2:
    raise SystemExit("[ERROR] Need at least 2 classes to train.")
X_train, X_val, y_train, y_val = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

es = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
ckpt = ModelCheckpoint(os.path.join(MODEL_DIR, "best_cnn.h5"), monitor="val_loss", save_best_only=True)

print("[INFO] Training CNN...")
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[es, ckpt],
    verbose=1
)

# ---------- FEATURE EXTRACTOR ----------
print("[INFO] Building feature extractor from CNN (layer 'feature_vector').")
feature_layer_name = "feature_vector"
feature_model = models.Model(inputs=model.input, outputs=model.get_layer(feature_layer_name).output)
feature_model.save(CNN_SAVE)
print(f"[INFO] Saved CNN feature extractor -> {CNN_SAVE}")

# ---------- EXTRACT FEATURES FOR ALL IMAGES ----------
print("[INFO] Extracting CNN features for all images...")
features = feature_model.predict(X, verbose=1)
print("[INFO] Features shape:", features.shape)

df_feat = pd.DataFrame(features)
df_feat["image_id"] = image_ids
df_feat["label"] = y
df_feat.to_csv(FEATURES_CSV, index=False)
print(f"[INFO] Saved features -> {FEATURES_CSV}")

# ---------- TRAIN RANDOM FOREST ----------
print("[INFO] Training RandomForest on CNN features...")
X_rf = features
y_rf = y
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42, stratify=y_rf)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_rf, y_train_rf)

y_pred = rf.predict(X_test_rf)
acc = accuracy_score(y_test_rf, y_pred)
print(f"[RESULT] RandomForest accuracy: {acc*100:.2f}%")
print(classification_report(y_test_rf, y_pred))

joblib.dump(rf, RF_SAVE)
print(f"[INFO] Saved RandomForest -> {RF_SAVE}")

# ---------- CREATE trained_dataset/ organized by class ----------
print("[INFO] Creating trained dataset folder:", TRAINED_DIR)
# create class subfolders and copy original images
classes = sorted(list(set(y)))
labels_map = {}
for cls in classes:
    cls_dir = os.path.join(TRAINED_DIR, cls)
    os.makedirs(cls_dir, exist_ok=True)
    labels_map[cls] = []

# copy images into their class folders
for img_id, cls in zip(image_ids, y):
    src = find_image_path(IMAGES_DIR, img_id)
    if src is None:
        continue
    ext = os.path.splitext(src)[1]
    dst = os.path.join(TRAINED_DIR, cls, f"{img_id}{ext}")
    shutil.copy(src, dst)
    labels_map[cls].append(os.path.basename(dst))

# save labels.json
with open(LABELS_JSON, "w") as f:
    json.dump(labels_map, f, indent=2)

print(f"[INFO] Trained dataset created at: {TRAINED_DIR}")
print(f"[INFO] Labels JSON saved -> {LABELS_JSON}")

print("[DONE] Training + export pipeline finished.")
