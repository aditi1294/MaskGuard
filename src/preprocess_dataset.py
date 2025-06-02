import os
import cv2
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

IMG_SIZE = 128
SOURCE_DIR = '../dataset'
OUTPUT_DIR = '../processed_dataset'

CATEGORIES = ['with_mask', 'without_mask'] 
def preprocess_and_split():
    data = []
    labels = []

    for category in CATEGORIES:
        path = os.path.join(SOURCE_DIR, category)
        label = CATEGORIES.index(category)

        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                data.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {img_name}: {e}")

    data = np.array(data)
    labels = np.array(labels)

    # Train / Validation / Test Split (70/20/10)
    X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.3, stratify=labels, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, stratify=y_temp, random_state=42)

    save_split(X_train, y_train, 'train')
    save_split(X_val, y_val, 'val')
    save_split(X_test, y_test, 'test')

def save_split(images, labels, split_name):
    for i, (img, label) in enumerate(zip(images, labels)):
        label_name = CATEGORIES[label]
        dir_path = os.path.join(OUTPUT_DIR, split_name, label_name)
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, f"{split_name}_{label_name}_{i}.jpg")
        cv2.imwrite(file_path, img)

if __name__ == "__main__":
    preprocess_and_split()
