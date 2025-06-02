from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

model = load_model('saved_model/mask_detector_model.h5')

test_gen = ImageDataGenerator(rescale=1./255)
test_data = test_gen.flow_from_directory(
    '../processed_dataset/test',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

predictions = model.predict(test_data)
predicted_classes = np.argmax(predictions, axis=1)

print("Classification Report:")
print(classification_report(test_data.classes, predicted_classes))
