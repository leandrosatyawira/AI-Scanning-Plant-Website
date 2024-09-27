import cv2
import tensorflow as tf
import numpy as np

# Load your trained model
model = tf.keras.models.load_model('plant.h5')

# Define a function to make predictions
def predict_leaf_disease(img, model):
    x = cv2.resize(img, (256, 256))
    x = x / 255.0  # Normalize the image
    x = np.expand_dims(x, axis=0)

    # Make predictions
    predictions = model.predict(x)
    class_indices = {0: "Pepper bell | Disease: Bacterial spot",
                     1: "Pepper bell | No Disease: Healthy",
                     2: "Potato | Disease: Early blight",
                     3: "Potato | Disease: Late blight",
                     4: "Potato | No Disease: Healthy",
                     5: "Tomato | Disease: Bacterial spot",
                     6: "Tomato | Disease: Early blight",
                     7: "Tomato | Disease: Late blight",
                     8: "Tomato | Disease: Leaf Mold",
                     9: "Tomato | Disease: Septoria leaf spot",
                     10: "Tomato | Disease: Spider mites Two spotted spider mite",
                     11: "Tomato | Disease: Target Spot",
                     12: "Tomato | Disease: Yellow Leaf Curl Virus",
                     13: "Tomato | Disease: Mosaic Virus",
                     14: "Tomato | No Disease: Healthy"}

    predicted_class_index = np.argmax(predictions)
    predicted_class = class_indices[predicted_class_index]

    return predicted_class

# Capture an image from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Press 'q' to capture an image and make a prediction
    if cv2.waitKey(1) & 0xFF == ord('q'):
        result = predict_leaf_disease(frame, model)
        print(result)

    # Press 'ESC' to exit the loop
    if cv2.waitKey(1) == 27:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
