#!/usr/bin/env python
# coding: utf-8



# import tensorflow as tf

video_capture = cv2.VideoCapture(0)
# video_capture = cv2.imread(video_capture, cv2.IMREAD_GRAYSCALE)
face_locations = []

model = model_from_json(open("fer.json", "r").read())
model.load_weights('fer.h5')

# model = load_model('model_neuro')

emotion_values = ['angry', "disgust", "fear", "happy", "neutral", "sad", "surprise"]

while True: # Grab a single frame of video
    ret, frame = video_capture.read()# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]# Find all the faces in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)# Display the results

    for top, right, bottom, left in face_locations:
    # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)# Display the resulting image
        face = frame[top:bottom, left:right]
        if len(face) > 0:
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face_gray48 = cv2.resize(face_gray, (48, 48))

            image_pixels = np.array(face_gray48)
            image_pixels = np.expand_dims(image_pixels, axis=0)
            image_pixels = image_pixels.reshape(image_pixels.shape[0], 48, 48, 1)

            predictions = model.predict(image_pixels)
            max_index = np.argmax(predictions)

            emotion_prediction = emotion_values[max_index]
            print(max_index, emotion_prediction)
            # num = random.randint(1,27)

            cv2.putText(frame, 'face', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
            cv2.putText(frame, emotion_values[max_index], (left, top - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
        cv2.imshow('Video', frame)# Hit ‘q’ on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# In[ ]:




