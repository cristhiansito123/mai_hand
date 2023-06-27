import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

cap = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles
ruta_actual = os.path.abspath(__file__)
carpeta_actual = os.path.dirname(ruta_actual)
ruta_modelo = os.path.join(carpeta_actual, 'random_forest_model.pkl')
model = joblib.load(ruta_modelo)
prediction = ['']
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.1)
imagen_recortada2 = np.zeros((240, 320,3), dtype=np.uint8)
# Set mediapipe model 
while cap.isOpened():

        # Read feed
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #blurred_image = cv2.GaussianBlur(frame_rgb, (5, 5), 0)
        #imagen_recortada2 = cv2.resize(blurred_image, (320, 240))
        resultado = hands.process(frame_rgb)
        if resultado.multi_hand_landmarks:
              
              # Obtener los puntos de referencia de la mano
              mano_landmarks = resultado.multi_hand_landmarks[0]

                  # Determinar los límites del rectángulo de recorte
              altura, ancho, _ = frame_rgb.shape
              puntos_x = [punto.x * ancho for punto in mano_landmarks.landmark]
              puntos_y = [punto.y * altura for punto in mano_landmarks.landmark]
              min_x, max_x = int(min(puntos_x)), int(max(puntos_x))
              min_y, max_y = int(min(puntos_y)), int(max(puntos_y))

                  # Determinar el factor de expansión del recorte
              factor_expansion = 0.4  # Ajusta este valor según tus necesidades (0.2 es solo un ejemplo)

                  # Calcular los valores de expansión
              expansion_x = int((max_x - min_x) * factor_expansion)
              expansion_y = int((max_y - min_y) * factor_expansion)

                  # Ajustar los límites del recorte
              min_x -= expansion_x
              max_x += expansion_x
              min_y -= expansion_y
              max_y += expansion_y

                  # Asegurarse de que los límites no excedan las dimensiones de la imagen
              min_x = max(0, min_x)
              max_x = min(ancho - 1, max_x)
              min_y = max(0, min_y)
              max_y = min(altura - 1, max_y)

                  # Recortar la imagen alrededor de la mano con los nuevos límites
              imagen_recortada2 = frame_rgb[min_y:max_y, min_x:max_x]
              ##frame= imagen_recortada

        resultado2 = hands.process(imagen_recortada2)
        if resultado2.multi_hand_landmarks:
              #Procesamiento de los puntos de la mano
              for hand_landmarks in resultado2.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame[min_y:max_y, min_x:max_x],hand_landmarks,mp_hands.HAND_CONNECTIONS,mp_drawing_styles.get_default_hand_landmarks_style(),mp_drawing_styles.get_default_hand_connections_style())

              for hand_landmarks in resultado2.multi_hand_landmarks:
                    hand_landmarks_cropped = resultado2.multi_hand_landmarks[0]
                    landmarks_data = []
                    for landmark in hand_landmarks_cropped.landmark:
                        landmarks_data.append((landmark.x, landmark.y, landmark.z))
              landmarks_data=np.asarray(landmarks_data)
              landmarks_data=landmarks_data.reshape(landmarks_data.shape[0]*landmarks_data.shape[1])
              landmarks_data=landmarks_data.reshape(1, -1)
              prediction = model.predict(landmarks_data)
              #print(landmarks_data)
        # Show to screen
        cv2.putText(frame, str(prediction[0]), (300, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.imshow('OpenCV Feed', frame)
        
        # Break gracefully
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()