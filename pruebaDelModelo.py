import cv2
import tensorflow as tf
import numpy as np

# Cargar el modelo entrenado
modelo = tf.keras.models.load_model('reconocimientoFacialModelV2.h5')

# Lista de nombres de las clases (deberías tener esto de acuerdo a tu modelo)
class_names = ['n000148',
               'n000082',
               'guido',
               'n000106',
               'n000078',
               'n000040',
               'n000129',
               'n000029',
               'n000009',
               'n000001',
               'n000178',
               'n000149']  # Asegúrate de que esto coincida con tus clases

# Iniciar la captura de video
cap = cv2.VideoCapture(0)

# Cargar el clasificador Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en la imagen
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Dibujar un rectángulo alrededor del rostro detectado
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extraer la región del rostro
        face_roi = gray[y:y + h, x:x + w]

        # Redimensionar el ROI a 224x224 (tamaño esperado por el modelo)
        face_roi_resized = cv2.resize(face_roi, (224, 224))

        # Normalizar los valores de los píxeles y agregar una dimensión para el canal
        face_roi_input = face_roi_resized.reshape(1, 224, 224, 1) / 255.0

        # Realizar la predicción
        prediction = modelo.predict(face_roi_input)

        # Obtener el índice de la clase predicha y su probabilidad
        predicted_class_index = np.argmax(prediction)
        predicted_probability = prediction[0][predicted_class_index]

        # Obtener el nombre de la clase predicha
        predicted_class_name = class_names[predicted_class_index]

        # Si el porcentaje es menor que 70%, etiquetar como "Desconocido"
        if predicted_probability < 0.70:
            label = "Desconocido"
        else:
            label = f'{predicted_class_name}: {predicted_probability}%'
            
        # Mostrar la clase y el porcentaje sobre el rostro detectado
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Mostrar el marco procesado
    cv2.imshow('webcam', frame)

    # Salir si se presiona la tecla ESC
    if cv2.waitKey(30) == 27:  # 27 es el código ASCII para la tecla ESC
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
