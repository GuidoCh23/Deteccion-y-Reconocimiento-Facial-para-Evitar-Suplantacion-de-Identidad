import cv2
import tensorflow as tf
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
# Definir nuevamente la función de pérdida personalizada
def Categorical_Cross_Entropy_with_Penalization(y_true, y_pred):
    # Pérdida base: Categorical Crossentropy
    perdida_base = K.categorical_crossentropy(y_true, y_pred)
    
    # Penalización: Entropía de la predicción
    entropia_pred = -K.sum(y_pred * K.log(y_pred + K.epsilon()), axis=-1)
    
    # Ajuste del factor de penalización
    penalizacion = 0.1 * entropia_pred
    
    # Combinamos la pérdida base y la penalización
    return perdida_base + penalizacion

# Cargar el modelo con la función personalizada
modelo = tf.keras.models.load_model(
    'reconocimientoFacialModelV10.keras', 
    custom_objects={'Categorical_Cross_Entropy_with_Penalization': Categorical_Cross_Entropy_with_Penalization}
)

#Agregamos la lista de nombres de las clases
class_names = ['n000148',
               'n000029',
               'n000129',
               'n000078',
               'n000001',
               'n000040',
               'n000106',
               'n000082',
               'n000009',
               'n000149',
               'n000178',
               'guido']

#cap es la camara que se utilizara, por defecto 0
cap = cv2.VideoCapture(0)

#cargamos el Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #convertimos la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detectamos el rostro en la imagen
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        #graficamos un rectangulo alrededor del rostro detectado
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        #extraemos la region del rostro
        face_roi = gray[y:y + h, x:x + w]

        #redimensionamos a 224x224 (tamaño esperado por el modelo)
        face_roi_resized = cv2.resize(face_roi, (224, 224))

        #normalizamos los valores de los pixeles y agregamos el decimos que sea de un solo canal
        face_roi_input = face_roi_resized.reshape(1, 224, 224, 1) / 255.0

        #realizamos la prediccion
        prediction = modelo.predict(face_roi_input)

        #obtenemos el indice de la clase predicha y su probabilidad
        predicted_class_index = np.argmax(prediction)
        predicted_probability = prediction[0][predicted_class_index]

        #obtenemos el nombre de la clase predicha
        predicted_class_name = class_names[predicted_class_index]

        if predicted_probability < 0.87:
            label = "Desconocido"
        else:
            label = f'{predicted_class_name}: {predicted_probability}%' #si es que es >0.75 entonces al label agregamos la clase predicha y el valor de probabilidad
            
        #mostraremos la clase y el porcentaje sobre el rostro detectado
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('webcam', frame)

    #salir si se presiona la tecla ESC
    if cv2.waitKey(30) == 27:  # 27 es el codigo ASCII para la tecla ESC
        break

cap.release()
cv2.destroyAllWindows()
