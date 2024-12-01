# Deteccion y Reconocimiento Facial para Evitar Suplantacion de Identidad
Este proyecto tiene como principal objetivo crear un modelo de CNN para el reconocimiento de personas mediante sus rostros con la finalidad de evitar suplantancion de identidad para diferentes rubros.
Para la deteccion de los rostros de las personas se uso el Haar Cascade de OpenCV, especificamente 'haarcascade_frontalface_default.xml' que es el indicado para la deteccion de rostros.
Para el modelo que se creo se utilizo el dataset VGGFace2, al final para el entrenamiento del modelo se utilizo una version reducida del dataset VGGFace2, que contiene 3751 imagenes entre 12 personas.
El modelo se guardo en un archivo .h5 para utilizarlo en otras maquinas.
Adjuto el link del drive para descargar el modelo en archivo .h5 y el dataset VGGFace2 reducido que tiene como nombre valVGGface2 que se utilizo para el entrenamiento:
- [LinkDrive](https://drive.google.com/drive/folders/17lbxGxF5LLky8pn39AZ43N8VXsoJgQoH?usp=sharing)

Se recomienda entrenar el modelo en google colab para utilizar los recursos de google ya que tuve problemas con los recursos de mi computador, por lo tanto hay que cargar el dataset VGGFace2 y haarcascade en el drive de su cuenta en el utilizara google colab.
En el google colab cargaremos el jupyter notebook que esta adjunto en este repositorio, seguir las indicaciones que estan en el jupyter notebook.
Los resultados obtenidos del modelo entrenado son los siguientes:
- Loss: 0.5390
- Accuracy: 0.8842

Segun los resultados obtenido hasta la fecha (30/11/2024), la Loss nos dice que el error del modelo al hacer predicciones es moderada y en el caso del Accuracy nos dice que el modelo esta clasicando bien en la mayoria de casos.
Veamos la ejecucion del programa para probar el modelo creado:
#### Reconocimiento de una persona que existe en el dataset
<img width="480" alt="imgGuidoCNN" src="https://github.com/user-attachments/assets/83d4716e-5e38-451a-8f15-b5bf4b1aa807">

Vemos que tenemos que la persona tiene un 0.99958 de probabilidad de que sea 'guido', vemos que lo identifica bien.
#### Persona no identificada, desconocida
<img width="480" alt="imgDesconocidoCNN" src="https://github.com/user-attachments/assets/ec57c83e-b690-4dff-bbff-94594823616d">

Vemos que no reconoce a la persona ya que la probabilidad es <0.75
