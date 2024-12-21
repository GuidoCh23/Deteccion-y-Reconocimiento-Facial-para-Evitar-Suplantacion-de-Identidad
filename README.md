# Deteccion y Reconocimiento Facial para Evitar Suplantacion de Identidad
Este proyecto tiene como principal objetivo crear un modelo de CNN para el reconocimiento de personas mediante sus rostros con la finalidad de evitar suplantancion de identidad para diferentes rubros.
Para la deteccion de los rostros de las personas se uso el Haar Cascade de OpenCV, especificamente 'haarcascade_frontalface_default.xml' que es el indicado para la deteccion de rostros.
Para el modelo que se creo se utilizo el dataset VGGFace2, al final para el entrenamiento del modelo se utilizo una version reducida del dataset VGGFace2, que contiene 3720 imagenes entre 12 personas.
El modelo se guardo en un archivo .keras para utilizarlo en otras maquinas.
Adjuto el link del drive para descargar el modelo en archivo .keras y el dataset VGGFace2 reducido que tiene como nombre valVGGface2 que se utilizo para el entrenamiento:
- [LinkDrive](https://drive.google.com/drive/folders/17lbxGxF5LLky8pn39AZ43N8VXsoJgQoH?usp=sharing)

Se recomienda entrenar el modelo en google colab para utilizar los recursos de google ya que tuve problemas con los recursos de mi computador, por lo tanto hay que cargar el dataset VGGFace2 y haarcascade en el drive de su cuenta en el utilizara google colab.
En el google colab cargaremos el jupyter notebook que esta adjunto en este repositorio, seguir las indicaciones que estan en el jupyter notebook.
Los resultados obtenidos del modelo entrenado son los siguientes:
- Loss: 0.4905
- Accuracy: 0.8642

Segun los resultados obtenido hasta la fecha (20/12/2024), la Loss nos dice que el error del modelo al hacer predicciones es moderada y en el caso del Accuracy nos dice que el modelo esta clasicando bien en la mayoria de casos.
Veamos la ejecucion del programa para probar el modelo creado:
#### Reconocimiento de una persona que existe en el dataset
<a href="https://postimg.cc/xXhXy1fF">
  <img src="https://i.postimg.cc/3wJ2hkP7/guidoImg.jpg" alt="guidoImg" width="450">
</a>

Vemos que tenemos que la persona tiene un 0.99958 de probabilidad de que sea 'guido', vemos que lo identifica bien.
#### Persona no identificada, desconocida
<a href="https://postimg.cc/xXhXy1fF">
  <img src="https://i.postimg.cc/SsH9cDjN/aronImg.png" alt="aronImg" width="450">
</a>

Vemos que no reconoce a la persona ya que la probabilidad es <0.87
