# Reporte del Modelo Final

## Resumen Ejecutivo

El modelo final fue desarrollado utilizando el lenguaje de programación Python para leer lenguaje de señas. Se utilizó el conjunto de datos American Sign Language Dataset para entrenar el modelo. El modelo se basa en la detección de señas utilizando la biblioteca MediaPipe y la clasificación utilizando el modelo ViT (Vision Transformer) de la biblioteca Transformers. Se evaluaron las métricas de precisión, recall y F1 score para evaluar el rendimiento del modelo. Además, se generó una matriz de confusión para visualizar los resultados de la clasificación..

## Descripción del Problema

El problema abordado con el modelo final es la interpretación del lenguaje de señas. El lenguaje de señas es utilizado por las personas con discapacidad auditiva para comunicarse, y este modelo tiene como objetivo reconocer y traducir las señas realizadas por los usuarios. El contexto en el que se desarrolla es el campo de la tecnología de reconocimiento de imágenes y procesamiento de lenguaje natural. Los objetivos del modelo son mejorar la comunicación entre las personas sordas y oyentes y facilitar su inclusión en la sociedad. La justificación del modelo radica en su potencial para brindar una forma más accesible y eficiente de interpretar el lenguaje de señas..

## Descripción del Modelo

El modelo final se compone de dos etapas principales: la detección de señas utilizando la biblioteca MediaPipe y la clasificación utilizando el modelo ViT (Vision Transformer) de la biblioteca Transformers. Primero, se utilizó MediaPipe para detectar las señas en imágenes de lenguaje de señas. Luego, se aplicó el modelo ViT preentrenado para clasificar las señas detectadas en diferentes categorías correspondientes a las letras del alfabeto. La metodología utilizada se basa en técnicas de aprendizaje profundo y procesamiento de imágenes.

## Evaluación del Modelo

El modelo final fue evaluado utilizando métricas de evaluación como la precisión, recall y F1 score. Estas métricas permiten evaluar el rendimiento y la precisión del modelo en la clasificación de las señas del lenguaje de señas. La precisión del modelo fue de 94.93670, lo que indica el porcentaje de muestras clasificadas correctamente. El recall del modelo fue de 95.08689, que representa la capacidad del modelo para encontrar todas las instancias relevantes. Además, el F1 score del modelo fue de 95.0154, que es una medida ponderada que combina la precisión y el recall del modelo. Estas métricas reflejan el rendimiento general del modelo en la clasificación del lenguaje de señas..

## Conclusiones y Recomendaciones

En conclusión, el modelo final desarrollado para la interpretación del lenguaje de señas muestra una efectividad superior al 95%. El modelo presenta una forma practica para lograr una traduccion efectiva, lo que demuestra su eficacia en la detección y clasificación de las señas del lenguaje de señas. Sin embargo, también tiene inconvenientes a la hora de leer las señas en tiempo real, lo que indica posibles áreas de mejora. Es importante tener en cuenta las limitaciones del modelo, al considerar su aplicación en diferentes escenarios. En general, el modelo muestra un buen potencial para mejorar la comunicación y la inclusión de las personas sordas en la sociedad.

Se recomienda utlizar solo para imagenes pre cargadas. Estas recomendaciones podrían incluir imagenes de gran calidad y luz. Además, se sugiere realizar pruebas y validaciones adicionales del modelo en diferentes contextos y con diferentes usuarios para obtener una evaluación más exhaustiva de su rendimiento.

## Referencias

American Sign Language Dataset - https://www.kaggle.com/datasets/ayuraj/asl-dataset
Biblioteca MediaPipe - https://developers.google.com/mediapipe
Biblioteca Transformers - https://huggingface.co/docs/transformers/main/es/index

