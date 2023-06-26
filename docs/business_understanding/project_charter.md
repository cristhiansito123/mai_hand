# Project Charter - 

## Nombre del Proyecto

Reconocimiento del lenguaje de señas con IA

## Objetivo del Proyecto

El objetivo principal del proyecto es desarrollar un sistema de reconocimiento de lenguaje de señas utilizando técnicas de aprendizaje profundo y procesamiento de imágenes. El objetivo es mejorar la comunicación entre personas sordas y oyentes al reconocer y traducir las señas realizadas por los usuarios, permitiendo una interacción más efectiva y facilitando la inclusión de las personas sordas en la sociedad.

## Alcance del Proyecto

El alcance del proyecto se centra en el desarrollo de un modelo de reconocimiento de lenguaje de señas utilizando el lenguaje de programación Python. Se utilizará el conjunto de datos American Sign Language Dataset para entrenar el modelo. El sistema se enfocará en la detección y clasificación de señas del lenguaje de señas correspondientes a las letras del alfabeto. Se implementará la biblioteca MediaPipe para la detección de señas y se utilizará el modelo ViT (Vision Transformer) de la biblioteca Transformers para la clasificación.

El proyecto se limitará al reconocimiento de señas en imágenes pre cargadas de alta calidad y luz adecuada. No se abordará en tiempo real ni se considerarán otras variantes del lenguaje de señas más complejas o gestos específicos. El sistema se evaluará utilizando métricas de precisión, recall y F1 score para evaluar su rendimiento en la clasificación de las señas.

Es importante tener en cuenta que el proyecto no contempla la implementación de una interfaz de usuario completa ni su integración en aplicaciones o dispositivos específicos. El enfoque principal es desarrollar y evaluar el modelo de reconocimiento de lenguaje de señas en sí mismo.

### Incluye:

Descripción de los datos disponibles: El proyecto utilizará el conjunto de datos American Sign Language Dataset para entrenar el modelo de reconocimiento de lenguaje de señas. Este conjunto de datos contiene imágenes que representan las señas correspondientes a las letras del alfabeto en el lenguaje de señas americano. Se utilizará este conjunto de datos para entrenar y evaluar el modelo.

Descripción de los resultados esperados: El objetivo principal del proyecto es desarrollar un sistema de reconocimiento de lenguaje de señas que pueda detectar y clasificar correctamente las señas correspondientes a las letras del alfabeto. Se espera que el modelo pueda reconocer con precisión y traducir las señas realizadas por los usuarios en imágenes pre cargadas.

Criterios de éxito del proyecto: Los criterios de éxito del proyecto se basarán en el rendimiento y la precisión del modelo de reconocimiento de lenguaje de señas. Se establecerán métricas de evaluación, como la precisión, el recall y el F1 score, para evaluar el desempeño del modelo. El criterio de éxito será alcanzar un nivel de precisión y rendimiento aceptable, que se determinará en función de los estándares establecidos en la literatura existente o los requisitos del proyecto.

### Excluye:

Variantes complejas del lenguaje de señas: El alcance del proyecto se centrará en el reconocimiento de las señas correspondientes a las letras del alfabeto en el lenguaje de señas. No se abordarán gestos más complejos o variantes específicas del lenguaje de señas, como expresiones faciales, movimiento de manos más complejo o gestos específicos de ciertas palabras o frases.

Interfaz de usuario completa: El proyecto no incluirá la implementación de una interfaz de usuario completa para interactuar con el sistema de reconocimiento de lenguaje de señas. No se desarrollarán funcionalidades como la captura en tiempo real de señas, la traducción en tiempo real o la integración en aplicaciones o dispositivos específicos.

Mejoras adicionales o extensiones del modelo: El alcance del proyecto se centrará en el desarrollo y evaluación del modelo de reconocimiento de lenguaje de señas utilizando técnicas de aprendizaje profundo y procesamiento de imágenes. No se incluirá la implementación de mejoras adicionales o extensiones del modelo, como el uso de técnicas de transfer learning, la optimización de hiperparámetros o la exploración de arquitecturas más avanzadas.

## Metodología

El proyecto se basará en una metodología que combina técnicas de aprendizaje profundo y procesamiento de imágenes para el reconocimiento de lenguaje de señas. A continuación, se describe una posible metodología que podría seguirse:

Recopilación y preparación de datos: Se adquirirá el conjunto de datos American Sign Language Dataset, que contiene imágenes que representan las señas correspondientes a las letras del alfabeto en el lenguaje de señas americano. Estos datos se prepararán para su uso en el entrenamiento y evaluación del modelo.

Entrenamiento del modelo de detección de señas: Se utilizará la biblioteca MediaPipe para la detección de señas en las imágenes. Se entrenará un modelo utilizando técnicas de aprendizaje profundo y el conjunto de datos preparado. El objetivo es que el modelo pueda identificar y localizar las señas presentes en las imágenes.

Entrenamiento del modelo de clasificación de señas: Se empleará el modelo ViT (Vision Transformer) de la biblioteca Transformers para la clasificación de las señas detectadas. Se utilizarán las imágenes de las señas recortadas por el modelo de detección como entrada para el modelo de clasificación. El objetivo es que el modelo pueda asignar la categoría correcta (letras del alfabeto) a cada señal detectada.

Evaluación y ajuste del modelo: Se realizarán pruebas y evaluaciones del modelo utilizando métricas de evaluación, como la precisión, el recall y el F1 score. Estas métricas permitirán medir el rendimiento del modelo en la clasificación de las señas del lenguaje de señas. En caso de ser necesario, se realizarán ajustes en los parámetros o en la arquitectura del modelo para mejorar su rendimiento.

Validación del modelo: Se realizarán pruebas y validaciones adicionales del modelo en diferentes contextos y con diferentes usuarios para obtener una evaluación más exhaustiva de su rendimiento. Esto puede implicar la recolección de datos adicionales o la colaboración con expertos en lenguaje de señas para validar la precisión y efectividad del modelo.

## Cronograma

| Etapa | Duración Estimada | 
|------|---------|
| Recopilación y preparación de datos | 2 semanas | 
| Entrenamiento del modelo de detección de señas | 1 semana | 
| Entrenamiento del modelo de clasificación de señas | 1 semana |
| Evaluación y ajuste del modelo | 1 semanas |
| Validación del modelo | 1 semanas | 

## Equipo del Proyecto

- Jhon Nelson Cáceres Leal
- Cristian David Sandoval Diaz
- Yuli Fernanda Alpala Cuaspa



