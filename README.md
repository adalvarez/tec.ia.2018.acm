<!-- MarkdownTOC autolink="true" autoanchor="true" bullets="*"-->

"*"--> [l Proyecto de Inteligencia Artificial](#l-proyecto-de-inteligencia-artificial)
			"*"--> [Sobre el proyecto:](#sobre-el-proyecto)
"*"--> [Marco Teórico](#marco-te%C3%B3rico)
		"*"--> [K-Nearest Neighbors utilizando K-d Trees:](#k-nearest-neighbors-utilizando-k-d-trees)
		"*"--> [Regresión Logística:](#regresi%C3%B3n-log%C3%ADstica)
		"*"--> [Árboles de decisión:](#%C3%81rboles-de-decisi%C3%B3n)
		"*"--> [Redes Neuronales:](#redes-neuronales)
		"*"--> [Support Vector Machines:](#support-vector-machines)
		"*"--> [Cross-Validation:](#cross-validation)
"*"--> [Análisis de modelos:](#an%C3%A1lisis-de-modelos)
		"*"--> [Regresión Logística:](#regresi%C3%B3n-log%C3%ADstica-1)
		"*"--> [Árboles de decisión:](#%C3%81rboles-de-decisi%C3%B3n-1)
		"*"--> [K-Nearest Neighbors utilizando K-d Trees:](#k-nearest-neighbors-utilizando-k-d-trees-1)
		"*"--> [Redes Neuronales:](#redes-neuronales-1)
		"*"--> [Support Vector Machines:](#support-vector-machines-1)
				"*"--> [Linear](#linear)
				"*"--> [Polynomial](#polynomial)
				"*"--> [RBF](#rbf)
				"*"--> [Sigmoid](#sigmoid)
	"*"--> [Conclusión](#conclusi%C3%B3n)
"*"--> [Manual de uso:](#manual-de-uso)
	"*"--> [Instalación  de los requerimientos:](#instalaci%C3%B3n-de-los-requerimientos)
			"*"--> [tec.ic.ia.pc1](#teciciapc1)
			"*"--> [SciPy con NumPy](#scipy-con-numpy)
			"*"--> [Scikit-learn](#scikit-learn)
			"*"--> [Keras](#keras)
			"*"--> [tensorflow](#tensorflow)
			"*"--> [Pandas](#pandas)
	"*"--> [Uso del sistema](#uso-del-sistema)

<!-- /MarkdownTOC -->

<a id="l-proyecto-de-inteligencia-artificial"></a>
# l Proyecto de Inteligencia Artificial
Autores:
Adrián Álvarez - Marlon Agüero - César Borge

<a id="sobre-el-proyecto"></a>
#### Sobre el proyecto:
Durante el mes de febrero y abril, se realizaron en Costa Rica las elecciones presidenciales. El propósito de este proyecto fue **analizar** el comportamiento y los resultados de diferentes **modelos de Inteligencia Artificial** particularmente dentro del área de **machine learning**, al ser utilizados para **predecir** a partir de datos de votantes ficticios, para cual partido político fueron sus votos para la primera y segunda ronda de las elecciones. Para cumplir con dicho objetivo se analizaron los siguientes modelos de inteligencia artificial:

* Árboles de decisión.
* K-Nearest Neighbors utilizando K-d Trees.
* Regresión Logística.
* Redes Neuronales.
* Support Vector Machines.

En el proyecto corto 1, el equipo de desarrollo tenía como objetivo generar datos ficticios de votantes de las elecciones presidenciales de Costa Rica, realizadas en febrero el 2018. Para ello, se tenía que tomar en cuenta datos proporcionados por el Tribunal Supremo de Elecciones, respecto a la cantidad de votos por junta y datos del Estado de la Nación, respecto a los indicadores cantonales a los que pertenecían dichas juntas.

Se le proporcionó al equipo los siguientes enlaces para obtener dicha información:


* [Votos por junta](http://www.tse.go.cr/elecciones2018/actas_escrutinio.htm)

* [Juntas por cantón](http://www.tse.go.cr/pdf/nacional2018/JRV.pdf)
 
* [Indicadores cantonales](https://www.estadonacion.or.cr/files/biblioteca_virtual/otras_publicaciones/Indicadores-cantonales_Censos-2000-y-2011.xlsx)

* [Distribución de edades](http://www.inec.go.cr/wwwisis/documentos/INEC/Indicadores_Demograficos_Cantonales/Indicadores_Demograficos_Cantonales_2013.pdf)


A continuación una lista de los datos tomados en cuenta de cada cantón, para generar los votantes:

1. Cantidad de votos de cada partido, nulos y blancos para primera ronda.
1. Cantidad de votos de cada partido, nulos y blancos para segunda ronda.
1. Población total.
1. Superficie.
1. Densidad de población. 
1. Porcentaje de población urbana.
1. Relación hombres-mujeres.
1. Relación de dependencia demográfica.
1. Viviendas individuales ocupadas.
1. Promedio de ocupantes.
1. Porcentaje de viviendas en buen estado.
1. Porcentaje de viviendas hacinadas.
1. Porcentaje de alfabetismo.
1. Escolaridad promedio.
1. Porcentaje de asistencia a la educación regular.
1. Personas fuera de la fuerza de trabajo.
1. Tasa neta de participación en la fuerza de trabajo.
1. Porcentaje de población ocupada no asegurada.
1. Porcentaje de población nacida en el extranjero.
1. Porcentaje de población con discapacidad.
1. Porcentaje de población no asegurada.
1. Porcentaje de hogares con jefatura femenina
1. Porcentaje de hogares con jefatura compartida.
1. Cantidad de personas mayores a 18 años en el país.
1. Cantidad de personas mayores a 65 años en el país.

Por su parte cada votante fue generado con los siguientes datos:


| Dato | Valores posibles | Criterio de asignación |
|---------------------------------------------|----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Cantón | Cualquiera de los cantones del país | La probabilidad de asignación se basó en la cantidad de votos de cada cantón. |
| Género | M, F | La probabilidad de asignación se basó en el índice de hombres-mujer de cada cantón. |
| Edad | 18 - 105 | La probabilidad de asignación se basó en la distribución de edades según el INEC. |
| Zona | Urbana, Rural | La probabilidad de asignación se basó en el índice de población urbana de cada cantón. |
| Dependiente | SI, NO | La probabilidad de asignación se basó en el índice de relación de dependencia demográfica de cada cantón, tomando en cuenta también la edad generada. |
| Casa en buen estado | SI, NO | La probabilidad de asignación se basó en el índice de viviendas en buen estado de cada cantón. |
| Casa Hacinada | SI, NO | La probabilidad de asignación se basó en el índice de viviendas hacinadas de cada cantón. |
| Alfabeta | SI, NO | La probabilidad de asignación se basó en el índice de alfabetismo de cada cantón. Tomando también en cuenta la edad generada. |
| Promedio de escolaridad | Cantidad de años de escolaridad. | Valor estático obtenido del índice de escolaridad promedio. La asignación de este valor depende de la edad generada. |
| Asistencia a educación regular | SI, NO | La probabilidad de asignación se basó en el índice de asistencia a educación regular de cada cantón. Tomando también en cuenta la edad generada. |
| En la fuerza de trabajo | SI, NO | La probabilidad de asignación se basó en el índice de participación en la fuerza de trabajo de cada cantón. Tomando también en cuenta el género generado. |
| Asegurado | SI, NO | La probabilidad de asignación se basó en el índice de población no asegurada de cada cantón. Tomando también en cuenta el dato anterior “En la fuerza de trabajo”. |
| Extranjero | SI, NO | La probabilidad de asignación se basó en el índice de la población nacida en el extranjero de cada cantón. |
| Discapacitado | SI, NO | La probabilidad de asignación se basó en el índice de la población discapacitada de cada cantón. |
| Jefe de Hogar | HOMBRE, MUJER, COMPARTIDO | La probabilidad de asignación se basó en los índices de jefatura de hogar femenina y compartida de cada cantón. |
| Voto Primera Ronda | Cualquier partido político de las elecciones presidenciales de Costa Rica 2018, NULO, BLANCO. | La probabilidad de asignación se basó en la cantidad de votos de cada partido, voto nulo o voto blanco, de cada cantón para la primera ronda. |
| Voto Primera Ronda | Cualquier partido político de las elecciones presidenciales de Costa Rica 2018 en segunda ronda, NULO, BLANCO. | La probabilidad de asignación se basó en la cantidad de votos de cada partido, voto nulo o voto blanco, de cada cantón para la segunda ronda. |
| Población total (cantón) | Cantidad de personas. | Valor estático por cada cantón. |
| Superficie (cantón) | Cantidad de kilómetros cuadrados. | Valor estático por cada cantón. |
| Densidad (cantón) | Personas por kilómetro cuadrado | Valor estático por cada cantón. |
| Viviendas ocupadas (cantón) | Cantidad de viviendas. | Valor estático por cada cantón. |
| Promedio de ocupantes por vivienda (cantón) | Promedio de ocupantes. | Valor estático por cada cantón. |

Este generador cuenta con dos funciones:
* `generar_muestra_pais`: Recibe un parámetro `n`. Genera una población de tamaño `n` tomando en cuenta todos los cantones del país.
* `generar_muestra_provincia`: Recibe un parámetro `n` y un parámetro `provincia`. Genera una población de tamaño `n` tomando en cuenta únicamente los cantones de `provincia`.

A continuación se presenta un marco teórico que resume la teoría necesaria, con el fin de que el lector comprenda el análisis de dichos modelos.

<a id="marco-te%C3%B3rico"></a>
# Marco Teórico

<a id="k-nearest-neighbors-utilizando-k-d-trees"></a>
### K-Nearest Neighbors utilizando K-d Trees:
La idea de **K-Nearest Neighbors** es que dado un ejemplo de entrada, se retornen los `k` ejemplos de entrenamiento con la menor distancia al ejemplo de entrada. El ejemplo de entrada será clasificado con la clase de mayor pluralidad entre los vecinos encontrados. Teniendo en cuenta que cada ejemplo será un vector de `n` dimensiones, la distancia entre cada ejemplo se puede calcular utilizando la fórmula de distancia euclidiana:

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/5d808dc9b184ca40b14c1950be6e48c0a323a583) 

En la siguiente imágen se puede apreciar un espacio de dos dimensiones donde se buscan los 3 y 6 vecinos más cercanos al punto de la estrella:

![](http://adataanalyst.com/wp-content/uploads/2016/07/kNN-1.png) 


La forma más sencilla de implementar este modelo es guardar cada uno de los ejemplos en una tabla y cuando se tenga un ejemplo de entrada, calcular la distancia con respecto a todos los ejemplos de la tabla y retornar los k ejemplos que tengan una menor distancia. Sin embargo esta solución tiene una complejidad  `O(n)`, por lo tanto cuando tenemos millones de ejemplos, se vuelve muy costosa. Para solucionar esto se utilizan los **K-d Trees**, en los cuales se realizan búsquedas igual que en los árboles binarios, lo cual reduce la complejidad a `O(log2(n))`.
En la siguiente imagen se visualiza un K-d Tree de 3 dimensiones:

![](https://gopalcdas.files.wordpress.com/2017/05/1.png) 



Para crear un algoritmo de K-Nearest Neighbors utilizando K-d Trees desde cero, se recomienda este [artículo](https://gopalcdas.com/2017/05/24/construction-of-k-d-tree-and-using-it-for-nearest-neighbour-search/).

<a id="regresi%C3%B3n-log%C3%ADstica"></a>
### Regresión Logística:
Antes de entender cómo funciona una regresión logística es importante entender el significado estadístico de `regresión`. Una regresión es un proceso estadístico a través del cual se puede estimar o predecir una relación entre un conjunto de variables. Estas variables tienen una diferencia importante, se dividen en variables independientes (o también llamadas predictoras) y en variables dependientes (predicción). La `regresión logística` es un tipo especial de regresión que se utiliza para explicar y predecir una variable categórica binaria en función de un conjunto de variables independientes que a su vez pueden ser cuantitativas o categóricas. Debido al proceso matemático bajo el cual la regresión logística funciona, es crucial y beneficioso la conversión de variables categóricas a cuantitativas. También resulta conveniente 'binarizar' estas conversiones (categórica->cuantitativa). Otro aspecto importante por realizar para mejorar los resultados de la regresión y al mismo tiempo hacer más justa dicha regresión es normalizar (convertir un conjunto de valores que pueden tener escalas muy diferentes en una escala común) los datos. Regresión logística puede a su vez dar una clasificación no binaria, a través de un concepto llamado `clasificación multinomial` en donde la predicción se puede dar en un conjunto de clases posibles, pero sólo en una clase (excluyente). Para la implementación de cualquier tipo regresión (lineal, logística) se puede utilizar [Tensorflow](https://www.tensorflow.org/tutorials/wide) un framework para hacer Machine Learning. Dicho framework fue utilizado en este proyecto.

<a id="%C3%81rboles-de-decisi%C3%B3n"></a>
### Árboles de decisión:

Un `árbol de decisión` toma como entrada un objeto o una situación descrita a través de un conjunto de atributos y devuelve una decisión: el valor previsto de la salida dada la entrada. Un árbol de decisión desarrolla una secuencia de test para poder alcanzar una decisión. Cada nodo interno del árbol corresponde con un test sobre el valor de una de las propiedades, y las ramas que salen del nodo están etiquetadas con los posibles valores de dicha propiedad. Cada nodo hoja del árbol representa el valor que ha de ser devuelto si dicho nodo hoja es alcanzado. Si el árbol de decisión pretende dar una respuesta binaria (aunque puede ser categórica) el aspecto crucial para la construcción de un árbol de decisión es elegir el atributo que separe en mayor medidas las decisiones. Para ello se utiliza `ganancia de información` para obtener una medida de discriminación para seleccionar el mejor atributo para bajar de nivel en el árbol.

![decisionTree](https://www.tutorialspoint.com/data_mining/images/dm_decision_tree.jpg)

<a id="redes-neuronales"></a>
### Redes Neuronales:
Desde los inicios de la Inteligencia Artificial algunos trabajos pretendían crear redes neuronales artificiales. Este modelo busca que cada neurona (nodo o unidad) tenga una función de activación y se interconecte con otras neuronas. Una neurona será activada cuando una combinación lineal en sus entradas exceda un umbral determinado por la función de activación elegida. Entre las funciones de activación se pueden encontrar la Softmax, Sigmoid, Relu. Para poder implementar una red neuronal en Python de manera sencilla y rápida se puede utilizar [Keras](https://keras.io/), que es un API a alto nivel para la creación de redes neuronales, además es el que se utilizó para realizar este proyecto. También puede seguir el siguiente [tutorial](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/) simple donde se crea una red neuronal sencilla con Keras. 
Nota: Los datos necesarios para la creación de las redes neuronales son los mismos utilizados por la regresión.

![Red Neuronal](https://upload.wikimedia.org/wikipedia/commons/thumb/6/64/RedNeuronalArtificial.png/400px-RedNeuronalArtificial.png)

<a id="support-vector-machines"></a>
### Support Vector Machines:
**Support Vector Machines o SVMs** son un conjunto de algoritmos de aprendizaje supervisado. Se puede utilizar para problemas de clasificación (como el de este proyecto).
Es decir, se pueden etiquetar una serie de datos para entrenar un SVM y así, construir un modelo que prediga la clase de una muestra de entrada.
Para lograr esto SVM, representa los puntos de muestra (aquellos que se utilizaron durante el entrenamiento) en el espacio y los separa en 2 espacios por medio de un hiperplano de separación. Este hiperplano, conocido como **vector soporte**, es un vector entre los 2 puntos, de las 2 clases, más cercanos. Cuando se introduce una nueva muestra, con el objetivo de predecir su clase, dicha muestra se introduce en el modelo entrenado y se verifica a cual espacio pertenece.

En la siguiente imagen se presenta un caso básico de SVM:

![](https://aitrends.com/wp-content/uploads/2018/01/1-19SVM-2.jpg) 

La forma más fácil para separar ambos espacios es utilizando una línea recta, sin embargo la mayoría de problemas de clasificación presentan más de dos posibles valores de predicción o de clasificación. Para ello se utilizan las funciones **Kernel** que permiten mapear el espacio en un nuevo espacio de características de mayor dimensionalidad.

![](http://scikit-learn.org/stable/_images/sphx_glr_plot_iris_001.png) 

Para implementar SVM en este proyecto se utilizó [SciKit Learn](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/).


<a id="cross-validation"></a>
### Cross-Validation:
**Cross-Validation** es un conjunto de algoritmos para entrenar y testear la precisión de un modelo. Para ello separa los datos a disposición en un `Training-Set` y en un `Test-Set`. En este proyecto se implementaron dos formas de Cross-Validation:

1. **Hold-Out**: Reserva un porcentaje de los datos para el `Test-Set`. Estos datos nunca son utilizados para entrenar el sistema, pero si para probarlo.

1. **K-Fold**: Cada ejemplo de los datos es usado para entrenar y para probar. Realiza `k` iteraciones, donde en cada una se separa un conjunto de tamaño `1/k` para `Test-Set`, y el resto se usa para `Training-Set`.



<a id="an%C3%A1lisis-de-modelos"></a>
# Análisis de modelos:
A continuación, el análisis desarrollado para cada uno de los modelos mencionados. Para cada prueba se utilizó el modelo para predecir los votos de primera ronda (`1R`), segunda ronda (`2R`) y segunda ronda tomando en cuenta el voto de primera ronda (`2R1R`) y se utilizó Hold Out Cross-Validation usando una población de 5000 votantes, separando un 25% como **test-set**. Excepto en las pruebas de K-Nearest Neighbors que se utilizó una población de 1000 para reducir tiempos de ejecución.

<a id="regresi%C3%B3n-log%C3%ADstica-1"></a>
### Regresión Logística:
El siguiente análisis de regresión logística tuvo como objetivo evaluar un conjunto de diferentes combinaciones para la regularización `l1` y `l2`. Regularización es una técnica usada para evitar o disminuir el overfitting. Overfitting sucede cuando el modelo aprende tanto de los datos de entrenamiento que no es capaz de generalizar y responder acertadamente ante nuevos datos o casos no vistos anteriormente. A continuación se enlistan las diferentes pruebas aplicadas:

1. Incremento igualitario para `l1` y `l2`:
Esta prueba consiste en utilizar `l1=l2=0.1`, posteriormente `l1=l2=0.3`, `l1=l2=0.5`, `l1=l2=0.7`, `l1=l2=0.9`. Cabe mencionar que esta prueba cuenta con la particularidad que los valores de exactitud (accuracy) son un promedio de 5 ejecuciones realizadas para la misma combinación.

![](https://i.imgur.com/morYKLg.png)

Podemos observar una constancia de accuracy para el set de test según sea la ronda. Pero es curioso como en general predecir la segunda ronda tienen a ser más acertado que la primera, quizás por el número de clases con las que se trabajan.

2. `l1=0` e incremento de `l2`:
Esta prueba consiste en utilizar `l1=0` y `l2=0.1`, posteriormente `l2=0.3`, `l2=0.5`, `l2=0.7`, `l2=0.9` con `l1=0`. Cabe mencionar que esta prueba cuenta con la particularidad que los valores de exactitud (accuracy) son un promedio de 3 ejecuciones realizadas para la misma combinación.

![](https://i.imgur.com/jyCQGzh.png)


Podemos observar una constancia de accuracy para el set de test según sea la ronda. Pero es curioso como en general predecir la segunda ronda tienen a ser más acertado que la primera, quizás por el número de clases con las que se trabajan.

3. `l2=0` e incremento de `l1`:
Esta prueba consiste en utilizar `l2=0` y `l1=0.1`, posteriormente `l1=0.3`, `l1=0.5`, `l1=0.7`, `l1=0.9` con `l2=0`.Cabe mencionar que esta prueba cuenta con la particularidad que los valores de exactitud (accuracy) son un promedio de 3 ejecuciones realizadas para la misma combinación.

![](https://i.imgur.com/axLBjWc.png)


Podemos observar una constancia de accuracy para el set de test según sea la ronda. Pero es curioso como en general predecir la segunda ronda tienen a ser más acertado que la primera, quizás por el número de clases con las que se trabajan.

Finalmente parece crucial comparar estas tres pruebas para tomar una decisión sobre qué modificación o combinación de `l1` y `l2` utilizar para maximizar el accuracy. A continuación se muestran los resultados de cada tipo de prueba para `1R`, `2R``2R1R`. 

![](https://i.imgur.com/zFvXj6M.png)
![](https://i.imgur.com/hJi9nWi.png)
![](https://i.imgur.com/3pCdqY2.png)

Se puede concluir que aumentar los valores de `l1` y `l2` simultáneamente contribuye para la predicción de la primera ronda, mientras que para las segunda ronda conviene más utilizar `l2=0` y `l1=x` donde `x` puede ser el valor que más se adecue.
La ejecución de este modelo podría ser como el siguiente:

`python g01.py --regresion-logistica --holdout --porcentaje-pruebas 25 --población 5000 --l2 0.1 --l1 0.0`

También cabe mencionar que dadas las pruebas realizadas con las variaciones de `l1` y `l2` se llegó a un máximo de 31% (trainning) y 27% (test) para primera ronda y, 64% (trainning) y 63% (test) segunda ronda. En general para este modelo, el aporte de la votación de la primera ronda para predecir la segunda tiende a ser bajo.

**Máximos generales**:

| Ronda        | Precisión Training Set | Precisión Test Set  |
| -------------|          -------------:|               -----:|
| 1R           | 31.10%                 |              27.89% |
| 2R           | 64.27%                 |              62.80% |
| 2R1R         | 64.45%                 |              62.77% |

<a id="%C3%81rboles-de-decisi%C3%B3n-1"></a>
### Árboles de decisión:

El siguiente análisis de árboles de decisión tuvo como objetivo evaluar diferentes niveles de umbrales de poda para el árbol creado. La poda de un árbol tiene el objetivo de evitar el overfitting Se parte de la población anteriormente indicada al igual que el porcentaje de test set.

Al utilizar el siguiente umbral se obtiene los siguientes resultados `--umbral=poda=0.1`

| Ronda        | Precisión Training Set | Precisión Test Set  |
| -------------|          -------------:|               -----:|
| 1R           | 91.37%                 |              18.13% |
| 2R           | 95.01%                 |              52.0% |
| 2R1R         | 98.41%                 |              53.62% |

Al utilizar el siguiente umbral se obtiene los siguientes resultados `--umbral=poda=0.3`

| Ronda        | Precisión Training Set | Precisión Test Set  |
| -------------|          -------------:|               -----:|
| 1R           | 90.53%                 |              18.72% |
| 2R           | 94.03%                 |              54.45% |
| 2R1R         | 98.82%                 |              54.16% |

Al utilizar el siguiente umbral se obtiene los siguientes resultados `--umbral=poda=0.5`

| Ronda        | Precisión Training Set | Precisión Test Set  |
| -------------|          -------------:|               -----:|
| 1R           | 89.39%                 |              19.84% |
| 2R           | 91.22%                 |              54.66% |
| 2R1R         | 98.01%                 |              54.53% |

Al utilizar el siguiente umbral se obtiene los siguientes resultados `--umbral=poda=0.7`

| Ronda        | Precisión Training Set | Precisión Test Set  |
| -------------|          -------------:|               -----:|
| 1R           | 88.13%                 |              19.43% |
| 2R           | 90.37%                 |              54.42% |
| 2R1R         | 97.47%                 |              54.32% |

Al utilizar el siguiente umbral se obtiene los siguientes resultados `--umbral=poda=0.9`

| Ronda        | Precisión Training Set | Precisión Test Set  |
| -------------|          -------------:|               -----:|
| 1R           | 85.26%                 |              20.29% |
| 2R           | 87.03%                 |              55.19% |
| 2R1R         | 93.60%                 |              53.81% |

Podemos notar que conforme el umbral de poda aumente el accuracy para el entrenamiento empieza a disminuir, lo cual es esperable. Por su parte el accuracy para el set de pruebas tiende a aumentar justamente porque podar el árbol hará que la generalización aumente. A continuación se muestra una comparativa que incluye solamente el accuracy de pruebas.

![](https://i.imgur.com/zuJrnrI.png)

Se puede apreciar una importante diferencia entre la precisión para primera ronda respecto a la segunda.

**Máximos generales**:

| Ronda        | Precisión Training Set | Precisión Test Set  |
| -------------|          -------------:|               -----:|
| 1R           | 91.37%                 |              20.29% |
| 2R           | 95.01%                 |              52.20% |
| 2R1R         | 98.82%                 |              54.53% |

La ejecución de este modelo podría ser dado por:
`python g01.py --arbol --holdout --porcentaje-pruebas 25 --poblacion 5000 --umbral-poda 0.1`


<a id="k-nearest-neighbors-utilizando-k-d-trees-1"></a>
### K-Nearest Neighbors utilizando K-d Trees:
Para este modelo el parámetro a cambiar corresponde a la cantidad de vecinos a buscar dado un ejemplo de entrada. Dicho parámetro será llamado de ahora en adelante `k`. Elegir el valor óptimo de `k` no es una tarea fácil. Normalmente se escoge un valor impar para `k`, para evitar empates a la hora de clasificar.
Un valor pequeño para `k`, permitirá que el ruido de la información tenga más influencia en los resultados.
Todas las pruebas para este modelo se realizaron con una población de **1000** votantes.
A continuación los resultados al realizar una prueba utilizando el algoritmo **Nearest Neighbor**, que consiste en buscar un único vecino, es decir que `k = 1`.
Para ejecutar esta prueba se utilizó el comando:
`python3 g01.py --knn --k 1 --poblacion 1000 --holdout --porcentaje-pruebas 25 --prefijo pruebaKnn`

Resultados:

| Ronda        | Precisión Training Set | Precisión Test Set  |
| -------------|          -------------:|               -----:|
| 1R           | 98.93                  |                20.0 |
| 2R           | 99.33                  |               53.59 |
| 2R1R         | 100.0                  |                49.2 |

Como se puede apreciar en la tabla, la precisión a la hora de evaluar el modelo con el mismo set con el que fue entrenado, fuy muy alta sin embargo a la hora de evaluarlo con el test set, la precisión se redujo considerablemente. Por lo tanto se dice que el modelo no es capaz de generalizar.

Es muy importante resaltar que la predicción para la `1R` **no** es binaria a diferencia de `2R` y `2R1R`. Lo cual explica su baja precisión.

Otra prueba realizada fue utilizando `k = sqrt(N)` es decir `k = sqrt(1000)`.
Para ejecutar esta prueba se utilizó el comando:
`python3 g01.py --knn --k 31 --poblacion 1000 --holdout --porcentaje-pruebas 25 --prefijo pruebaKnn`

Resultados:

| Ronda        | Precisión Training Set | Precisión Test Set  |
| -------------|          -------------:|               -----:|
| 1R           | 34.53                  |                24.0 |
| 2R           | 62.93                  |                62.8 |
| 2R1R         | 62.13                  |                58.80|

La precisión al evaluar el training set bajo, sin embargo al evaluar el test set la precisión subió.

Se realizaron pruebas con los siguientes `k`: 1, 31, 51, 101, 301, 501, 901. Los resultados de dichas pruebas se pueden visualizar en los siguientes gráficos.

![](https://i.imgur.com/gzlK5QP.png)

![](https://i.imgur.com/3FmetfT.png)

Como se aprecia en los gráficos la precisión a la hora de evaluar el test set mejora cuando el `k` es mayor a 1, pero decae al evaluar el training set. La precisión se estabiliza aunque se siga aumentando el `k`.

Como conclusión el modelo de K-Nearest Neighbors, al utilizar una muestra de 1000 votantes, nunca alcanza una precisión superior al 70%. 



<a id="redes-neuronales-1"></a>
### Redes Neuronales:
Las pruebas que se realizarán en las redes neuronales siempre serán con una población de 5000 y un 25% de esa población para pruebas. Los parámetros que se pueden cambiar en redes neuronales, son los que afectan la estructura como tal que serían --numero-capas y las --  --unidades-por-capa y el que afecta la función de activación de cada unidad en este caso –funcion-activacion. 

Primeramente, se harán pruebas que afecten la estructura por lo que la función de activación elegida en estas pruebas será relu. Se iniciará verificando cuantas unidades por capa dan mejores resultados, en estas pruebas sólo se utilizará una capa y diferentes unidades por capa.

Promediando algunos de los resultados se obtuvo los siguientes resultados.
Resultados cuando se esta usando 1 unidad por capa:

| Ronda        | Precisión Training Set | Precisión Test Set  |
| -------------|          -------------:|               -----:|
| 1R           | 24,91%                 |              25,28% |
| 2R           | 59,56%                 |              58,08% |
| 2R1R         | 59,56%                 |              58,08% |

Resultados cuando se está usando 10 unidades por capa:

| Ronda        | Precisión Training Set | Precisión Test Set  |
| -------------|          -------------:|               -----:|
| 1R           | 25,87%                 |              25,76% |
| 2R           | 63,74%                 |              61,14% |
| 2R1R         | 62,81%                 |              61,74% |

Resultados cuando se está usando 20 unidades por capa:

| Ronda        | Precisión Training Set | Precisión Test Set  |
| -------------|          -------------:|               -----:|
| 1R           | 30,34%                 |              26,86% |
| 2R           | 63,23%                 |              62,51% |
| 2R1R         | 63,81%                 |              62,87% |

Resultados cuando se está usando 30 unidades por capa:

| Ronda        | Precisión Training Set | Precisión Test Set  |
| -------------|          -------------:|               -----:|
| 1R           | 30,70%                 |              26,56% |
| 2R           | 64,25%                 |              61,54% |
| 2R1R         | 64,51%                 |              60,92% |

Tal vez con estas tablas no se logre ver que en el caso de la precisión del test set, esta empieza a subir y cuando llega a 30 baja, por lo que es probable que un buen número para unidades por capa sea 20 o un número cercano. Esto se verá mejor en los siguientes gráficos. 

![](https://i.imgur.com/oUBAYzB.png)

Pero no hay que confundir esto con que entre más unidades por capa peor resultado, porque no es cierto, si hacemos una prueba más usando 40 unidades por capa nos daremos cuenta de la verdad.

![](https://i.imgur.com/ycmepaZ.png)

Por lo que vemos que no sigue bajando y más bien podría empezar a subir.

Ahora nos toca probar la cantidad de capas, como el anterior usaremos relu como función de activación y usaremos 20 unidades por capa, la cantidad de muestras se mantiene igual que el porcentaje de muestras. La prueba de una capa y 20 unidades ya se realizó en las pruebas pasadas, por lo que esta vez se iniciara en 5 capas.
Resultados cuando se esta usando 1 capa:

| Ronda        | Precisión Training Set | Precisión Test Set  |
| -------------|          -------------:|               -----:|
| 1R           | 30,34%                 |              26,86% |
| 2R           | 63,23%                 |              62,51% |
| 2R1R         | 63,81%                 |              62,87% |

Resultados cuando se está usando 5 capas:

| Ronda        | Precisión Training Set | Precisión Test Set  |
| -------------|          -------------:|               -----:|
| 1R           | 25,27%                 |              25,56% |
| 2R           | 63,31%                 |              61,48% |
| 2R1R         | 63,05%                 |              61,30% |

Resultados cuando se está usando 10 capas:

| Ronda        | Precisión Training Set | Precisión Test Set  |
| -------------|          -------------:|               -----:|
| 1R           | 24,41%                 |              25,54% |
| 2R           | 60,91%                 |              59,36% |
| 2R1R         | 60,97%                 |              59,40% |

Resultados cuando se está usando 20 capas:

| Ronda        | Precisión Training Set | Precisión Test Set  |
| -------------|          -------------:|               -----:|
| 1R           | 25,29%                 |              25,18% |
| 2R           | 59,61%                 |              59,70% |
| 2R1R         | 59,61%                 |              59,70% |

Estos ejemplos parecen tener un comportamiento diferente a lo que pasaba con unidades por capa, porque en este en la precisión del test set parece que va bajando y no subiendo, por lo que se podría decir que para estos datos es mejor usar menos capas. En el gráfico también se puede apreciar cómo va decreciendo, pero no quiere decir que va a seguir bajando tan rápidamente, porque como se ve en el gráfico empieza a marcar como una asíntota con algún valor por debajo.

![](https://i.imgur.com/eJPHz26.png)

Ahora solo quedan los ejemplos correspondientes a la función de activación, para este informe se utilizarán 3 funciones de activación, relu que ya se ha estado usando, softmax y sigmoid. Para estas pruebas se mantendrán la cantidad de población y el porcentaje para pruebas, para mantener lo que ya se ha probado, usaremos 1 capa y 20 unidades por capa.

Resultados cuando se está usando relu:

| Ronda        | Precisión Training Set | Precisión Test Set  |
| -------------|          -------------:|               -----:|
| 1R           | 30,34%                 |              26,86% |
| 2R           | 63,23%                 |              62,51% |
| 2R1R         | 63,81%                 |              62,87% |

Resultados cuando se está usando softmax:

| Ronda        | Precisión Training Set | Precisión Test Set  |
| -------------|          -------------:|               -----:|
| 1R           | 24,83%                 |              23,70% |
| 2R           | 59,47%                 |              60,76% |
| 2R1R         | 59,47%                 |              60,76% |

Resultados cuando se está usando sigmoid:

| Ronda        | Precisión Training Set | Precisión Test Set  |
| -------------|          -------------:|               -----:|
| 1R           | 27,87%                 |              26,02% |
| 2R           | 64,05%                 |              61,54% |
| 2R1R         | 64,25%                 |              61,00% |

En este caso nos ayudaremos de 2 gráficos para ver cuál función de activación dio mejores resultados, primero comparando el training set.
![](https://i.imgur.com/p8Jd5pJ.png)
En este caso se podría decir que sigmoid muestra mejores resultados, pero claro se está usando el training set, si usamos el test set el gráfico sería el siguiente.
![](https://i.imgur.com/X6aofv8.png)
En este caso se ve claramente como el relu se muestra superior al usar el Test Set. Se debe aclarar que en este proyecto la función de activación en la capa de entrada y de salida se usa de forma predeterminada sigmoid, por ser la que dio mejores resultados en la predicción. 

Por otro lado, se debe hacer una aclaración con la función de pérdida utilizada en este proyecto, en todos los ejemplos aquí mostrados se usó la función de pérdida categorical_crossentropy, esto debido a recomendaciones encontradas en la documentación de Keras, que puedes encontrar en el siguiente [link](https://keras.io/losses/) al final de la página. Pero algunas personas también recomiendan binary_crossentropy por dar mejores resultados, luego de realizar algunas pruebas con el predictor, nos dimos cuenta de que, si bien mejora la precisión, la predicción no corresponde, para explicarlo mejor comparemos el mejor resultado que dio con categorical_crossentropy, con un resultado usando binary_crossentropy. 

Resultados cuando se está usando el Test Set:

| Ronda        | categorical_crossentropy| binary_crossentropy |
| -------------|          -------------:|               -----:|
| 1R           | 26,86%                 |              93,33% |
| 2R           | 62,51%                 |              80,10% |
| 2R1R         | 62,87%                 |              79,76% |

![](https://i.imgur.com/XSKyq4m.png)


Resultados cuando se está usando el Training Set:

| Ronda        | categorical_crossentropy| binary_crossentropy |
| -------------|          -------------:|               -----:|
| 1R           | 30,34%                 |              93,33% |
| 2R           | 63,23%                 |              81,54% |
| 2R1R         | 63,81%                 |              81,47% |

![](https://i.imgur.com/8T7NIHB.png)

Donde claramente se puede ver que con binary_crossentropy da muchos mejores resultados, pero a la hora de realizar predicciones, no coinciden, es decir, si la red neuronal tiene una precisión de más del 90% debería de acertar en la mayoría de las predicciones, pero luego de probarlo nos dimos cuenta de que esto no fue así, entonces por eso se decidió utilizar categorical_crossentropy, no solo por ser la recomendación de keras, si no por ser la que presentaba resultados más congruentes tanto en precisión, como a la hora de realizar la predicción.   

<a id="support-vector-machines-1"></a>
### Support Vector Machines:
El parámetro a cambiar en SVMs es el `kernel` a utilizar. La librería SciKit Learn, permite utilizar los siguientes `kernel`:
* `linear` 
* `poly` 
* `rbf`
* `sigmoid`

Además, al hacer una predicción multiclase como la de la primera ronda, hay que especificar un parámetro llamado `decision_function_shape` con valor `ovo` que corresponde a **one vs one**, un enfoque en el que si hay `n` clases, entonces se construyen `n * (n-1) / 2` clasificadores y cada uno se entrena con datos de dos clases.
A continuación los resultados al realizar una prueba para cada uno de los diferentes `kernel`:

<a id="linear"></a>
##### Linear
Para ejecutar esta prueba se utilizó el comando:
`python3 g01.py --svm --kernel linear --poblacion 1000 --holdout --porcentaje-pruebas 25 --prefijo pruebaSVM`

Resultados:

| Ronda        | Precisión Training Set | Precisión Test Set  |
| -------------|          -------------:|               -----:|
| 1R           | 40.8                  |                26.4 |
| 2R           | 64.66                  |                62.8 |
| 2R1R         | 65.47                  |                64.0|
<a id="polynomial"></a>
##### Polynomial
Para ejecutar esta prueba se utilizó el comando:
`python3 g01.py --svm --kernel poly --poblacion 1000 --holdout --porcentaje-pruebas 25 --prefijo pruebaSVM`

Resultados:

| Ronda        | Precisión Training Set | Precisión Test Set  |
| -------------|          -------------:|               -----:|
| 1R           | 28.13                  |                22.79 |
| 2R           | 60.26                  |                62.0 |
| 2R1R         | 60.26                  |                62.0|
<a id="rbf"></a>
##### RBF
Para ejecutar esta prueba se utilizó el comando:
`python3 g01.py --svm --kernel rbf --poblacion 1000 --holdout --porcentaje-pruebas 25 --prefijo pruebaSVM`

Resultados:

| Ronda        | Precisión Training Set | Precisión Test Set  |
| -------------|          -------------:|               -----:|
| 1R           | 26.93                  |                22.39 |
| 2R           | 61.06                  |                63.6 |
| 2R1R         | 61.06                  |                63.6|
<a id="sigmoid"></a>
##### Sigmoid
Para ejecutar esta prueba se utilizó el comando:
`python3 g01.py --svm --kernel sigmoid --poblacion 1000 --holdout --porcentaje-pruebas 25 --prefijo pruebaSVM`

Resultados:

| Ronda        | Precisión Training Set | Precisión Test Set  |
| -------------|          -------------:|               -----:|
| 1R           | 24.26                  |                18.79 |
| 2R           | 60.4                  |                61.6 |
| 2R1R         | 60.4                  |                61.6|


<a id="conclusi%C3%B3n"></a>
## Conclusión
Todos los modelos analizados en este proyecto, tuvieron una precisión inferior al 70%, lo cual no es para nada óptimo si se desea utilizar esta herramienta para un proyecto crítico. Entre las posibles causas para este fenomeno estan:

* Los datos tomados en cuenta para cada votante, no son determinantes para predecir el voto.
* Los datos generados tienen ruido. Por ejemplo, dos votantes diferentes, con los mismos datos e índices, votaron por partidos distintos.
* No se seleccionaron los parámetros óptimos para cada modelo a la hora de realizar pruebas. Por ejemplo, en el caso de K-Nearest Neighbors puede que la cantidad de votantes generados o bien el `k` ingresado como parámetro, no fuera el indicado para mejorar la precisión.

Respecto al rendimiento de los modelos utilizados, sin duda el de peor rendimiento fue K-Nearest Neighbors utilizando K-d Trees. Las pruebas realizadas con una población de 1000 votantes, tomaban aproximadamente 5 minutos, mientras que los demás modelos duraban segundos. Por su parte el modelo de árboles de decisión siempre fue el más veloz.

La variación de la precisión a la hora de cambiar el tamaño de la población de entrenamiento fue nula para todos los modelos. Las pruebas más sencillas se realizaron con 100 votantes, y las pruebas registradas en este informe fueron con 1000 o 5000 votantes. Otras pruebas no incluidas en este reporte fueron con 50000 votantes, pero los resultados fueron muy similares a los de las pruebas que se mostraron previamente. 

A nivel de implementación queda claro que frameworks como `TensorFlow`, `SciKit Learn` o `Keras` facilitan mucho la creación de modelos.


<a id="manual-de-uso"></a>
# Manual de uso:
<a id="instalaci%C3%B3n-de-los-requerimientos"></a>
## Instalación  de los requerimientos:
Este proyecto está desarrollado y probado en Python 3.6 con Ubuntu 17.10, es probable que funcione en otros sistemas operativos, pero en este manual se proporciona las instrucciones para Ubuntu, por otro lado, se aclara que no funciona en Python inferior a 3, por lo que primeramente se espera que ya el mismo esté instalado, si aun no lo tiene puede descargarlo de [aquí](https://www.python.org/downloads/).

Además, es necesario pip que es un sistema de gestión de paquetes, utilizado para instalar bibliotecas para Python, este cuenta con una versión compatible con Python 3, por lo que es esta la que se debe instalar.

Instalación para Ubuntu/Debian para python 3.x
> sudo apt-get install python3-pip

Una vez instalado pip se podrán instalar todas las bibliotecas necesarias para correr el proyecto, **se recomienda instalar estas bibliotecas con permisos de administrador**.  Estas bibliotecas son:

* tec.ic.ia.pc1 (biblioteca para la generación de la población necesaria, para el proyecto)
* SciPy con NumPy (para uso de arrays y compatibilidad con otras bibliotecas del proyecto)
* scikit-learn (ayuda en la creación de SVM, también es útil con Redes neuronales y regresión logística)
* keras (Para la creación de redes neuronales)
* tensorflow (Utilizado en regresión lineal y en redes neuronales como backend).
* Pandas (Utilizado para la normalización discretizada de los datos) 

<a id="teciciapc1"></a>
#### tec.ic.ia.pc1 

Esta biblioteca también fue creada por nosotros y también está en este mismo repositorio, pero se debe instalar por medio de pip, para su correcto funcionamiento en el proyecto, para esto se deben seguir las siguientes instrucciones.
1.	Se espera que ya se tenga clonado este repositorio
2.	Se debe ubicar dentro de la carpeta src.
3.	Se debe abrir una terminal en esta carpeta.
4.	Se ejecutará el siguiente comando: 
> pip3 install -e .  --ignore-installed

En la consola debería aparecer un mensaje que dice “Successfully installed tec”


<a id="scipy-con-numpy"></a>
#### [SciPy con NumPy](https://www.scipy.org/)

Primero se instalará SciPy, ésta comúnmente incluye a NumPy, pero para estar seguros también se instalará NumPy, para esto desde cualquier localización se necesita abrir una terminal y en ella escribir lo siguiente:

Para SciPy
> pip3 install scipy

Para NumPy
> pip3 install numpy

<a id="scikit-learn"></a>
#### [Scikit-learn](http://scikit-learn.org/stable/install.html)
Esta biblioteca tiene algunos requisitos, pero sí ha seguido el manual no debería de preocuparse, porque los requisitos son NumPy en una versión superior o igual a 1.8.2 y SciPy en una versión superior o igual a 0.13.3. Para instalar Scikit se debe escribir el siguiente comando en una terminal:

> pip3 install -U scikit-learn

<a id="keras"></a>
#### [Keras](https://keras.io/)
Keras es un API de alto nivel para crear redes neuronales para instalarlo se debe abrir una terminal y escribir el siguiente comando:

> pip3 install -U keras

<a id="tensorflow"></a>
#### [tensorflow](https://www.tensorflow.org/install/)
TensorFlow servirá de backend para ser utilizado por keras en las redes neuronales y además es utilizado por la regresión logística. Para su instalación se debe abrir una terminal y escribir lo siguiente:
> pip3 install -U tensorflow

<a id="pandas"></a>
#### [Pandas](https://pandas.pydata.org/)
En este proyecto se utilizó exclusivamente para hacer una normalización discretizada de algunos datos de nuestro set generado. Para instalarlo es necesario abrir una terminal y escribir lo siguiente:
> pip3 install -U pandas

<a id="uso-del-sistema"></a>
## Uso del sistema
Después de haber instalado todas las bibliotecas necesarias para el proyecto, solo queda poder usar el sistema, para esto se espera que ya está clonado este repositorio y se debe ubicar en la siguiente ruta:

> src/tec/ic/ia/p1

una vez aquí se debe abrir una terminal o bien abrirla donde sea e ir a la ruta que ya se dio. Luego de esto se debe escribir 

> python3 g01.py

pero este programa recibe varias banderas para su correcto funcionamiento y para decirle que deseas hacer, estas banderas son:

| Bandera                	| Explicación                                                                                                   	| Valores Posibles                                                                                    	|
|------------------------	|---------------------------------------------------------------------------------------------------------------	|-----------------------------------------------------------------------------------------------------	|
| - -poblacion           	| Se utiliza para seleccionar la cantidad de población, justo después de la bandera se debe escribir el número. 	| Números enteros positivo                                                                            	|
| - -porcentaje-pruebas  	| Es el porcentaje de pruebas que se guardará para la prueba final                                              	| Números mayor a 0 y menor a 100                                                                     	|
| - -provincia           	| Bandera que dirá que se quiere hacer el análisis por provincia                                                	| SAN JOSE, HEREDIA, ALAJUELA, CARTAGO, LIMON, PUNTARENAS, GUANACASTE                                 	|
| - -prefijo             	| Será el nombre que se le pondrá al archivo csv generado                                                       	| Cualquier nombre                                                                                    	|
| - -kfold               	| Si se escribe el programa se probara con kfold crossvalidation                                                	| True o False                                                                                        	|
| - -kfolds              	| Son los k grupos en los que se dividirá el set de entrenamiento                                               	| Número entero positivo                                                                              	|
| - -holdout             	| Si se escribe el programa se probará con holdout crossvalidation                                              	| True o False                                                                                        	|
| - -regresion-logistica 	| Activa la regresión logística                                                                                 	| True o False                                                                                        	|
| - -l1                  	| Para seleccionar el resultado de la función de pérdida L1                                                     	| Número entre 0 y 1.                                                                                 	|
| - -l2                  	| Para seleccionar el resultado de la función de pérdida L2                                                     	| Número entre 0 y 1.                                                                                 	|
| - -red-neuronal        	| Activa la red neuronal                                                                                        	| True o False                                                                                        	|
| - -numero-capas        	| Para seleccionar el número de capas en la red neuronal                                                        	| Numero entero positivo                                                                              	|
| - -unidades-por-capa   	| Para seleccionar el número de unidad por capa en la red neuronal                                              	| Número entero positivo                                                                              	|
| - -funcion-activacion  	| Para seleccionar la función de activación en la red neuronal                                                  	| softmax, softplus, relu, sigmoid, linear para más funciones de activación revisar el siguiente link 	|
| - -arbol               	| Activa el árbol de decisión                                                                                   	| True o False                                                                                        	|
| - -umbral-poda         	| Para seleccionar el umbral con el que se podara el árbol .                                                    	| Numero entre 0 y 1.                                                                                 	|
| - -knn                 	| Para activar K-Nearest Neighbors                                                                              	| True o False                                                                                        	|
| - -k                   	| Cantidad de vecinos que tomara en cuenta, para los cálculos.                                                  	| Número mayor que 0 y menor que el tamaño muestra.                                                   	|
| - -svm                 	| Activa Support Vector Machines                                                                                	| True o False                                                                                        	|
| - -kernel              	| Para seleccionar el kernel del SVM.                                                                           	| linear, polynomial, rbf, sigmoid                                                                    	|                                                                 	|


Una vez se entiendan estas banderas y junto al comando que se explicó antes se podrán ejecutar una red neuronal de una capa con 21 unidades por capa, con una función de activación relu, usando una población de 10000 y un 10% de pruebas, además creará un archivo csv llamado pruebaRedes de la siguiente forma:


> python3 g01.py --red-neuronal --numero-capas 1 --unidades-por-capa 21 --funcion-activacion relu --prefijo pruebaRedes --poblacion 10000 --porcentaje-pruebas 10 --holdout

Se espera que se ejecute sólo un modelo a la vez.
