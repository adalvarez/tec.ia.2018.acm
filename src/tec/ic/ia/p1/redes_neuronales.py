#bibliotecas utilizadas
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.optimizers import SGD
from numpy import array
import numpy
import tensorflow as tf



#se encarga de realizar el modelo de redes neuronales, recibe los datos de entranamiento, de evaluacion
#el numero de capas, la cantidad de capas y la funcion de activacion que se desea utilizar
def redes_neuronales(XEntranamiento, YEntrenamiento, XEvaluacion, YEvaluacion, capas, unidadCapas, activacion):
	
	# crea el modelo secuencial
	model = Sequential()
	# crea la capa que se encargara de las entradas
	model.add(Dense(unidadCapas, input_dim=len(XEntranamiento[0]), activation=activacion))
	# crea las capas que se solicitaron para el modelo
	for i in range(capas):
		model.add(Dense(unidadCapas, activation = activacion))

	#crea la capa que se encarga de las salidas
	model.add(Dense(len(YEntrenamiento[0]), activation="sigmoid"))
	# compila el modelo, donde se define a funcion de perdida, el optimizador y la metrica
	# sgd = SGD(lr=0.01, momentum=0.8, decay=0.0, nesterov=False)
	model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])#categorical_crossentropy
	# entrena el modelo, aqui se definen los epochs que son las recursiones que hara y el batch sera 
	# parecido al learning rate de una funcion lineal
	model.fit(XEntranamiento, YEntrenamiento, epochs = 50, batch_size=50, verbose=0)
	# Se evalua el modelo con si mismo y con un set de evaluacion
	scores = model.evaluate(XEntranamiento, YEntrenamiento)
	scores2 = model.evaluate(XEvaluacion, YEvaluacion)

	#concatena todos los datos de x, para predecir los resultados
	predecir = numpy.concatenate((XEvaluacion,XEntranamiento), axis=0)
	yPredict = model.predict_classes(predecir)
	predicciones = []	
	for i in range(len(predecir)):
		#print("Predicted=%s" % (yPredict[i]))
		predicciones.append(yPredict[i])

	return numpy.asarray(predicciones), scores[1]*100, scores2[1]*100


