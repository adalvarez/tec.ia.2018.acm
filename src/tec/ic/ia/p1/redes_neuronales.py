from tec.ic.ia.pc1 import g01
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from numpy import array
from numpy import argmax
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy
import tensorflow as tf
import pandas as pd
import csv




def redes_neuronales(XEntranamiento, YEntrenamiento, XEvaluacion, YEvaluacion, capas, unidadCapas, activacion):
	# create model
	model = Sequential()
	model.add(Dense(unidadCapas, input_dim=len(XEntranamiento[0]), activation=activacion))

	for i in range(capas):
		model.add(Dense(unidadCapas, activation = activacion))

	model.add(Dense(len(YEntrenamiento[0]), activation="sigmoid"))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])#categorical_crossentropy
	# Fit the model
	model.fit(XEntranamiento, YEntrenamiento, epochs = 50, batch_size=10)#, verbose=0)
	# evaluate the model
	scores = model.evaluate(XEntranamiento, YEntrenamiento)
	scores2 = model.evaluate(XEvaluacion, YEvaluacion)
	#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	predecir = numpy.concatenate((XEntranamiento,XEvaluacion), axis=0)
	ynew = model.predict_classes(predecir)
	predicciones = []
	# show the inputs and predicted outputs
	#print(ynew)
	
	for i in range(len(predecir)):
		#print("Predicted=%s" % (ynew[i]))
		predicciones.append(ynew[i])

	return numpy.asarray(predicciones), scores[1], scores2[1]

def separarXY(datos):
      #X = datos
      Y = []
      for i in datos:
        Y.append(i[-1])
        #numpy.delete(i, -1)
        del i[-1]

      return numpy.asarray(datos),numpy.asarray(Y)

def data_rn_rl_svm(muestras):
	#muestras = generar_muestra_pais(50000)

	salida = []
	muestrasT = numpy.transpose(muestras)
	#transformarlo a numeros
	label_encoder = LabelEncoder()
	min_max_scaler = preprocessing.MinMaxScaler()
	cantones = label_encoder.fit_transform(muestrasT[0])#no es transpuesta porque hay que volver a transformarla para hacerla binaria
	genero = label_encoder.fit_transform(muestrasT[1])[numpy.newaxis, :].T##transpuesta
	#normalizar edad
	edad = muestrasT[2].astype(float).reshape(-1,1)
	edad = min_max_scaler.fit_transform(edad)#ya queda esta transpuesto
	zona = label_encoder.fit_transform(muestrasT[3])[numpy.newaxis, :].T##transpuesta
	dependiente = label_encoder.fit_transform(muestrasT[4])[numpy.newaxis, :].T##transpuesta
	casaEstado = label_encoder.fit_transform(muestrasT[5])[numpy.newaxis, :].T##transpuesta
	casaHacinada = label_encoder.fit_transform(muestrasT[6])[numpy.newaxis, :].T##transpuesta
	alfabeta = label_encoder.fit_transform(muestrasT[7])[numpy.newaxis, :].T##transpuesta
	escolaridad = (muestrasT[8])[numpy.newaxis, :].T##transpuesta
	asistEducacion = label_encoder.fit_transform(muestrasT[9])[numpy.newaxis, :].T##transpuesta
	trabajo = label_encoder.fit_transform(muestrasT[10])[numpy.newaxis, :].T##transpuesta
	asegurado = label_encoder.fit_transform(muestrasT[11])[numpy.newaxis, :].T##transpuesta
	extranjero = label_encoder.fit_transform(muestrasT[12])[numpy.newaxis, :].T##transpuesta
	discapacitado = label_encoder.fit_transform(muestrasT[13])[numpy.newaxis, :].T##transpuesta
	jefeHogar = label_encoder.fit_transform(muestrasT[14])#no es transpuesta porque hay que volver a transformarla para hacerla binaria
	poblacion = pd.qcut(stringToFloat(muestrasT[15]), 5, labels=False)[numpy.newaxis, :].T##transpuesta
	superficie = pd.qcut(stringToFloat(muestrasT[16]), 4, labels=False)[numpy.newaxis, :].T##transpuesta
	densidad = pd.qcut(stringToFloat(muestrasT[17]), 4, labels=False)[numpy.newaxis, :].T##transpuesta
	vOcupadas = pd.qcut(stringToFloat(muestrasT[18]), 4, labels=False)[numpy.newaxis, :].T##transpuesta
	ocupantes = (muestrasT[19])[numpy.newaxis, :].T##transpuesta

	#voto = label_encoder.fit_transform(muestrasT[20])[numpy.newaxis, :].T##transpuesta
	voto1 = (muestrasT[20])[numpy.newaxis, :].T##transpuesta
	voto2 = (muestrasT[21])[numpy.newaxis, :].T##transpuesta

	#convertirlo en listas binarias
	onehot_encoder = OneHotEncoder(sparse=False)
	cantones = cantones.reshape(len(cantones), 1)
	cantones = onehot_encoder.fit_transform(cantones)
	jefeHogar = jefeHogar.reshape(len(jefeHogar), 1)
	jefeHogar = onehot_encoder.fit_transform(jefeHogar)

	salida = numpy.concatenate((cantones,genero, edad, zona, dependiente, casaEstado, casaHacinada, alfabeta, escolaridad, 
	                            asistEducacion, trabajo, asegurado, extranjero, discapacitado, jefeHogar, poblacion, superficie, 
	                            densidad, vOcupadas, ocupantes),axis=1)
	salida = salida.astype("float")
	salidaCopy = numpy.copy(salida)
	salida2 = agregarY(salidaCopy, voto1.tolist()) #datos con primera ronda
	salida3 = agregarY(salida, voto2.tolist()) #datos con segunda ronda
	#binarizar porque ahora va a ser parte de X
	voto1Binarizado = voto1[:]
	voto1Binarizado = label_encoder.fit_transform(voto1Binarizado)
	voto1Binarizado = voto1Binarizado.reshape(len(voto1Binarizado), 1)
	voto1Binarizado = onehot_encoder.fit_transform(voto1Binarizado)
	voto1Binarizado = numpy.asarray(voto1Binarizado)
	salida = numpy.array(salida , dtype="float")
	salida4 = numpy.concatenate((salida, voto1Binarizado), axis = 1)#datos mas primera ronda como X, mas segunda ronda como Y
	salida4 = salida4.astype("float")
	salida4 = agregarY(salida4, voto2.tolist())
	return salida2, salida3, salida4



def agregarY(lista, y):
	lista = lista.tolist()
	for i in range(len(lista)):
	    lista[i].append(y[i])

	return lista


def stringToFloat(lista):
    for i in range(len(lista)):
        lista[i] = lista[i].replace(",","")
    return lista.astype(float)

def createCSV(filename, data):
    my_file = open(filename, 'w')
    with my_file:
        writer = csv.writer(my_file, lineterminator='\n')
        writer.writerows(data)

def main():
	datos = g01.generar_muestra_pais(10000)
	r1,r2,r3 = data_rn_rl_svm(datos)
	X,Y = separarXY(r1)
	X = numpy.asarray(X)
	Y = numpy.asarray(Y)
	label_encoder = LabelEncoder()
	Y = label_encoder.fit_transform(Y)
	#print(Y)
	#onehot_encoder = OneHotEncoder(sparse=False)
	#voto = array(Y)
	#voto = voto.reshape(len(voto), 1)
	#voto = onehot_encoder.fit_transform(voto)
	voto = to_categorical(Y)

	prediccion, acc1, acc2 = redes_neuronales(X[:8000], voto[:8000], X[8000:], voto[8000:],8,15,'softmax')#softmax(malos resultados)-relu(buenos resultados)

	respuestaCategorica = label_encoder.inverse_transform(prediccion)

	archivo = numpy.concatenate((datos,respuestaCategorica[numpy.newaxis, :].T), axis=1)

	print(acc1)
	print(acc2)

	createCSV("prueba.csv", archivo)

main()
