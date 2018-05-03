from tec.ic.ia.pc1 import g01
from keras.models import Sequential
from keras.layers import Dense
from numpy import array
from numpy import argmax
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
import numpy
import tensorflow as tf
import time
import pandas as pd
import collections
import dataModifier

'''
Calcula el accuracy para un conjunto de predicciones y labels
'''
def lr_accuracy(predictions, labels):
	return (100.0 * numpy.sum(numpy.argmax(predictions,1) == numpy.argmax(labels,1))/predictions.shape[0])

'''
Verifica si una predicción coincide con un label
'''
def lr_checklabel(prediction, label):
	return (numpy.argmax([prediction],1) == numpy.argmax([label],1))[0]

'''
Retorna una tupla con:
	- Accuracy del set de entrenamiento
	- Accuracy del set de test
	- Pérdida
	- Modelo
	- Duración
'''
def lr_train_test(trn_X, trn_Y, tst_X, tst_Y, labels_Y, l1, l2):

	# Time
	start_time = time.time()

	# Learning rate
	learning_rate = 0.5

	test_data = numpy.array(tst_X).astype(numpy.float32)

	# Se crea un grafo
	graph = tf.Graph()
	with graph.as_default():

		# Beta para regularizacion
		# beta = .001
		# Número de atributos por entrenar
		nAtt = len(trn_X[0])
		# Número de líneas en entrenamiento
		nTraining = len(trn_X)
		# Número de etiquetas por clasificar
		num_labels = len(trn_Y[0])
		
		# Input data for the training data, we use placeholder that will be fed
		# at run time with a training minibatch
		tf_train_dataset = tf.placeholder(tf.float32, shape=(nTraining, nAtt))
		tf_train_labels = tf.placeholder(tf.float32, shape=(nTraining, num_labels))
		tf_test_dataset = tf.constant(test_data)

		# weights and baises for output/logit layer
		w_logit = tf.Variable(tf.truncated_normal([nAtt, num_labels]))
		b_logit = tf.Variable(tf.zeros([num_labels]))

		# Función que crea el modelo
		def model(data):
			return tf.matmul(data, w_logit) + b_logit

		# Training computations
		logits = model(tf_train_dataset)
		# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
		loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
		# regularized_loss = tf.nn.l2_loss(w_logit)
		# total_loss = loss + beta * regularized_loss

		# Optimizador
		optimizer = tf.train.FtrlOptimizer(
			learning_rate=learning_rate,
			l1_regularization_strength=l1,
			l2_regularization_strength=l2).minimize(loss)
		#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

		# Predicciones para training y test
		train_prediction = tf.nn.softmax(logits)
		test_prediction = tf.nn.softmax(model(tf_test_dataset))

	# Define un Session
	with tf.Session(graph=graph) as session:
		# Inicializa las variables del modelo
		tf.initialize_all_variables().run()
		
		for i in range(0,5000):
			# Se obtiene los datos para trainning: atributos y labels
			batch_data = numpy.array(trn_X).astype(numpy.float32)
			batch_labels = numpy.array(trn_Y).astype(numpy.float32)

			# Se prepara el feed_dict para alimentar a la ejecución
			feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}

			# Ejecución de optimizador, perdida y predicción de trainning
			_, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

			# Actualizar accuracy para trainnig y test.
			train_acc = lr_accuracy(predictions, batch_labels)
		test_acc = lr_accuracy(test_prediction.eval(), numpy.array(tst_Y).astype(numpy.float32))

		duration = time.time() - start_time

		trn_prd = []
		for p in predictions:
			for l in labels_Y:
				if(lr_checklabel(p,l)):
					for index,bit in enumerate(l):
						if(bit==1):
							trn_prd.append(index)

		tst_prd = []
		for p in test_prediction.eval():
			for l in labels_Y:
				if(lr_checklabel(p,l)):
					for index,bit in enumerate(l):
						if(bit==1):
							tst_prd.append(index)

		prd = numpy.concatenate((tst_prd,trn_prd), axis=0)

		return train_acc, test_acc, prd, duration


def data_rl(muestras):
	salida = []

	muestrasT = numpy.transpose(muestras)
	#transformarlo a numeros
	label_encoder = LabelEncoder()
	min_max_scaler = preprocessing.MinMaxScaler()
	cantones = label_encoder.fit_transform(muestrasT[0])
	genero = label_encoder.fit_transform(muestrasT[1])[numpy.newaxis, :].T
	#normalizar edad
	edad = muestrasT[2].astype(float).reshape(-1,1)
	edad = min_max_scaler.fit_transform(edad)

	zona = label_encoder.fit_transform(muestrasT[3])[numpy.newaxis, :].T
	dependiente = label_encoder.fit_transform(muestrasT[4])[numpy.newaxis, :].T
	casaEstado = label_encoder.fit_transform(muestrasT[5])[numpy.newaxis, :].T
	casaHacinada = label_encoder.fit_transform(muestrasT[6])[numpy.newaxis, :].T
	alfabeta = label_encoder.fit_transform(muestrasT[7])[numpy.newaxis, :].T

	escolaridad = (muestrasT[8].astype(numpy.float32))[numpy.newaxis, :].T

	##No se tomo en cuenta promedio de escolaridad
	asistEducacion = label_encoder.fit_transform(muestrasT[9])[numpy.newaxis, :].T
	trabajo = label_encoder.fit_transform(muestrasT[10])[numpy.newaxis, :].T
	asegurado = label_encoder.fit_transform(muestrasT[11])[numpy.newaxis, :].T
	extranjero = label_encoder.fit_transform(muestrasT[12])[numpy.newaxis, :].T
	discapacitado = label_encoder.fit_transform(muestrasT[13])[numpy.newaxis, :].T
	jefeHogar = label_encoder.fit_transform(muestrasT[14])

	poblacion = pd.qcut(stringToFloat(muestrasT[15]).astype(numpy.float32), 5, labels=False)[numpy.newaxis, :].T
	superficie = pd.qcut(stringToFloat(muestrasT[16]).astype(numpy.float32), 4, labels=False)[numpy.newaxis, :].T
	densidad = pd.qcut(stringToFloat(muestrasT[17]).astype(numpy.float32), 4, labels=False)[numpy.newaxis, :].T
	vOcupadas = pd.qcut(stringToFloat(muestrasT[18]).astype(numpy.float32), 4, labels=False)[numpy.newaxis, :].T
	ocupantes = (muestrasT[19].astype(numpy.float32))[numpy.newaxis, :].T

	voto1 = (muestrasT[20])[numpy.newaxis, :].T
	voto2 = (muestrasT[21])[numpy.newaxis, :].T

	onehot_encoder = OneHotEncoder(sparse=False)
	cantones = cantones.reshape(len(cantones), 1)
	cantones = onehot_encoder.fit_transform(cantones)
	jefeHogar = jefeHogar.reshape(len(jefeHogar), 1)
	jefeHogar = onehot_encoder.fit_transform(jefeHogar)

	salida1 = numpy.concatenate((cantones,genero, edad, zona, dependiente, casaEstado, casaHacinada, 
		alfabeta, escolaridad, asistEducacion, trabajo, asegurado, extranjero, discapacitado, 
		jefeHogar, poblacion, superficie, densidad, vOcupadas, ocupantes, voto1),axis=1)

	salida2 = numpy.concatenate((cantones,genero, edad, zona, dependiente, casaEstado, casaHacinada, 
		alfabeta, escolaridad, asistEducacion, trabajo, asegurado, extranjero, discapacitado, 
		jefeHogar, poblacion, superficie, densidad, vOcupadas, ocupantes, voto2),axis=1)

	voto1Binarizado = voto1[:]
	voto1Binarizado = label_encoder.fit_transform(voto1Binarizado)
	voto1Binarizado = to_categorical(voto1Binarizado)

	salida3 = numpy.concatenate((cantones,genero, edad, zona, dependiente, casaEstado, casaHacinada, 
		alfabeta, escolaridad, asistEducacion, trabajo, asegurado, extranjero, discapacitado, 
		jefeHogar, poblacion, superficie, densidad, vOcupadas, ocupantes, voto1Binarizado, voto2),axis=1)

	return salida1, salida2, salida3;



def stringToFloat(lista):
    for i in range(len(lista)):
        lista[i] = lista[i].replace(",","")
    return lista.astype(float)

def separarXY(datos):
      X = []
      Y = []
      for i in datos:
        Y.append(i[-1])
        X.append(i[:-1])
      return numpy.asarray(X), numpy.asarray(Y)

def main():
	datos = g01.generar_muestra_pais(50000)
	#r1,r2,r3 = data_rl(datos)
	r1,r2,r3 = dataModifier.data_rn_rl_svm(datos)

	print("Running R1")

	# R1
	r1_X,r1_Y = separarXY(r1)
	r1_X = r1_X.astype(numpy.float32)

	label_encoder = LabelEncoder()
	r1_Y = label_encoder.fit_transform(r1_Y)
	onehot_encoder = OneHotEncoder(sparse=False)

	r1_unique_labels = to_categorical(list(set(r1_Y)))

	r1_voto = to_categorical(r1_Y)

	train_acc, test_acc, prd, duration = lr_train_test(r1_X[:35000], r1_voto[:35000], r1_X[35000:], r1_voto[35000:], r1_unique_labels, 0.01, 0.01)

	print(train_acc, test_acc,duration)

	prd_class = label_encoder.inverse_transform(prd)

	print(collections.Counter(prd_class))

	print("Running R2")

	# R2
	r2_X,r2_Y = separarXY(r2)
	r2_X = r2_X.astype(numpy.float32)

	label_encoder = LabelEncoder()
	r2_Y = label_encoder.fit_transform(r2_Y)
	onehot_encoder = OneHotEncoder(sparse=False)

	r2_unique_labels = to_categorical(list(set(r2_Y)))

	r2_voto = to_categorical(r2_Y)

	train_acc, test_acc, prd, duration = lr_train_test(r2_X[:35000], r2_voto[:35000], r2_X[35000:], r2_voto[35000:], r2_unique_labels, 0.01, 0.01)

	print(train_acc, test_acc,duration)

	prd_class = label_encoder.inverse_transform(prd)

	print(collections.Counter(prd_class))

	print("Running R3")

	# R3
	r3_X,r3_Y = separarXY(r3)
	r3_X = r3_X.astype(numpy.float32)

	label_encoder = LabelEncoder()
	r3_Y = label_encoder.fit_transform(r3_Y)
	onehot_encoder = OneHotEncoder(sparse=False)

	r3_unique_labels = to_categorical(list(set(r3_Y)))

	r3_voto = to_categorical(r3_Y)

	train_acc, test_acc, prd, duration = lr_train_test(r3_X[:35000], r3_voto[:35000], r3_X[35000:], r3_voto[35000:], r3_unique_labels, 0.01, 0.01)

	print(train_acc, test_acc,duration)

	prd_class = label_encoder.inverse_transform(prd)

	print(collections.Counter(prd_class))

	#archivo = numpy.concatenate((datos,prd_class[numpy.newaxis, :].T), axis=1)

	#g01.createCSV("rl.csv", archivo)

#main()

