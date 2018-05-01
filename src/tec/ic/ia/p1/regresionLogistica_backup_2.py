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
def lr_train_test(trainning_set, test_set, l1, l2):

	# Time
	start_time = time.time()

	# Obtiene la longitud de todo el dataset
	len_total_dataset = len(trainning_set) + len(test_set)

	# Learning rate
	learning_rate = 0.5

	# Se separa los labels
	trn_X = []
	trn_Y = []
	for i in trainning_set:
		trn_Y.append(i[-1])
		trn_X.append(i[:-1])
	trn_X = numpy.asarray(trn_X)
	trn_Y = numpy.asarray(trn_Y)

	# Se separa los labels
	tst_X = []
	tst_Y = []
	for i in test_set:
		tst_Y.append(i[-1])
		tst_X.append(i[:-1])
	tst_X = numpy.asarray(tst_X)
	tst_Y = numpy.asarray(tst_Y)

	# Trainning unique labels
	ul_trn = list(set(trn_Y))
	ul_labels_trn = to_categorical(ul_trn)

	ul_tst = list(set(tst_Y))
	ul_labels_tst = to_categorical(ul_tst)

	# Convierte el Y categorico a "binario"
	trn_Y = to_categorical(array(trn_Y))
	# Convierte el Y categorico a "binario"
	tst_Y = to_categorical(array(tst_Y))

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
			for l in ul_labels_trn:
				if(lr_checklabel(p,l)):
					for index,bit in enumerate(l):
						if(bit==1):
							trn_prd.append(index)

		tst_prd = []
		for p in test_prediction.eval():
			for l in ul_labels_tst:
				if(lr_checklabel(p,l)):
					for index,bit in enumerate(l):
						if(bit==1):
							tst_prd.append(index)

		# print(trn_prd)
		# print(tst_prd)

		return train_acc, test_acc, l, duration


# muestras = g01.readCSV('./Irisv2.csv')

# trn = numpy.array(muestras[:120]).astype(numpy.float32)
# tst = numpy.array(muestras[120:]).astype(numpy.float32)

# print(trn[0])

# print(lr_train_test(trn, tst, 0.9, 0.01))

# ------------------------

def stringToFloat(lista):
    for i in range(len(lista)):
        lista[i] = lista[i].replace(",","")
    return lista.astype(float)

muestras = g01.generar_muestra_pais(1000)

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

escolaridad = (muestrasT[8].astype(numpy.float32))[numpy.newaxis, :].T##transpuesta

##No se tomo en cuenta promedio de escolaridad
asistEducacion = label_encoder.fit_transform(muestrasT[9])[numpy.newaxis, :].T
trabajo = label_encoder.fit_transform(muestrasT[10])[numpy.newaxis, :].T
asegurado = label_encoder.fit_transform(muestrasT[11])[numpy.newaxis, :].T
extranjero = label_encoder.fit_transform(muestrasT[12])[numpy.newaxis, :].T
discapacitado = label_encoder.fit_transform(muestrasT[13])[numpy.newaxis, :].T
jefeHogar = label_encoder.fit_transform(muestrasT[14])

poblacion = pd.qcut(stringToFloat(muestrasT[15]).astype(numpy.float32), 5, labels=False)[numpy.newaxis, :].T##transpuesta
superficie = pd.qcut(stringToFloat(muestrasT[16]).astype(numpy.float32), 4, labels=False)[numpy.newaxis, :].T##transpuesta
densidad = pd.qcut(stringToFloat(muestrasT[17]).astype(numpy.float32), 4, labels=False)[numpy.newaxis, :].T##transpuesta
vOcupadas = pd.qcut(stringToFloat(muestrasT[18]).astype(numpy.float32), 4, labels=False)[numpy.newaxis, :].T##transpuesta
ocupantes = (muestrasT[19].astype(numpy.float32))[numpy.newaxis, :].T##transpuesta

#no se toma en cuenta poblacion total, superficie, densidad, viviendas ocupadas, promedio de ocupantes por vivienda
voto = label_encoder.fit_transform(muestrasT[20])[numpy.newaxis, :].T

#convertirlo en listas binarias
onehot_encoder = OneHotEncoder(sparse=False)
cantones = cantones.reshape(len(cantones), 1)
cantones = onehot_encoder.fit_transform(cantones)
jefeHogar = jefeHogar.reshape(len(jefeHogar), 1)
jefeHogar = onehot_encoder.fit_transform(jefeHogar)
# salida = numpy.concatenate((cantones,genero, edad, zona, dependiente, casaEstado, casaHacinada, alfabeta, escolaridad, asistEducacion, trabajo, asegurado, extranjero, discapacitado, jefeHogar, poblacion, superficie, densidad, vOcupadas, ocupantes, voto),axis=1)
salida = numpy.concatenate((cantones,genero, edad, zona, dependiente, casaEstado, casaHacinada, alfabeta, escolaridad, asistEducacion, trabajo, asegurado, extranjero, discapacitado, jefeHogar, poblacion, superficie, densidad, vOcupadas, ocupantes, voto),axis=1)


trn = salida[:800]
tst = salida[800:]

print(lr_train_test(trn, tst, 0.9, 0.01))
print("\n\n\n\n\n\n")

