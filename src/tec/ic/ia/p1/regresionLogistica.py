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