#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn import svm
import csv
import numpy

#Valores posibles para el parametro kernel:
#linear polynomial, rbf, sigmoid


def readCSV(filename):
    with open(filename, 'r', encoding="ISO-8859-1") as f:
        reader = csv.reader(f)
        return list(reader)

#Recibe un set de entrenamiento y retorna una lista con las respuestas esperadas por cada ejemplo
def get_real_results(matrix):
  return [row[len(row)-1] for row in matrix]


#Recibe una matriz de ejemplos, un array de respuestas, retorna un modelo. 
#Valores para tipo: 'ovo' cuando es multiclass, 'ovr' cuando no
#Valores para kernel: 'linear', 'polynomial', 'rbf', 'sigmoid'

def generate_svm_model(ejemplos, respuestas, function_shape, tipo_kernel):
	
	#ejemplos = numpy.array(readCSV('./iris.csv')).astype('float')
	#print(ejemplos)
	

	#respuestas = readCSV('./iris_target.csv')

	
	#print("----------------------------------")
	#print(respuestas[0])

	X = ejemplos
	Y = respuestas
	if(function_shape == "ovr"):
		clf = svm.SVC(kernel=tipo_kernel)
	else:
		clf = svm.SVC(decision_function_shape=function_shape, kernel=tipo_kernel)
	clf.fit(X, Y)
	return clf
	


#Recibe un modelo y un ejemplo para predicir sobre dicho modelo
def svm_predict(ejemplo, modelo):
	return modelo.predict([ejemplo])[0]

#modelo = generate_svm_model()
#svm_predict([6.7,3.1,4.7,1.5], modelo)

