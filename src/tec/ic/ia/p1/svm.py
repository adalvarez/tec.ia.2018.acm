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

def svm_primera_ronda():
	
	ejemplos = numpy.array(readCSV('./iris.csv')).astype('float')
	print(ejemplos)
	

	respuestas = readCSV('./iris_target.csv')

	
	print("----------------------------------")
	print(respuestas[0])

	X = ejemplos.tolist()
	Y = respuestas[0]
	clf = svm.SVC(decision_function_shape='ovo')
	print(clf.fit(X, Y))
	

	print(clf.predict([[6.7,3.1,4.7,1.5]]))

svm_primera_ronda()

