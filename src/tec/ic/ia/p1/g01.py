#!/usr/bin/env python
# -*- coding: utf-8 -*-

from optparse import OptionParser
from tec.ic.ia.pc1 import g01
from keras.models import Sequential
from keras.layers import Dense
from numpy import array
from numpy import argmax
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy
import tensorflow as tf
import pandas as pd
import crossValidation
import dataModifier

'''
def main():
      print("en redes")
      datos = g01.generar_muestra_pais(50000)
      datos = monstruorizarDatos(datos)
      redes_neuronales(datos[:40000],datos[40000:], 5,8,'softmax')
'''

##toma el set de datos y le aplica normalizacion,normalizacion discretizada y binariza los datos necesarios para que no haya datos categoricos
def monstruorizarDatos(muestras):
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
    poblacion = pd.qcut(stringToFloat(muestrasT[15]), 10, labels=False)[numpy.newaxis, :].T##transpuesta
    superficie = pd.qcut(stringToFloat(muestrasT[16]), 4, labels=False)[numpy.newaxis, :].T##transpuesta
    densidad = pd.qcut(stringToFloat(muestrasT[17]), 4, labels=False)[numpy.newaxis, :].T##transpuesta
    vOcupadas = pd.qcut(stringToFloat(muestrasT[18]), 4, labels=False)[numpy.newaxis, :].T##transpuesta
    ocupantes = (muestrasT[19])[numpy.newaxis, :].T##transpuesta

    voto = label_encoder.fit_transform(muestrasT[20])[numpy.newaxis, :].T##transpuesta
    
    #convertirlo en listas binarias
    onehot_encoder = OneHotEncoder(sparse=False)
    cantones = cantones.reshape(len(cantones), 1)
    cantones = onehot_encoder.fit_transform(cantones)
    jefeHogar = jefeHogar.reshape(len(jefeHogar), 1)
    jefeHogar = onehot_encoder.fit_transform(jefeHogar)
    salida = numpy.concatenate((cantones,genero, edad, zona, dependiente, casaEstado, casaHacinada, alfabeta, escolaridad, 
                                asistEducacion, trabajo, asegurado, extranjero, discapacitado, jefeHogar, poblacion, superficie, 
                                densidad, vOcupadas, ocupantes,  voto),axis=1)
    return salida

def stringToFloat(lista):
    for i in range(len(lista)):
        lista[i] = lista[i].replace(",","")
    return lista.astype(float)

def redes_neuronales(entrenar, evaluar, capas, unidadCapas, activacion):
      #datos de entranamiento
      XEntranamiento, YEntrenamiento = separarXY(entrenar)
      #datos para evaluar
      XEvaluacion, YEvaluacion = separarXY(evaluar)
      ##convertirlos a numpy array
      XEntranamiento = numpy.asarray(XEntranamiento)
      YEntrenamiento = numpy.asarray(YEntrenamiento)
      XEvaluacion = numpy.asarray(XEvaluacion)
      YEvaluacion = numpy.asarray(YEvaluacion)

      #binarizo los datos de los YEntrenamiento
      onehot_encoder = OneHotEncoder(sparse=False)
      votoEntrenar = array(YEntrenamiento)
      votoEntrenar = votoEntrenar.reshape(len(votoEntrenar), 1)
      votoEntrenar = onehot_encoder.fit_transform(votoEntrenar)
      #binarizo los datos de los YEvalucion
      votoEvaluacion = array(YEvaluacion)
      votoEvaluacion = votoEvaluacion.reshape(len(votoEvaluacion), 1)
      votoEvaluacion = onehot_encoder.fit_transform(votoEvaluacion)

      # create model
      model = Sequential()
      model.add(Dense(unidadCapas, input_dim=len(XEntranamiento[0]), activation=activacion))

      for i in range(capas):
        model.add(Dense(unidadCapas, activation = activacion))

      model.add(Dense(len(votoEntrenar[0]), activation="softmax"))
      # Compile model
      model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
      # Fit the model
      model.fit(XEntranamiento, votoEntrenar, epochs=2, batch_size=10)
      # evaluate the model
      scores = model.evaluate(XEvaluacion, votoEvaluacion)
      print(scores)
      print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

def separarXY(datos):
      X = datos[:]
      Y = []
      for i in X:
        Y.append(i[-1])
        numpy.delete(i, -1)

      return X,Y


parser = OptionParser()

parser.add_option("", "--regresion-logistica",
                  action="store_true", dest="rl", default=False,
                  help="Regresión logística")

parser.add_option("", "--l1", dest="l1", default=0,
                  help="L1")

parser.add_option("", "--l2", dest="l2", default=0,
                  help="L2")

parser.add_option("", "--red-neuronal",
                  action="store_true", dest="rn", default=False,
                  help="Red neuronal")

parser.add_option("", "--numero-capas", dest="nc", default=0,
                  help="Número de capas")

parser.add_option("", "--unidades-por-capa", dest="uc", default=0,
                  help="Unidades por capa")

parser.add_option("", "--funcion-activacion", dest="fa", default=0,
                  help="Función de activación")

parser.add_option("", "--arbol",
                  action="store_true", dest="a", default=False,
                  help="Árbol de decisión")

parser.add_option("", "--umbral-poda", dest="up", default=0,
                  help="Umbral poda")

parser.add_option("", "--knn",
                  action="store_true", dest="knn", default=False,
                  help="KNN")

parser.add_option("", "--k", dest="k", default=0,
                  help="K para KNN")

parser.add_option("", "--svm",
                  action="store_true", dest="svm", default=False,
                  help="SVM")


parser.add_option("", "--prefijo", dest="prefijo", default="",
                  help="Prefijo")

parser.add_option("", "--provincia", dest="prov", default="",
                  help="Provincia")

parser.add_option("", "--poblacion", dest="poblacion", default=0,
                  help="Poblacion")


parser.add_option("", "--kfold",
                  action="store_true", dest="kf", default=False,
                  help="KFOLD-CV")

parser.add_option("", "--holdout",
                  action="store_true", dest="ho", default=False,
                  help="HOLDOUT-CV")


parser.add_option("", "--kfolds", dest="kfolds", default=0,
                  help="Cantidad de KFolds")

parser.add_option("", "--porcentaje-pruebas", dest="porcentaje_pruebas", default=0,
                  help="Porcentaje de pruebas")

(options, args) = parser.parse_args()



#Generamos los datos utilizando nuestro simulador tomando en cuenta la cantidad de votantes solicitados
#en la bandera poblacion

#Se generan datos para todo el pais
if options.prov == "":
  datos = g01.generar_muestra_pais(int(options.poblacion))
else:
  datos = g01.generar_muestra_provincia(int(options.poblacion), options.prov)



#datos1r tendra los datos para el modelo q predice 1R
#datos2r tendra los datos para el modelo que predice 2R
#datos2r1r tendra los datos para el modelo que predice 2R_1R

#Se les da a los datos el formato necesario dependiendo del tipo de modelo solicitado
if options.rl or options.rn or options.knn or options.svm:
  #Se adaptan los datos
  datos1r, datos2r, datos2r1r = dataModifier.data_rn_rl_svm(datos)
  
elif options.a:
  datos1r, datos2r, datos2r1r = dataModifier.data_dt(datos)
  print("Datos1r")
  print(datos1r)
  print("------------------------------------------------------------------")
  print("Datos2r")
  print(datos2r)
  print("------------------------------------------------------------------")
  print("Datos2r1r")
  print(datos2r1r)      


#Verificamos el tipo de CV solicitado y ejecutamos el CV correspondiente,
#CV hay q ejecutarlo 3 veces, uno para predicciones de 1R, otro para predicciones 2R y otro para 2R_1R

#Se hace kfold
if options.kf == True:
      test_percentage = int(options.porcentaje_pruebas)
      validation_k = int(options.kfolds)
      #Se aplica cv para predecir 1r
      crossValidation.k_fold_cross_validation(validation_k, test_percentage, datos1r, options, "1r")
      #Se aplica cv para predecir 2r
      crossValidation.k_fold_cross_validation(validation_k, test_percentage, datos2r, options, "2r")
      #Se aplica cv para predecir 2r1r
      crossValidation.k_fold_cross_validation(validation_k, test_percentage, datos2r1r, options, "2r1r")


else: #Se hace holdout por defecto
      test_percentage = int(options.porcentaje_pruebas)
      #Se aplica cv para predecir 1r
      crossValidation.hold_out_cross_validation(test_percentage, datos1r, options, "1r")
      #Se aplica cv para predecir 2r
      crossValidation.hold_out_cross_validation(test_percentage, datos2r, options, "2r")
      #Se aplica cv para predecir 2r1r
      crossValidation.hold_out_cross_validation(test_percentage, datos2r1r, options, "2r1r")


#Cuando ya se tengan los resultados de cada CV, se genera el informe.

#GG