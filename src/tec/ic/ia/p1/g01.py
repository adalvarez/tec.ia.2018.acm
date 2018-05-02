#!/usr/bin/env python
# -*- coding: utf-8 -*-

from optparse import OptionParser
from tec.ic.ia.pc1 import g01
import crossValidation
import dataModifier
import numpy
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical


def separarXY(datos):
  #X = datos
  Y = []
  for i in datos:
    Y.append(i[-1])
    #numpy.delete(i, -1)
    del i[-1]

  return numpy.asarray(datos),numpy.asarray(Y)


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

parser.add_option("", "--kernel", dest="kernel", default="",
                  help="Kernel")

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
if options.knn or options.svm:
  #Se adaptan los datos
  datos1r, datos2r, datos2r1r = dataModifier.data_rn_rl_svm(datos)

elif options.rl or options.rn:
  datos1r, datos2r, datos2r1r = dataModifier.data_rn_rl_svm(datos)

  #Prepara los datos para 1R para redes neuronales
  X1,Y1 = separarXY(datos1r)
  X1R = numpy.asarray(X1)
  Y1 = numpy.asarray(Y1)
  label_encoder1 = LabelEncoder()
  Y1 = label_encoder1.fit_transform(Y1)
  
  
  #Prepara los datos para 2R para redes neuronales
  X2,Y2 = separarXY(datos2r)
  X2R = numpy.asarray(X2)
  Y2 = numpy.asarray(Y2)
  label_encoder2 = LabelEncoder()
  Y2 = label_encoder2.fit_transform(Y2)
  

  #Prepara los datos para 2R1R para redes neuronales 
  X3,Y3 = separarXY(datos2r1r)
  X2R1R = numpy.asarray(X3)
  Y3 = numpy.asarray(Y3)
  label_encoder3 = LabelEncoder()
  Y3 = label_encoder3.fit_transform(Y3)

  r1_unique_labels = None
  r2_unique_labels = None
  r3_unique_labels = None
  

  if options.rl:
    r1_unique_labels = to_categorical(list(set(Y1)))
    r2_unique_labels = to_categorical(list(set(Y2)))
    r3_unique_labels = to_categorical(list(set(Y3)))

  voto1R = to_categorical(Y1)
  voto2R = to_categorical(Y2)
  voto2R1R = to_categorical(Y3)




 
elif options.a:
  datos1r, datos2r, datos2r1r = dataModifier.data_dt(datos)


headers = ["Canton",
                "Genero",
                "Edad",
                "Zona(Urbana/Rural)",
                "Dependiente",
                "Casa en buen estado",
                "Casa Hacinada",
                "Alfabeta",
                "Promedio de escolaridad",
                "Asistencia a eduacion regular",
                "En la fuerza de trabajo",
                "Asegurado",
                "Poblacion nacida en el extranjero",
                "Discapacitado",
                "Jefe de hogar",
                "Poblacion Total(Canton)",
                "Superficie(Canton)",
                "Densidad(Canton)",
                "Viviendas ocupadas(Canton)",
                "Promedio de ocupantes por vivienda(Canton)",
                "Voto1R",
                "Voto2R",
                "Usado en entrenamiento",
                "Prediccion1R",
                "Prediccion2R",
                "Prediccion2R1R"]

es_entrenamiento = []
test_percentage = int(options.porcentaje_pruebas)
chunk_size = (len(datos) * test_percentage) // 100
es_entrenamiento += ['NO'] * chunk_size
es_entrenamiento += ['SI'] * (len(datos) - chunk_size)
es_entrenamiento = numpy.asarray(es_entrenamiento)

archivo = numpy.concatenate((datos,es_entrenamiento[numpy.newaxis, :].T), axis=1)


#Verificamos el tipo de CV solicitado y ejecutamos el CV correspondiente,
#CV hay q ejecutarlo 3 veces, uno para predicciones de 1R, otro para predicciones 2R y otro para 2R_1R

#Se hace kfold
if options.kf == True:
      
      validation_k = int(options.kfolds)

      
      #Se aplica cv para predecir 1r
      respuestas, fold_error_t, fold_error_v, final_error_t, final_error_v = crossValidation.k_fold_cross_validation(validation_k, test_percentage, datos1r, options, "1r")
      print("Ronda 1")
      print("FoldAccuracyT", 100 - fold_error_t)
      print("FoldAccuracyV", 100 -fold_error_v)
      print("FinalAccuracyT",100 - final_error_t)
      print("FinalAccuracyV",100 - final_error_v)
      print("--------------------------------------")
      respuestas = numpy.asarray(respuestas)

      archivo = numpy.concatenate((archivo,respuestas[numpy.newaxis, :].T), axis=1)
      #Se aplica cv para predecir 2r
      respuestas, fold_error_t, fold_error_v, final_error_t, final_error_v = crossValidation.k_fold_cross_validation(validation_k, test_percentage, datos2r, options, "2r")
      print("Ronda 2")
      print("FoldAccuracyT", 100 - fold_error_t)
      print("FoldAccuracyV", 100 - fold_error_v)
      print("FinalAccuracyT",100 - final_error_t)
      print("FinalAccuracyV",100 - final_error_v)
      print("--------------------------------------")
      respuestas = numpy.asarray(respuestas)

      archivo = numpy.concatenate((archivo, respuestas[numpy.newaxis, :].T), axis=1)
      #Se aplica cv para predecir 2r1r
      respuestas, fold_error_t, fold_error_v, final_error_t, final_error_v = crossValidation.k_fold_cross_validation(validation_k, test_percentage, datos2r1r, options, "2r1r")
      print("Ronda 3")
      print("FoldAccuracyT", 100 - fold_error_t)
      print("FoldAccuracyV", 100 -fold_error_v)
      print("FinalAccuracyT",100 - final_error_t)
      print("FinalAccuracyV",100 - final_error_v)
      respuestas = numpy.asarray(respuestas)

      archivo = numpy.concatenate((archivo, respuestas[numpy.newaxis, :].T), axis=1)

else: #Se hace holdout por defecto
      
      if options.rn or options.rl:

        #Realiza el modelo de 1r
        respuestas, accuracy_training, accuracy_validation = crossValidation.hold_out_cross_validation_rn(test_percentage, X1R.tolist(), voto1R.tolist(), options, r1_unique_labels)
        print("Ronda 1")
        print("AccuracyT", accuracy_training)
        print("AccuracyV", accuracy_validation)
        
        respuestaCategorica = label_encoder1.inverse_transform(respuestas)
        archivo = numpy.concatenate((archivo,respuestaCategorica[numpy.newaxis, :].T), axis=1)
        
        
        print("--------------------------------------")
        

        #Realiza el modelo de 2r
       
        respuestas, accuracy_training, accuracy_validation = crossValidation.hold_out_cross_validation_rn(test_percentage, X2R.tolist(), voto2R.tolist(), options, r2_unique_labels)
        print("Ronda 2")
        print("AccuracyT", accuracy_training)
        print("AccuracyV", accuracy_validation)
        
        
        respuestaCategorica = label_encoder2.inverse_transform(respuestas)
        archivo = numpy.concatenate((archivo,respuestaCategorica[numpy.newaxis, :].T), axis=1)
                
        print("--------------------------------------")
        #Realiza el modelo de 2r1r
       
        respuestas, accuracy_training, accuracy_validation = crossValidation.hold_out_cross_validation_rn(test_percentage, X2R1R.tolist(), voto2R1R.tolist(), options, r3_unique_labels)
        print("Ronda 3")
        print("AccuracyT", accuracy_training)
        print("AccuracyV", accuracy_validation)
        
        
        respuestaCategorica = label_encoder3.inverse_transform(respuestas)
        archivo = numpy.concatenate((archivo,respuestaCategorica[numpy.newaxis, :].T), axis=1)
        
        

      else:
        #Se aplica cv para predecir 1r
        print("Ronda 1")
        respuestas, error_t, error_v = crossValidation.hold_out_cross_validation(test_percentage, datos1r, options, "1r")
        print("AccuracyT",100 - error_t)
        print("AccuracyV",100 - error_v)
        respuestas = numpy.asarray(respuestas)
        archivo = numpy.concatenate((archivo,respuestas[numpy.newaxis, :].T), axis=1)
        #Se aplica cv para predecir 2r
        print("----------------------------------")
        print("Ronda 2")
        respuestas, error_t, error_v = crossValidation.hold_out_cross_validation(test_percentage, datos2r, options, "2r")
        print("AccuracyT",100 - error_t)
        print("AccuracyV",100 - error_v)
        respuestas = numpy.asarray(respuestas)
        archivo = numpy.concatenate((archivo,respuestas[numpy.newaxis, :].T), axis=1)
        #Se aplica cv para predecir 2r1r
        print("----------------------------------")
        print("Ronda 2r1r")
        respuestas, error_t, error_v =crossValidation.hold_out_cross_validation(test_percentage, datos2r1r, options, "2r1r")
        respuestas = numpy.asarray(respuestas)
        archivo = numpy.concatenate((archivo,respuestas[numpy.newaxis, :].T), axis=1)
        print("AccuracyT",100 - error_t)
        print("AccuracyV",100 - error_v)

headers = headers
archivo = archivo.tolist()
archivo_final = []
archivo_final.append(headers) 
archivo_final+= archivo

nombre_archivo = options.prefijo + ".csv"
g01.createCSV(nombre_archivo,archivo_final)


#GG