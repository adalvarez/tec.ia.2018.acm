#!/usr/bin/env python
# -*- coding: utf-8 -*-

from optparse import OptionParser
from tec.ic.ia.pc1 import g01
import crossValidation
import dataModifier

'''
def main():
      print("en redes")
      datos = g01.generar_muestra_pais(50000)
      datos = monstruorizarDatos(datos)
      redes_neuronales(datos[:40000],datos[40000:], 5,8,'softmax')
'''


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