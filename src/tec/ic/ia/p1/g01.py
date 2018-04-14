#!/usr/bin/env python
# -*- coding: utf-8 -*-

from optparse import OptionParser

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


parser.add_option("", "--prefijo", dest="prefijo", default="",
                  help="Prefijo")

parser.add_option("", "--poblacion", dest="poblacion", default=0,
                  help="Poblacion")

parser.add_option("", "--porcentaje-pruebas", dest="porcentaje_pruebas", default=0,
                  help="Porcentaje de pruebas")

(options, args) = parser.parse_args()

print(options)