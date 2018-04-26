#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
from random import randint

# Retorna un array con los rangos(basados en los votos) de cada canton y
# una lista con los nombres de los cantones


def leer_juntas():
    totales_canton, cantones, rangos_partido_canton = crear_estructura_votos_cantones()
    rangos_votos = obtener_rangos_votos_canton(totales_canton)
    return rangos_votos, cantones, rangos_partido_canton

# Lee un archivo csv y genera una lista, donde cada fila del archivo es
# una nueva lista.


def readCSV(filename):
    with open(filename, 'r', encoding="ISO-8859-1") as f:
        reader = csv.reader(f)
        return list(reader)

# Genera un archivo csv a partir de una lista de listas, cada lista es una
# fila.


def createCSV(filename, data):
    my_file = open(filename, 'w')
    with my_file:
        writer = csv.writer(my_file, lineterminator='\n')
        writer.writerows(data)

# Obtiene las estructura de los cantones
# Retorna los rangos de cantidad de votos por canton de una provincia
# dada. Para con ellos, elegir de forma aleatoria el canton del votante


def leer_juntas_provincia(provincia):
    totales_canton, cantones, rangos_partido_canton = crear_estructura_votos_cantones()
    return obtener_rangos_votos_provincia(
        totales_canton, cantones, provincia) + (rangos_partido_canton,)

# Retorna los rangos de cantidad de votos por canton de una provincia dada. Para con ellos, elegir de forma aleatoria el canton del votante
# Tambien retorna una lista con los cantones de una provincia dada


def obtener_rangos_votos_provincia(totales_canton, cantones_list, provincia):
    limitadores = {
        "SAN JOSE": [0, 20],
        "ALAJUELA": [20, 35],
        "CARTAGO": [35, 43],
        "HEREDIA": [43, 53],
        "GUANACASTE": [53, 64],
        "PUNTARENAS": [64, 75],
        "LIMON": [75, 81]
    }
    rangos_votos = []
    cantones = []
    nuevo_rango = 0
    limit_prov = limitadores[provincia]
    for indexCanton in range(limit_prov[0], limit_prov[1]):
        nuevo_rango += totales_canton[indexCanton]
        rangos_votos.append(nuevo_rango)
        cantones.append(cantones_list[indexCanton])
    return rangos_votos, cantones


# Retorna una lista, donde cada elemento es el total de votos de un canton
# Retorna una lista, con todos los cantones del pais.
# Retorna los rangos de cada partido, en cada canton.
def crear_estructura_votos_cantones(filename='./summaryJuntas.csv'):
    totales_canton = []
    cantones = []
    rangos_partido_canton = {}
    data = readCSV(filename)
    for i in range(1, len(data)):
        rangos_partido = []
        votos = 0
        for j in data[i][1:]:
            votos += int(j)
            rangos_partido.append(votos)
        rangos_partido_canton[data[i][0]] = rangos_partido
        cantones.append(data[i][0])
        totales_canton.append(sum([int(y) for y in data[i][1:]]))
    return totales_canton, cantones, rangos_partido_canton


# Retorna un array con los rangos(basados en los votos) de cada Canton
def obtener_rangos_votos_canton(totales_canton):
    rangos_votos = []
    nuevo_rango = 0
    for total_canton in totales_canton:
        nuevo_rango += total_canton
        rangos_votos.append(nuevo_rango)
    return rangos_votos


# Retorna un canton, segun un numero aleatorio, utilizando los rangos
# previamente generados
def asignar_canton(numero_aleatorio, rangos_votos, cantones):
    for k in range(len(rangos_votos)):
        if numero_aleatorio <= rangos_votos[k]:
            return cantones[k]

# Retorna una estructura con los indices de cada canton


def crear_estructura_indices_cantones(cantones):
    matriz_indices = readCSV("./indices.csv")
    indices_cantones = {}  # Estructura con los indices de desarrollo de cada canton
    for canton_i in range(len(cantones)):  # Recorremos los cantones
        # Aqui tenemos los indices de un solo canton
        indices_canton = matriz_indices[canton_i + 1]
        contenido_canton = {
            "poblacion": indices_canton[0],
            "superficie": indices_canton[1],
            "densidad": indices_canton[2],
            "porcentaje_poblacion_urbana": indices_canton[3],
            "hombres_mujeres": indices_canton[4],
            "dependencia_demografica": indices_canton[5],
            "viviendas_ocupadas": indices_canton[6],
            "promedio_ocupantes": indices_canton[7],
            "porcentaje_viviendas_buenas": indices_canton[8],
            "porcentaje_viviendas_hacinadas": indices_canton[9],
            "porcentaje_alfabetismo": indices_canton[10],
            "porcentaje_alfabetismo_10_24": indices_canton[11],
            "porcentaje_alfabetismo_25_mas": indices_canton[12],
            "promedio_escolaridad": indices_canton[13],
            "promedio_escolaridad_25_49": indices_canton[14],
            "promedio_escolaridad_50_mas": indices_canton[15],
            "porcentaje_educacion_regular": indices_canton[16],
            "porcentaje_educacion_regular_0_5": indices_canton[17],
            "porcentaje_educacion_regular_5_17": indices_canton[18],
            "porcentaje_educacion_regular_18_24": indices_canton[19],
            "porcentaje_educacion_regular_25_mas": indices_canton[20],
            "porcentaje_no_trabajo": indices_canton[21],
            "porcentaje_si_trabajo": indices_canton[22],
            "porcentaje_hombre_trabajo": indices_canton[23],
            "porcentaje_mujer_trabajo": indices_canton[24],
            "porcentaje_trabajo_no_asegurado": indices_canton[25],
            "porcentaje_extranjero": indices_canton[26],
            "porcentaje_discapacitado": indices_canton[27],
            "porcentaje_no_asegurado": indices_canton[28],
            "porcentaje_jefe_mujer": indices_canton[29],
            "porcentaje_jefe_compartido": indices_canton[30]}

        # Agregamos el nuevo canton con sus indices respectivos
        indices_cantones[cantones[canton_i]] = contenido_canton

    return indices_cantones


# Retorna el genero del votante, segun la cantidad de hombres por 100
# mujeres en el canton
def asignar_genero(numero_hombres, na=None):
    numero_hombres = float(numero_hombres)
    resultado = 100 + numero_hombres
    porcentaje_hombres = (numero_hombres * 100) / resultado
    numero_aleatorio = na if na is not None else randint(1, 100)
    if numero_aleatorio <= porcentaje_hombres:
        return "M"
    else:
        return "F"

 # Retorna una edad, tomando en cuenta los datos del INEC


def asignar_edad(na=None):
    numero_aleatorio = na if na is not None else randint(1, 3592463)
    if numero_aleatorio <= 3264515:
        return randint(18, 64)
    else:
        return randint(65, 105)


# Retorna un tipo de zona
def asignar_zona(porcentaje_poblacion_urbana, na=None):
    porcentaje_poblacion_urbana = float(porcentaje_poblacion_urbana)
    numero_aleatorio = na if na is not None else randint(1, 100)
    if numero_aleatorio <= porcentaje_poblacion_urbana:
        return "URBANA"
    else:
        return "RURAL"

# Retorna si un votante es dependiente o no


def asignar_dependencia(dependencia_demografica, na=None):
    dependencia_demografica = float(dependencia_demografica)
    resultado = 100 + dependencia_demografica
    porcentaje_dependiente = (dependencia_demografica * 100) / resultado
    numero_aleatorio = na if na is not None else randint(1, 100)
    if numero_aleatorio <= porcentaje_dependiente:
        return "SI"
    else:
        return "NO"

# Recibe cualquier indicador que sea un porcentaje y retorna si un votante
# cumple o no con dicho indicador


def asignar_por_porcentaje(porcentaje, na=None):
    porcentaje = float(porcentaje)
    numero_aleatorio = na if na is not None else randint(1, 100)
    if numero_aleatorio <= porcentaje:
        return "SI"
    else:
        return "NO"

# Retorna el tipo de jefe de hogar de un votante


def asignar_jefe_hogar(
        porcentaje_jefe_mujer,
        porcentaje_jefe_compartido,
        na=None):
    porcentaje_jefe_mujer = float(porcentaje_jefe_mujer)
    porcentaje_jefe_compartido = float(porcentaje_jefe_compartido)
    numero_aleatorio = na if na is not None else randint(1, 100)
    if numero_aleatorio <= porcentaje_jefe_mujer:
        return "MUJER"
    elif numero_aleatorio <= porcentaje_jefe_mujer + porcentaje_jefe_compartido:
        return "COMPARTIDO"
    else:
        return "HOMBRE"


# Estructura donde se guardaran los totales de votos por partido, despues
# de la generacion de votantes.
votos_final = {
    "ACCESIBILIDAD SIN EXCLUSION": 0,
    "ACCION CIUDADANA": 0,
    "ALIANZA DEMOCRATA CRISTIANA": 0,
    "DE LOS TRABAJADORES": 0,
    "FRENTE AMPLIO": 0,
    "INTEGRACION NACIONAL": 0,
    "LIBERACION NACIONAL": 0,
    "MOVIMIENTO LIBERTARIO": 0,
    "NUEVA GENERACION": 0,
    "RENOVACION COSTARRICENSE": 0,
    "REPUBLICANO SOCIAL CRISTIANO": 0,
    "RESTAURACION NACIONAL": 0,
    "UNIDAD SOCIAL CRISTIANA": 0,
    "NULO": 0,
    "BLANCO": 0}


# Retorna un partido o tipo de voto para un votante
def asignar_voto(rango_partidos, na=None):
    numero_aleatorio = na if na is not None else randint(1, rango_partidos[-1])
    tipo_voto = [
        "ACCESIBILIDAD SIN EXCLUSION",
        "ACCION CIUDADANA",
        "ALIANZA DEMOCRATA CRISTIANA",
        "DE LOS TRABAJADORES",
        "FRENTE AMPLIO",
        "INTEGRACION NACIONAL",
        "LIBERACION NACIONAL",
        "MOVIMIENTO LIBERTARIO",
        "NUEVA GENERACION",
        "RENOVACION COSTARRICENSE",
        "REPUBLICANO SOCIAL CRISTIANO",
        "RESTAURACION NACIONAL",
        "UNIDAD SOCIAL CRISTIANA",
        "NULO",
        "BLANCO"]
    for k in range(len(rango_partidos)):
        if numero_aleatorio <= rango_partidos[k]:
            votos_final[tipo_voto[k]] += 1
            return tipo_voto[k]

# Utiliza todas las funciones anteriores para retornar un votante


def generar_votante(
        rangos_votos,
        cantones,
        indices_cantones,
        rangos_partido_canton):
    canton = asignar_canton(
        randint(1, rangos_votos[-1]), rangos_votos, cantones)
    genero = asignar_genero(indices_cantones[canton]["hombres_mujeres"])
    edad = asignar_edad()
    zona = asignar_zona(
        indices_cantones[canton]["porcentaje_poblacion_urbana"])

    if edad > 64:
        dependencia = asignar_dependencia(
            indices_cantones[canton]["dependencia_demografica"])
    else:
        dependencia = "NO"

    casa_buen_estado = asignar_por_porcentaje(
        indices_cantones[canton]["porcentaje_viviendas_buenas"])
    casa_hacinada = asignar_por_porcentaje(
        indices_cantones[canton]["porcentaje_viviendas_hacinadas"])

    if edad <= 24:
        alfabeta = asignar_por_porcentaje(
            indices_cantones[canton]["porcentaje_alfabetismo_10_24"])
    else:
        alfabeta = asignar_por_porcentaje(
            indices_cantones[canton]["porcentaje_alfabetismo_25_mas"])

    if edad < 50:
        promedio_escolaridad = indices_cantones[canton]["promedio_escolaridad_25_49"]
    else:
        promedio_escolaridad = indices_cantones[canton]["promedio_escolaridad_50_mas"]

    if edad <= 24:
        asistencia_educacion_regular = asignar_por_porcentaje(
            indices_cantones[canton]["porcentaje_educacion_regular_18_24"])
    else:
        asistencia_educacion_regular = asignar_por_porcentaje(
            indices_cantones[canton]["porcentaje_educacion_regular_25_mas"])

    if genero == "M":
        fuerza_trabajo = asignar_por_porcentaje(
            indices_cantones[canton]["porcentaje_hombre_trabajo"])
    else:
        fuerza_trabajo = asignar_por_porcentaje(
            indices_cantones[canton]["porcentaje_mujer_trabajo"])

    if fuerza_trabajo == "SI":
        asegurado = asignar_por_porcentaje(
            100 - float(indices_cantones[canton]["porcentaje_trabajo_no_asegurado"]))
    else:
        asegurado = asignar_por_porcentaje(
            100 - float(indices_cantones[canton]["porcentaje_no_asegurado"]))

    extranjero = asignar_por_porcentaje(
        indices_cantones[canton]["porcentaje_extranjero"])
    discapacitado = asignar_por_porcentaje(
        indices_cantones[canton]["porcentaje_discapacitado"])

    jefe_hogar = asignar_jefe_hogar(
        indices_cantones[canton]["porcentaje_jefe_mujer"],
        indices_cantones[canton]["porcentaje_jefe_compartido"])

    poblacion_total = indices_cantones[canton]["poblacion"]
    superficie = indices_cantones[canton]["superficie"]
    densidad = indices_cantones[canton]["densidad"]
    viviendas_ocupadas = indices_cantones[canton]["viviendas_ocupadas"]
    promedio_ocupantes = indices_cantones[canton]["promedio_ocupantes"]

    voto = asignar_voto(rangos_partido_canton[canton])

    votante = [
        canton,
        genero,
        edad,
        zona,
        dependencia,
        casa_buen_estado,
        casa_hacinada,
        alfabeta,
        promedio_escolaridad,
        asistencia_educacion_regular,
        fuerza_trabajo,
        asegurado,
        extranjero,
        discapacitado,
        jefe_hogar,
        poblacion_total,
        superficie,
        densidad,
        viviendas_ocupadas,
        promedio_ocupantes,
        voto]
    return votante

# Retorna una muestra de n votantes, de todos los cantones del pais


def generar_muestra_pais(n):
    print("Generar muestra país")
    rangos_votos, cantones, rangos_partido_canton = leer_juntas()
    indices_cantones = crear_estructura_indices_cantones(cantones)
    votantes = []
    for i in range(n):
        votantes.append(
            generar_votante(
                rangos_votos,
                cantones,
                indices_cantones,
                rangos_partido_canton))
    return votantes

# Retorna una muestra de n votantes de los cantones de una provincia dada
def generar_muestra_provincia(n, provincia):
    print("Generar muestra provincia")
    rangos_votos, cantones, rangos_partido_canton = leer_juntas_provincia(
        provincia)
    indices_cantones = crear_estructura_indices_cantones(cantones)
    votantes = []
    for i in range(n):
        votantes.append(
            generar_votante(
                rangos_votos,
                cantones,
                indices_cantones,
                rangos_partido_canton))
    return votantes

# Create your first MLP in Keras
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

# fix random seed for reproducibility
numpy.random.seed(7)

def prueba(n, capas, unidadCapas, activacion):
	X = sustituir()
	Y = []
	for i in X:
	    Y.append(i[-1])
	    del i[-1]
	X = numpy.asarray(X)
	Y = numpy.asarray(Y)
	#dataset = numpy.loadtxt("refactorizado.csv", delimiter=",")
	# split into input (X) and output (Y) variables
	#X = dataset[:,0:19]
	#Y = dataset[:,19]
	Y = array(Y)
	encoded = to_categorical(Y)
	print("Elemetos de x")
	print(X)
	print("Elemetos de y")
	print(encoded)
	# create model
	model = Sequential()
	model.add(Dense(unidadCapas, input_dim=19, activation=activacion))

	for i in range(capas):
	    model.add(Dense(unidadCapas, activation = activacion))

	model.add(Dense(len(encoded[0]), activation="softmax"))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# Fit the model
	model.fit(X, encoded, epochs=150, batch_size=10)
	# evaluate the model
	scores = model.evaluate(X, encoded)
	print(scores)
	print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


def sustituir():
    muestras = generar_muestra_pais(50000)

    salida = []
    """
    datosCambiantes = {"SI":1, "NO":0, "URBANA":0, "RURAL":1, "M":1, "F":0, "MUJER":1, "HOMBRE":0, "COMPARTIDO":2, 
    					"RESTAURACION NACIONAL":0,"ACCION CIUDADANA":1,"LIBERACION NACIONAL":2,"UNIDAD SOCIAL CRISTIANA":3,
    					"INTEGRACION NACIONAL":4,"REPUBLICANO SOCIAL CRISTIANO":5,"MOVIMIENTO LIBERTARIO":6,"FRENTE AMPLIO":7,
    					"NUEVA GENERACION":8,"ALIANZA DEMOCRATA CRISTIANA":9,"RENOVACION COSTARRICENSE":10,"ACCESIBILIDAD SIN EXCLUSION":11,
    					"DE LOS TRABAJADORES":12,"NULO":13,"BLANCO":14}
    """
    #print(muestras)
    muestrasT = numpy.transpose(muestras)
    #print(muestrasT)
    #transformarlo a numeros
    label_encoder = LabelEncoder()
    min_max_scaler = preprocessing.MinMaxScaler()
    cantones = label_encoder.fit_transform(muestrasT[0])
    genero = label_encoder.fit_transform(muestrasT[1])[numpy.newaxis, :].T##transpuesta
    #normalizar edad
    edad = muestrasT[2].astype(float).reshape(-1,1)
    edad = min_max_scaler.fit_transform(edad)
    #edad = edad.T

    zona = label_encoder.fit_transform(muestrasT[3])[numpy.newaxis, :].T##transpuesta
    dependiente = label_encoder.fit_transform(muestrasT[4])[numpy.newaxis, :].T##transpuesta
    casaEstado = label_encoder.fit_transform(muestrasT[5])[numpy.newaxis, :].T##transpuesta
    casaHacinada = label_encoder.fit_transform(muestrasT[6])[numpy.newaxis, :].T##transpuesta
    alfabeta = label_encoder.fit_transform(muestrasT[7])[numpy.newaxis, :].T##transpuesta
    ##No se tomo en cuenta promedio de escolaridad
    asistEducacion = label_encoder.fit_transform(muestrasT[9])[numpy.newaxis, :].T##transpuesta
    trabajo = label_encoder.fit_transform(muestrasT[10])[numpy.newaxis, :].T##transpuesta
    asegurado = label_encoder.fit_transform(muestrasT[11])[numpy.newaxis, :].T##transpuesta
    extranjero = label_encoder.fit_transform(muestrasT[12])[numpy.newaxis, :].T##transpuesta
    discapacitado = label_encoder.fit_transform(muestrasT[13])[numpy.newaxis, :].T##transpuesta
    jefeHogar = label_encoder.fit_transform(muestrasT[14])
    #no se toma en cuenta poblacion total, superficie, densidad, viviendas ocupadas, promedio de ocupantes por vivienda
    voto = label_encoder.fit_transform(muestrasT[20])[numpy.newaxis, :].T##transpuesta
    
    #convertirlo en listas binarias
    onehot_encoder = OneHotEncoder(sparse=False)
    cantones = cantones.reshape(len(cantones), 1)
    cantones = onehot_encoder.fit_transform(cantones)
    jefeHogar = jefeHogar.reshape(len(jefeHogar), 1)
    jefeHogar = onehot_encoder.fit_transform(jefeHogar)
    #voto = voto.reshape(len(voto), 1)
    #voto = onehot_encoder.fit_transform(voto)
    #print(onehot_encoded)
    salida = numpy.concatenate((cantones,genero, edad, zona, dependiente, casaEstado, casaHacinada, alfabeta, asistEducacion, trabajo, asegurado, extranjero, discapacitado, jefeHogar, voto),axis=1)
    print(salida)
    print(len(salida[0]))
    """
    print(cantones)
    print(genero)
    print(len(edad))
    print(len(zona))
    print(len(dependiente))
    print(len(casaEstado))
    print(len(casaHacinada))
    print(len(alfabeta))
    print(len(asistEducacion))
    print(len(trabajo))
    print(len(asegurado))
    print(len(extranjero))
    print(len(discapacitado))
    print(len(jefeHogar))
    print(len(voto))
    """
    """
    for filas in range(len(muestras)):
    	salida.append([])
    	for i in range(len(muestras[filas]),1):
    		if (muestras[filas][i] in datosCambiantes):
    			salida[filas].append(datosCambiantes[muestras[filas][i]])
    		else:
    			if (isinstance (muestras[filas][i], str)):
    				salida[filas].append(muestras[filas][i].replace(',',''))
    			else:
    				salida[filas].append(muestras[filas][i])
    """
    createCSV("refactorizado.csv",muestras)
    return salida


sustituir()
#prueba(500, 5,8,'softmax')

