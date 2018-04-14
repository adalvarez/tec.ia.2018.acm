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
    print("Totales canton")
    print(totales_canton)
    print("Cantones")
    print(cantones)
    print("Rangos partido canton")
    print(rangos_partido_canton)
    return totales_canton, cantones, rangos_partido_canton

def crear_estructura_votos_cantones_ronda_2(filename='./summaryJuntasRonda2.csv'):
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

# Retorna un partido o tipo de voto para un votante
def asignar_voto_ronda_2(rango_partidos, na=None):
    numero_aleatorio = na if na is not None else randint(1, rango_partidos[-1])
    tipo_voto = [        
        "ACCION CIUDADANA",
        "RESTAURACION NACIONAL",
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
    # votoRonda2 = 

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

generar_muestra_pais(100)

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
