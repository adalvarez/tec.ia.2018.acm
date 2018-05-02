from numpy import array
from numpy import argmax
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy
import pandas as pd

def stringToFloat(lista):
    for i in range(len(lista)):
        lista[i] = lista[i].replace(",","")
    return lista.astype(float)

##toma el set de datos y le aplica normalizacion,normalizacion discretizada y binariza los datos necesarios para que no haya datos categoricos
def data_rn_rl_svm(muestras):
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
    poblacion = pd.qcut(stringToFloat(muestrasT[15]), 5, labels=False, duplicates='drop')[numpy.newaxis, :].T##transpuesta
    superficie = pd.qcut(stringToFloat(muestrasT[16]), 4, labels=False, duplicates='drop')[numpy.newaxis, :].T##transpuesta
    densidad = pd.qcut(stringToFloat(muestrasT[17]), 4, labels=False, duplicates='drop')[numpy.newaxis, :].T##transpuesta
    vOcupadas = pd.qcut(stringToFloat(muestrasT[18]), 4, labels=False, duplicates='drop')[numpy.newaxis, :].T##transpuesta
    ocupantes = (muestrasT[19])[numpy.newaxis, :].T##transpuesta

    #voto = label_encoder.fit_transform(muestrasT[20])[numpy.newaxis, :].T##transpuesta
    voto1 = (muestrasT[20])[numpy.newaxis, :].T##transpuesta
    voto2 = (muestrasT[21])[numpy.newaxis, :].T##transpuesta

    #convertirlo en listas binarias
    onehot_encoder = OneHotEncoder(sparse=False)
    cantones = cantones.reshape(len(cantones), 1)
    cantones = onehot_encoder.fit_transform(cantones)
    jefeHogar = jefeHogar.reshape(len(jefeHogar), 1)
    jefeHogar = onehot_encoder.fit_transform(jefeHogar)

    salida = numpy.concatenate((cantones,genero, edad, zona, dependiente, casaEstado, casaHacinada, alfabeta, escolaridad, 
                                asistEducacion, trabajo, asegurado, extranjero, discapacitado, jefeHogar, poblacion, superficie, 
                                densidad, vOcupadas, ocupantes),axis=1)
    salida = salida.astype("float")
    salidaCopy = numpy.copy(salida)
    salida2 = agregarY(salidaCopy, voto1.T) #datos con primera ronda
    salida3 = agregarY(salida, voto2.T) #datos con segunda ronda
    #binarizar porque ahora va a ser parte de X
    voto1Binarizado = voto1[:]
    voto1Binarizado = label_encoder.fit_transform(voto1Binarizado)
    voto1Binarizado = voto1Binarizado.reshape(len(voto1Binarizado), 1)
    voto1Binarizado = onehot_encoder.fit_transform(voto1Binarizado)
    voto1Binarizado = numpy.asarray(voto1Binarizado)
    salida = numpy.array(salida , dtype="float")
    salida4 = numpy.concatenate((salida, voto1Binarizado), axis = 1)#datos mas primera ronda como X, mas segunda ronda como Y
    salida4 = salida4.astype("float")
    salida4 = agregarY(salida4, voto2.T)
    return salida2, salida3, salida4



def agregarY(lista, y):
    lista = lista.tolist()
    y = y[0].tolist()
    for i in range(len(lista)):
        lista[i].append(y[i])

    return lista

def data_dt(muestras):
    salida = []
    muestrasT = numpy.transpose(muestras)
    #transformarlo a numeros
    label_encoder = LabelEncoder()
    min_max_scaler = preprocessing.MinMaxScaler()
    cantones = (muestrasT[0])[numpy.newaxis, :].T#queda categorica
    genero = (muestrasT[1])[numpy.newaxis, :].T##dejar categorica
    #normalizar edad
    edad = numpy.asarray(pd.cut(stringToFloat(muestrasT[2]), 6, labels=["Nino", "Adolecente", "Joven", "Adulto Joven", "Adulto", "Adulto Mayor"], duplicates='drop')) [numpy.newaxis, :].T ##categotiza valor continuo
    zona = (muestrasT[3])[numpy.newaxis, :].T##queda categorizado
    dependiente = (muestrasT[4])[numpy.newaxis, :].T##queda categorizado
    casaEstado = (muestrasT[5])[numpy.newaxis, :].T##queda categorizado
    casaHacinada = (muestrasT[6])[numpy.newaxis, :].T##queda categorizado
    alfabeta = (muestrasT[7])[numpy.newaxis, :].T##queda categorizado
    escolaridad = numpy.asarray(pd.qcut(stringToFloat(muestrasT[8]), 5, labels=["Muy Baja","Baja", "Media", "Alta", "Muy Alta"], duplicates='drop'))[numpy.newaxis, :].T#(muestrasT[8])[numpy.newaxis, :].T##transpuesta
    asistEducacion = (muestrasT[9])[numpy.newaxis, :].T##queda categorizado
    trabajo = (muestrasT[10])[numpy.newaxis, :].T##queda categorizado
    asegurado = (muestrasT[11])[numpy.newaxis, :].T##queda categorizado
    extranjero = (muestrasT[12])[numpy.newaxis, :].T##queda categorizado
    discapacitado = (muestrasT[13])[numpy.newaxis, :].T##queda categorizado
    jefeHogar = (muestrasT[14])[numpy.newaxis, :].T#queda categorizada
    poblacion = numpy.asarray(pd.qcut(stringToFloat(muestrasT[15]), 4, labels=["Muy poco poblado", "Poco poblado", "Medianamente poblado", "Altamente Poblado"], duplicates='drop'))[numpy.newaxis, :].T##transpuesta
    superficie = numpy.asarray(pd.qcut(stringToFloat(muestrasT[16]), 5, labels=["Muy pequena","Pequena","Mediana", "Grande", "Muy Grande"], duplicates='drop'))[numpy.newaxis, :].T##transpuesta
    densidad = numpy.asarray(pd.qcut(stringToFloat(muestrasT[17]),4, labels=["Muy poco denso", "Poco denso", "Medianamente denso", "Muy denso"], duplicates='drop'))[numpy.newaxis, :].T##transpuesta
    vOcupadas = numpy.asarray(pd.qcut(stringToFloat(muestrasT[18]), 4, labels=["Muy pocas","Pocas","Regular","Muchas"], duplicates='drop'))[numpy.newaxis, :].T##transpuesta
    ocupantes = numpy.asarray(pd.qcut(stringToFloat(muestrasT[19]), 3, labels=["Bajo el promedio","promedio","Sobre el promedio"], duplicates='drop'))[numpy.newaxis, :].T##transpuesta

    #voto = label_encoder.fit_transform(muestrasT[20])[numpy.newaxis, :].T##transpuesta
    voto1 = (muestrasT[20])[numpy.newaxis, :].T##transpuesta
    voto2 = (muestrasT[21])[numpy.newaxis, :].T##transpuesta

    salida = numpy.concatenate((cantones,genero, edad, zona, dependiente, casaEstado, casaHacinada, alfabeta, escolaridad, 
                            asistEducacion, trabajo, asegurado, extranjero, discapacitado, jefeHogar, poblacion, superficie, 
                            densidad, vOcupadas, ocupantes),axis=1)
    salida2 = numpy.concatenate((salida, voto1), axis = 1) #datos con primera ronda
    salida3 = numpy.concatenate((salida, voto2), axis = 1) #datos con segunda ronda
    salida4 = numpy.concatenate((salida, voto1, voto2), axis = 1) #datos mas primera ronda como X, mas segunda ronda como Y

    return salida2.tolist(), salida3.tolist(), salida4.tolist()