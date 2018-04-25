import math
import random

##retorna el diccionario con la frecuencia
def frecuencia(data, frecuencias, index):

    for fila in data:
        if (fila[index] in frecuencias):
            frecuencias[fila[index]] += 1.0 
        else:
            frecuencias[fila[index]] = 1.0
    return frecuencias

#buscar los atributos mas comunes
def mayoria(attributes, data, target):
    valFreq = {}
    #buscar el target
    index = attributes.index(target)
    #calcular la frecuencia de los datos
    valFreq = frecuencia(data, valFreq, index)
    #no hay mayoria
    mayoriaA = True
    max = 0.0
    major = ""
    for key in valFreq.keys():
        if valFreq[key]>max:
            max = valFreq[key]
            major = key
            mayoriaA = True
        elif valFreq[key] == max:
            mayoriaA = False

    ##no hay mayoria entonces se retorna un random
    if not mayoriaA:
        valores = list(valFreq.keys())
        rand = random.choice(valores)
        return rand

    return major

#calcula la entropia de los datos
def calcularEntropia(attributes, data, Attr):
    valFreq = {}
    entropia = 0.0
    #selecciona index inicial
    i = 0

    for valor in attributes:
        if (Attr == valor):
            break
        i += 1

    #caulcular la frecuencia de los datos
    valFreq = frecuencia(data, valFreq, i)

    # Calculando la entropia
    for freq in valFreq.values():
        entropia += (-freq/len(data)) * math.log(freq/len(data), 2)


    return entropia

##devuelve la ganancia de informacion del atributo
def calcularGanaciaInformacion(attributes, data, attr, targetAttr):
    #Calcular la ganacia de informacion para saber si hay que dividir con este atributo
    valFreq = {}
    subsetEntropy = 0.0
    
    
    #index del atributo
    i = attributes.index(attr)

    # Calculamos cuantas veces aparece cada valor del atributo al que le estamos calculando el gain
    valFreq = frecuencia(data, valFreq, i)

    # Calcula la entropia para cada uno de los subgrupos 
    for val in valFreq.keys():
        valProb        = valFreq[val] / sum(valFreq.values())
        dataSubset     =  []

        for fila in data:
            if fila[i] == val:
                dataSubset.append(fila)

        entropiaValor = calcularEntropia(attributes, dataSubset, targetAttr)
        subsetEntropy += valProb * entropiaValor #Resto

    entropiaTotal = calcularEntropia(attributes, data, targetAttr)
    gain = entropiaTotal - subsetEntropy

    return (gain)

#escoger el mejor atributo
def escogerInformacion(data, attributes, target):
    best = attributes[0]
    maxGain = 0
    for attr in attributes:
        if attr != target: 
            newGain = calcularGanaciaInformacion(attributes, data, attr, target) 
            if newGain>maxGain:
                maxGain = newGain
                best = attr
    return best, maxGain

#Retorna una lista con todos los posibles valores de un atributo
def getValues(data, attributes, attr):
    index = attributes.index(attr)
    values = []
    
    for fila in data:
        if fila[index] not in values: 
            values.append(fila[index])
    
    return values

#Retorna una lista con los valores que se borraron el el camino, pero que se ocupan igual 
def getDifValues(valoresOriginales, valoresActuales):
    valoresFaltantes = list(set(valoresOriginales)-set(valoresActuales))

    return valoresFaltantes

#retorna las filas que a este nivel del arbol siguen siendo validas
def obtenerFilasValidas(data, attributes, best, val):
    filasValidas = [[]]
    index = attributes.index(best)
    for valor in data:
        if (valor[index] == val):
            newEntry = []

            #agrega el valor si no es el mejor
            for i in range(0,len(valor)):
                if(i != index):
                    newEntry.append(valor[i])
            filasValidas.append(newEntry)
    filasValidas.remove([])
    return filasValidas

def crearArbol(data, attributes, target):
    
    data = data[:] #Copia del parametro data, no una referencia
    
    valores = [] 
    
    #Una lista con todos los valores actuales del target
    for fila in data:
        valores.append(fila[attributes.index(target)])
    default = mayoria(attributes, data, target) #Retorna cual partido tiene mas votos, ese partido sera el default
    

    # Si el dataset esta limpio retorna el default value. 
    # Se verifica si aun quedan atributos, se resta - 1 para no tomar en cuenta el atributo target.
    if not data or (len(attributes) - 1) <= 0:
        return default
    # Si todos los ejemplos votaron por el mismo partido, retorne ese partido
    if valores.count(valores[0]) == len(valores):
        return valores[0]
    else:
        # Elija el siguiente mejor atributo
        best, maxGain = escogerInformacion(data, attributes, target)

        # Creamos un nuevo nodo con el mejor atributo
        tree = {best:{'__GI':maxGain, '__PL':default}}

        # Verificamos si hay nodos que falten de agregar
        valoresActuales = getValues(data, attributes, best)

        # Creamos un nuevo nodo por cada posible valor del mejor atributo
        for val in valoresActuales:
            
            filasValidas = obtenerFilasValidas(data, attributes, best, val) #Retorna todos los ejemplos para un valor del mejor atributo
            newAttr = attributes[:] #Hacemos una copia de los atributos para no modificar los originales
            newAttr.remove(best) #Quitamos el mejor atributo
            subtree = crearArbol(filasValidas, newAttr, target) #Creamos el nuevo nodo
    
            # Asignamos el nuevo nodo
            tree[best][val] = subtree
        
    return tree

def builDecisionTreeFile(filename, target):
    file = open(filename)
    data = [[]]
    for line in file:
        line = line.strip("\r\n")
        data.append(line.split(','))
    data.remove([])
    attributes = data[0]
    data.remove(attributes)
    #Recibe una lista de listas, los atributos, y el atributo que vamos a querer predecir
    return crearArbol(data, attributes, target)

def main():
    #De aqui para atras tenemos q reemplazarlo con el codigo de nuestro generador
    tree = builDecisionTreeFile('datasets/ejemploLibro.csv', 'Y')
    print(tree)

if __name__ == '__main__':
    main()