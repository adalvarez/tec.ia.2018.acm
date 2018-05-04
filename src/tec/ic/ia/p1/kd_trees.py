from operator import itemgetter
import math
import operator
puntos_cercanos = []


def construir_kd_tree(puntos, profundidad, cantidad_dimensiones):
  global puntos_cercanos
  puntos_cercanos = []
  cantidad_puntos = len(puntos)
  
  #Si no hay puntos, no se hace ningun kdtree
  if cantidad_puntos <= 0:
    return None

  #Obtenemos cual es la dimension q nos interesa, dada la profundidad actual
  #y la cantidad de dimensiones 
  dimension = profundidad % cantidad_dimensiones

  puntos_ordenados = sorted(puntos, key=itemgetter(dimension))

  #Creamos el nuevo nodo, donde su punto es la mediana del set actual y sus nodos izq y derecho
  #Son un nuevo nodo con las mitades del set actual
  punto = puntos_ordenados[cantidad_puntos // 2]
  nuevo_nodo = {
    'punto': punto[:len(punto)-2],
    'left': construir_kd_tree(puntos_ordenados[:cantidad_puntos//2], profundidad + 1, 2),
    'right': construir_kd_tree(puntos_ordenados[cantidad_puntos//2 + 1:], profundidad + 1, 2),
    'clase': punto[len(punto)-2],
    'id': punto[len(punto)-1]
  }

  return nuevo_nodo

#Calcula la distancia euclidiana entre 2 puntos
def calcular_distancia(punto1, punto2):
    
    distance = 0
    for i in range(len(punto1)):
      di = punto1[i] - punto2[i]
      distance += di*di
    return math.sqrt(distance)






#Retorna cual punto entre punto1 y punto2 esta mas cerca al punto_entrada
def distancia_mas_cercana(punto_entrada,punto1, punto2):
  
  if punto1 is None:
    distancia2 = calcular_distancia(punto_entrada, punto2['punto'])
    punto2['distancia'] = distancia2
    return punto2

  if punto2 is None:
    distancia1 = calcular_distancia(punto_entrada, punto1['punto'])
    punto1['distancia'] = distancia1
    return punto1

  #Distancia entre el punto de entrada y el punto 1
  distancia1 = calcular_distancia(punto_entrada, punto1['punto'])
  punto1['distancia'] = distancia1

  #Distancia entre el punto de entrada y el punto 2
  distancia2 = calcular_distancia(punto_entrada, punto2['punto'])
  punto2['distancia'] = distancia2

  if distancia1 < distancia2:
   
    return punto1
  else:
    
    return punto2

#Recibe un punto para insertar en puntos. Puntos tiene que tener un len maximo de k.
#En caso de tener mas de K, reemplaza a aquel q tenga la peor distancia
def insertar_punto(punto, k):
  global puntos_cercanos
  
  for i_punto in puntos_cercanos:
    if punto["id"] == i_punto["id"]:
      return

  if len(puntos_cercanos) < k:
    punto_resumido = {'punto':punto['punto'],'clase':punto['clase'], 'distancia': punto['distancia'], 'id':punto['id']}
    puntos_cercanos.append(punto_resumido)

  else:
    
    #Ordena la lista de puntos para saber cual es el q tiene mayor distancia
    puntos_cercanos.sort(key=lambda x:x['distancia'])
    
    #Si el punto de entrada tiene menor distancia, que el de peor distancia de la lista actual
    ultimo_punto = puntos_cercanos[len(puntos_cercanos) -1]
    
    if punto['distancia'] < ultimo_punto['distancia']:
      #Saca el que tiene mayor distancia
      puntos_cercanos.pop()
      #Ingresa el nuevo punto
      punto_resumido = {'punto':punto['punto'],'clase':punto['clase'], 'distancia': punto['distancia'], 'id':punto['id']}
      puntos_cercanos.append(punto_resumido)
      
    



#Recibe una raiz para empezar a recorrer, el punto al que le vamos a encontrar los vecinos cercanos
#Un contador para saber la profundidad actual, la cantidad de dimensiones de cada ejemplo
#El k representa la cantidad de vecinos q estamos buscando
def kd_tree_punto_mas_cercano(raiz, punto, profundidad, cantidad_dimensiones,k):
  global puntos_cercanos
  if raiz is None:
    
    return None

  #Obtenemos cual es la dimension q nos interesa, dada la profundidad actual
  #y la cantidad de dimensiones 
  dimension = profundidad % cantidad_dimensiones

  rama_siguiente = None
  rama_opuesta = None

  #Definimos a q rama pertenece el punto de entrada, y seteamos como rama opuesta, a la q no pertenece
  if punto[dimension] < raiz['punto'][dimension]:
    rama_siguiente = raiz['left']
    rama_opuesta = raiz['right']
  else:
    rama_siguiente = raiz['right']
    rama_opuesta = raiz['left']

  punto_cercano = distancia_mas_cercana(punto, kd_tree_punto_mas_cercano(rama_siguiente, punto, profundidad + 1, cantidad_dimensiones, k), raiz)
  
  insertar_punto(raiz, k)
  #Verificamos si hay un punto aun mas cercano al encontrado, en la rama opuesta.
  #Esto puede darse si la distancia entre del mejor actual es mayor, que la distancia entre el punto de entrada, y el punto de ramificacion
  if calcular_distancia(punto, punto_cercano['punto']) > abs(punto[dimension] - raiz['punto'][dimension]):
        punto_cercano = distancia_mas_cercana(punto, kd_tree_punto_mas_cercano(rama_opuesta, punto,profundidad + 1, cantidad_dimensiones, k), punto_cercano)
        insertar_punto(raiz, k)
  return punto_cercano

def kd_tree_punto_mas_cercano_aux(raiz, punto, profundidad, cantidad_dimensiones,k):
  global puntos_cercanos
  punto_cercano = kd_tree_punto_mas_cercano(raiz, punto , profundidad, cantidad_dimensiones,k)
  return punto_cercano, puntos_cercanos

def kd_predict(raiz, punto, profundidad, cantidad_dimensiones,k):
  global puntos_cercanos
  puntos_cercanos = []
  
  punto_cercano, puntos_cercanos = kd_tree_punto_mas_cercano_aux(raiz, punto, profundidad, cantidad_dimensiones,k)
  
  frecuencias = {}
  for punto in puntos_cercanos:
    if (punto['clase'] in frecuencias):
      frecuencias[punto['clase']] += 1.0 
    else:
      frecuencias[punto['clase']] = 1.0
  
  return max(frecuencias.items(), key=operator.itemgetter(1))[0]




