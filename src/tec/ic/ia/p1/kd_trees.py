from operator import itemgetter
import math

def construir_kd_tree(puntos, profundidad, cantidad_dimensiones):
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
  nuevo_nodo = {
    'punto': puntos_ordenados[cantidad_puntos // 2],
    'left': construir_kd_tree(puntos_ordenados[:cantidad_puntos//2], profundidad + 1, 2),
    'right': construir_kd_tree(puntos_ordenados[cantidad_puntos//2 + 1:], profundidad + 1, 2)
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
    return punto2

  if punto2 is None:
    return punto1

  #Distancia entre el punto de entrada y el punto 1
  distancia1 = calcular_distancia(punto_entrada, punto1)

  #Distancia entre el punto de entrada y el punto 2
  distancia2 = calcular_distancia(punto_entrada, punto2)

  if distancia1 < distancia2:
    return punto1
  else:
    return punto2





def kd_tree_punto_mas_cercano(raiz, punto, profundidad, cantidad_dimensiones):
  
  if raiz is None:
    return None

  #Obtenemos cual es la dimension q nos interesa, dada la profundidad actual
  #y la cantidad de dimensiones 
  dimension = profundidad % cantidad_dimensiones

  rama_siguiente = None
  rama_opuesta = None

  if punto[dimension] < raiz['punto'][dimension]:
    rama_siguiente = raiz['left']
    rama_opuesta = raiz['right']
  else:
    rama_siguiente = raiz['right']
    rama_opuesta = raiz['left']

  punto_cercano = distancia_mas_cercana(punto, kd_tree_punto_mas_cercano(rama_siguiente, punto, profundidad + 1, cantidad_dimensiones), raiz['punto'])

  if calcular_distancia(punto, punto_cercano) > abs(punto[dimension] - raiz['punto'][dimension]):
        punto_cercano = distancia_mas_cercana(punto, kd_tree_punto_mas_cercano(rama_opuesta, punto,profundidad + 1), punto_cercano)

  return punto_cercano





kd_tree = construir_kd_tree([[3,6],[17,15],[13,15],[6,12],[9,1],[2,7],[10,19]], 0, 2)
print(kd_tree)



