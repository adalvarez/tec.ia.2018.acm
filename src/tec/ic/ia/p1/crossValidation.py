import decisionTree
import svm
import numpy
import kd_trees
import copy
import redes_neuronales


def get_error_rate(results, real_results):
  print("Resultados")
  print(results)
  #print("Resultados reales")
  #print(real_results)
  errors = 0
  for indice_resultado in range(len(results)):
    if results[indice_resultado] != real_results[indice_resultado]:
      errors += 1

  return errors

#Recibe un set de entrenamiento y retorna una lista con las respuestas esperadas por cada ejemplo
def get_real_results(matrix):
  return [row[len(row)-1] for row in matrix]

def separarXY(datos):
      #X = datos
      #Y = []
      for i in datos:
        #Y.append(i[-1])
        #numpy.delete(i, -1)
        del i[-1]

      return datos


#Recibe un training set y un validation set. Busca el tipo de learner que se va a usar en options, asi como otros parametros necesarios
#f
def get_results(training_set, validation_set, options, tipo_modelo):

  result_training = []
  result_validation = []

  if options.rl == True:
    print("Realizando regresion logistica")
  elif options.rn == True:
    print("Realizando redes neuronales")
    ##no hice esta parte porque hay que cambiar cosas desde atras y no queria cambiarle su codigo y luego cagar algo
  
  elif options.a == True:
    print("Realizando arbol de decision")
    
    #Obtenemos los atributos y el target, q van a variar dependiendo del tipo de corrida
    attributes = ["CANTON", "GENERO","EDAD","ZONA","DEPENDIENTE","CASA_ESTADO","CASA_HACINADA","ALFABETA", "ESCOLARIDAD", "EDUACION", "TRABAJADO", "ASEGURADO","EXTRANJERO", "DISCAPACITADO", "JEFE_HOGAR", "POBLACION","SUPERFICIE","DENSIDAD","V_OCUPADAS","OCUPANTES","VOTO1", "VOTO2"]
    if tipo_modelo == "1r":
      del attributes[-1]
    elif tipo_modelo == "2r":
      del attributes[-2]
    target = attributes[-1]

    #Generamos el arbol
    tree = decisionTree.crearArbol(training_set, attributes, target)
    
    #Realizamos la poda
    decisionTree.pruneTree(tree, float(options.up))
   
    
    #Realizamos las predicciones con el training set
    for example in training_set:
      example_dic = {}
      for i in range(len(example)):
        example_dic[attributes[i]] = example[i]

      newResult = decisionTree.decisionTreePredict(tree, example_dic)
      result_training.append(newResult)

    #Realizamos las predicciones con el validation set
    for example in validation_set:
      example_dic = {}
      for i in range(len(example)):
        example_dic[attributes[i]] = example[i]
      newResult = decisionTree.decisionTreePredict(tree, example_dic)
      result_validation.append(newResult)

  elif options.knn == True:
    print("Realizando k nearest neighbors")
    
    training_set_copia = copy.deepcopy(training_set)
    #print("Asi era trainingset")
    #print(training_set) 
    #Se agrega un identificador unico a cada ejemplo
    for i in range(len(training_set_copia)):
      training_set_copia[i].append(i)
    #print("Asi quedo trainingset")
    #print(training_set)
    #print("Asi quedo trainingsetcopia")
    #print(training_set_copia)
    kd_tree = kd_trees.construir_kd_tree(training_set_copia,0,len(training_set_copia[0]) - 2) #Se le resta 2, ya que el target y el identificador no deben ser tomados como dimensiones
    
    '''
    example = training_set[0]
    print("Este es el ejemplo q voy a testear")
    print(example)
    del example[-1]
    del example[-1]
    print("Asi queda despues de los deletes")
    print(example)
    newResult = kd_trees.kd_predict(kd_tree, example, 0, len(example), int(options.k))
    print("Esta fue la prediccion")
    print(newResult)
    '''
    
    print("Training")
    for example in training_set:
      #print("Ejemplo mandado")
      #print(example)
      #print("Asi era example")
      #print(example)
      example_copia = example[:]
      del example_copia[-1]
      
      #print("Asi quedo example")
      #print(example)
      #print("asi quedo example copia")
      #print(example_copia)

      newResult = kd_trees.kd_predict(kd_tree, example_copia, 0, len(example), int(options.k))
      result_training.append(newResult)
    print("Validation")
    #Realizamos las predicciones con el validation set
    for example in validation_set:
      #print("Ejemplo mandado")
      #print(example)
      example_copia = example[:]
      del example_copia[-1]
      
      newResult = kd_trees.kd_predict(kd_tree, example_copia, 0, len(example), int(options.k))
      result_validation.append(newResult)
    


  elif options.svm == True:
    print("Realizando SVM")

    #Obtenemos las respuestas del training set
    #print("Training set antes de separarle el y")
    #print(training_set)
    respuestas = get_real_results(training_set)
    #print("Estas son las respuestas")
    #print(respuestas)
    training_set_x = separarXY(copy.deepcopy(training_set))
    #print("Con este training set voy a entrenar svm")
    #print(training_set)
    
    
    if tipo_modelo == "1r":
      modelo = svm.generate_svm_model(training_set_x, respuestas, 'ovo', options.kernel)
    else:
      modelo = svm.generate_svm_model(training_set_x, respuestas, 'ovr', options.kernel)
    
    for example in training_set_x:

      newResult = svm.svm_predict(example, modelo)
      result_training.append(newResult)

    #Realizamos las predicciones con el validation set
    for example in validation_set:
      example_copy = example[:]
      del example_copy[-1]
      newResult = svm.svm_predict(example_copy, modelo)
      result_validation.append(newResult)

  return result_training, result_validation

#Retorna el training y validation set para un kfold cv
def partition_k(examples, i, validation_k):
  training_set = []
  validation_set = []
  chunk_size = len(examples)//validation_k

  #Primera parte del training_set
  training_set += examples[0: i*chunk_size]

  #Validation set
  validation_set += examples[i*chunk_size:i*chunk_size+chunk_size]

  #Todo lo que sobra va para el training
  training_set += examples[i*chunk_size+chunk_size:len(examples)]

  return training_set, validation_set 





#Retorna el training y validation set para un hold out cv
def partition_h(examples, test_percentage):
  chunk_size = (len(examples) * test_percentage) // 100
  #Primera parte del training_set
  validation_set = []
  training_set = []
  
  #Validation set
  validation_set += examples[0:chunk_size]

  #Todo lo que sobra va para el training
  training_set += examples[chunk_size:len(examples)]

  return training_set, validation_set


#Retorna el training y validation set para un hold out cv
def partition_h_rn(examples, respuestas, test_percentage):
  
  chunk_size = (len(examples) * test_percentage) // 100
  #Primera parte del training_set
  validation_set_x = []
  training_set_x = []

  validation_set_y = []
  training_set_y = []
  
  #Validation set
  validation_set_x += examples[0:chunk_size]
  validation_set_y += respuestas[0:chunk_size]

  #Todo lo que sobra va para el training
  training_set_x += examples[chunk_size:len(examples)]
  training_set_y += respuestas[chunk_size:len(respuestas)]


  return training_set_x, training_set_y, validation_set_x, validation_set_y 

#Recibe un tipo de learner, retorna el error promedio usando el training set, y el error promedio usando el validation
def k_fold_cross_validation(validation_k, test_percentage, examples, options, tipo_modelo):
  fold_error_t = 0
  fold_error_v = 0
  validation_k_original = validation_k

  k_fold_examples, test_set = partition_h(examples, test_percentage) #Dejamos un 70% para fold, y un 30% para test set
  
  print("k_fold_examples", len(k_fold_examples))
  print("test_set", len(test_set))

  k_fold_examples_original = numpy.copy(k_fold_examples)
  test_set_original = numpy.copy(test_set)

  if len(k_fold_examples) % validation_k != 0:
    validation_k +=1
  print("Estoy haciendo FOLDS")
  for i in range(validation_k):
    print("Fold", i)
    training_set, validation_set = partition_k(k_fold_examples, i, validation_k_original)
    print("TrainingSet", len(training_set))
    print("ValidationSet", len(validation_set))
    training_set_original = numpy.copy(training_set)
    validation_set_original = numpy.copy(validation_set)
   
    
    result_training, result_validation = get_results(training_set, validation_set, options, tipo_modelo)

    
    #Si lo aplicado fue rl o rn entonces ya en result training y result validation tengo el error rate
    if options.rl or options.rn:
      fold_error_t += result_training
      fold_error_v += result_validation

    #Si no, tengo q calcularlo a pata  
    else:
      fold_error_t += get_error_rate(result_training, get_real_results(training_set_original) )
      fold_error_v += get_error_rate(result_validation, get_real_results(validation_set_original) )
    
  #Prueba final con test set
  #Si lo aplicado fue rl o rn entonces ya en result training y result validation tengo el error rate
  print("Estoy probando test set")
  result_training, result_validation = get_results(k_fold_examples, test_set, options, tipo_modelo)
  if options.rl or options.rn:
    final_error_t = result_training
    final_error_v = result_validation
    
  #Si no, tengo q calcularlo a pata
  else:
    final_error_t = (get_error_rate(result_training, get_real_results(k_fold_examples_original) )/ len(k_fold_examples_original)) * 100
    final_error_v = (get_error_rate(result_validation, get_real_results(test_set_original) ) / len(test_set_original)) * 100
    

  return (fold_error_t/len(k_fold_examples))*100, (fold_error_v/len(k_fold_examples))*100, final_error_t, final_error_v


def hold_out_cross_validation(test_percentage, examples, options, tipo_modelo):
  training_set, validation_set = partition_h(examples, test_percentage)
  training_set_original = numpy.copy(training_set)
  validation_set_original = numpy.copy(validation_set)
  
  
  result_training, result_validation = get_results(training_set, validation_set, options, tipo_modelo)
  #Si lo aplicado fue rl o rn entonces ya en result training y result validation tengo el error rate
  if options.rl or options.rn:
    error_t = result_training
    error_v = result_validation

  

  #Si no, tengo q calcularlo a pata
  else:
        
    error_t = (get_error_rate(result_training, get_real_results(training_set_original) ) / len(training_set)) * 100
    error_v = (get_error_rate(result_validation, get_real_results(validation_set_original) ) / len(validation_set)) * 100
  return error_t, error_v

def hold_out_cross_validation_rn(test_percentage, examples, respuestas, options):
  training_set_x, training_set_y, validation_set_x, validation_set_y = partition_h_rn(examples, respuestas, test_percentage)
  respuestas, accuracy_training, accuracy_validation = redes_neuronales.redes_neuronales(numpy.asarray(training_set_x),numpy.asarray(training_set_y),numpy.asarray(validation_set_x) ,numpy.asarray(validation_set_y), int(options.nc), int(options.uc),options.fa)
  return respuestas, accuracy_training, accuracy_validation
  
  






