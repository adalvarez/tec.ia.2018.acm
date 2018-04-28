import decisionTree

def get_error_rate(results, real_results):
  
  errors = 0
  for indice_resultado in range(len(results)):
    if results[indice_resultado] != real_results[indice_resultado]:
      errors += 1

  return errors

#Recibe un set de entrenamiento y retorna una lista con las respuestas esperadas por cada ejemplo
def get_real_results(matrix):
  return [row[len(row)-1] for row in matrix]  


#Recibe un training set y un validation set. Busca el tipo de learner que se va a usar en options, asi como otros parametros necesarios
#f
def get_results(training_set, validation_set, options, tipo_modelo):

  if options.rl == True:
    print("Realizando regresion logistica")
  elif options.rn == True:
    print("Realizando redes neuronales") 
  elif options.a == True:
    print("Realizando arbol de decision")
    
    result_training = []
    result_validation = []

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

    #Realizamos las predicciones con el training set
    for example in training_set:
      newResult = decisionTree.decisionTreePredict(tree, example)
      result_training.append(newResult)

    #Realizamos las predicciones con el validation set
    for example in validation_set:
      newResult = decisionTree.decisionTreePredict(tree, example)
      result_validation.append(newResult)

  elif options.knn == True:
    print("Realizando k nearest neighbors")
  elif options.svm == True:
    print("Realizando SVM") 

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

#Recibe un tipo de learner, retorna el error promedio usando el training set, y el error promedio usando el validation
def k_fold_cross_validation(validation_k, test_percentage, examples, options, tipo_modelo):
  fold_error_t = 0
  fold_error_v = 0
  validation_k_original = validation_k

  k_fold_examples, test_set = partition_h(examples, test_percentage) #Dejamos un 70% para fold, y un 30% para test set
  print("k_fold_examples", len(k_fold_examples))
  print("test_set", len(test_set))

  if len(k_fold_examples) % validation_k != 0:
    validation_k +=1

  for i in range(validation_k):

    training_set, validation_set = partition_k(k_fold_examples, i, validation_k_original)
    print(i,"trainingset",len(training_set))
    print(i,"validationset",len(validation_set))
    '''
    result_training, result_validation = get_results(training_set, validation_set, options, tipo_modelo)

    
    #Si lo aplicado fue rl o rn entonces ya en result training y result validation tengo el error rate
    if options.rl or options.rn:
      fold_error_t += result_training
      fold_error_v += result_validation

    #Si no, tengo q calcularlo a pata  
    else:
      fold_error_t += get_error_rate(result_training, get_real_results(training_set) )
      fold_error_v += get_error_rate(result_validation, get_real_results(validation_set) )
    
  #Prueba final con test set
  #Si lo aplicado fue rl o rn entonces ya en result training y result validation tengo el error rate
  result_training, result_validation = get_results(k_fold_examples, test_set, options, tipo_modelo)
  if options.rl or options.rn:
    final_error_t = result_training
    final_error_v = result_validation
    
  #Si no, tengo q calcularlo a pata
  else:
    final_error_t = get_error_rate(result_training, get_real_results(k_fold_examples) )
    final_error_v = get_error_rate(result_validation, get_real_results(test_set) )
    '''

  return fold_error_t/validation_k, fold_error_v/validation_k, final_error_t, final_error_v


def hold_out_cross_validation(test_percentage, examples, options, tipo_modelo):
  training_set, validation_set = partition_h(examples, test_percentage)
  print("Len Trainin Set", len(training_set))
  print("Len Validation Set", len(validation_set))
  result_training, result_validation = get_results(training_set, validation_set, options, tipo_modelo)
  #Si lo aplicado fue rl o rn entonces ya en result training y result validation tengo el error rate
  if options.rl or options.rn:
    error_t = result_training
    error_v = result_validation

  #Si no, tengo q calcularlo a pata
  else:
    error_t = (get_error_rate(result_training, get_real_results(training_set) ) / len(training_set)) * 100
    error_v = (get_error_rate(result_validation, get_real_results(validation_set) ) / len(validation_set)) * 100
  return error_t, error_v






