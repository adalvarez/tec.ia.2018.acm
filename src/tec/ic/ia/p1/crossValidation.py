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
def get_results(training_set, validation_set):
  if options["rl"] == True:
    print("Realizando regresion lineal")      

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


#Recibe un tipo de learner, retorna el error promedio usando el training set, y el error promedio usando el validation
def k_fold_cross_validation(validation_k, examples):
  fold_error_t = 0
  fold_error_v = 0
  validation_k_original = validation_k
  if len(examples) % validation_k != 0:
    validation_k +=1

  for i in range(validation_k):
    training_set, validation_set = partition_k(examples, i, validation_k_original)
    
    result_training, result_validation = get_results(training_set, validation_set)

    fold_error_t += get_error_rate(result_training, get_real_results(training_set) )
    fold_error_v += get_error_rate(result_validation, get_real_results(validation_set) )

  return fold_error_t/validation_k, fold_error_v/validation_k


#Retorna el training y validation set para un hold out cv
def partition_h(examples, test_percentage):
  chunk_size = (len(examples) * test_percentage) // 100
  #Primera parte del training_set
  
  #Validation set
  validation_set += examples[0:chunk_size]

  #Todo lo que sobra va para el training
  training_set += examples[chunk_size:len(examples)]

  return training_set, validation_set 


def hold_out_cross_validation(test_percentage, examples):
  training_set, validation_set = partition_h(examples, test_percentage)
  result_training, result_validation = get_results(training_set, validation_set)
  error_t = (get_error_rate(result_training, get_real_results(training_set) ) / len(training_set)) * 100
  error_v = (get_error_rate(result_validation, get_real_results(validation_set) ) / len(validation_set)) * 100
  return error_t, error_v





