from decisionTree import *
from tec.ic.ia.pc1 import g01

def test_frecuencia():
	# Data input
	d1 = [['M', 'Nino', 'URBANA'], ['M', 'Adulto Mayor', 'URBANA'], ['F', 'Nino', 'URBANA']]
	d2 = {}
	d3 = 1
	# Execution
	result = frecuencia(d1, d2, d3)
	# Assertions
	assert type(result) is dict
	assert result == {'Nino': 2.0, 'Adulto Mayor': 1.0}

def test_mayoria():
	# Data input
	d1 = ['GENERO', 'EDAD', 'ZONA']
	d2 = [['M', 'Nino', 'URBANA'], ['M', 'Adulto Mayor', 'URBANA'], ['F', 'Nino', 'URBANA']]
	d3 = 'EDAD'
	# Execution
	result = mayoria(d1, d2, d3)
	# Assertions
	assert type(result) is str
	assert result == 'Nino'

def test_calcularEntropia():
	# Data input
	d1 = ['GENERO', 'EDAD', 'ZONA']
	d2 = [['M', 'Nino', 'URBANA'], ['M', 'Adulto Mayor', 'URBANA'], ['F', 'Nino', 'URBANA']]
	d3 = 'EDAD'
	# Execution
	result = calcularEntropia(d1, d2, d3)
	# Assertions
	assert type(result) is float
	assert result == 0.9182958340544896

def test_calcularGanaciaInformacion():
	# Data input
	d1 = ['GENERO', 'EDAD', 'ZONA']
	d2 = [['M', 'Nino', 'URBANA'], ['M', 'Adulto Mayor', 'URBANA'], ['F', 'Nino', 'URBANA']]
	d3 = 'EDAD'
	d4 = 'GENERO'
	# Execution
	result = calcularGanaciaInformacion(d1, d2, d3, d4)
	# Assertions
	assert type(result) is float
	assert result == 0.2516291673878229

def test_escogerInformacion():
	# Data input
	d1 = [['M', 'Nino', 'URBANA'], ['M', 'Adulto Mayor', 'URBANA'], ['F', 'Nino', 'URBANA']]
	d2 = ['GENERO', 'EDAD', 'ZONA']
	d3 = 'EDAD'
	# Execution
	result = escogerInformacion(d1, d2, d3)
	# Assertions
	assert type(result) is tuple
	assert len(result) == 2
	assert type(result[0]) is str
	assert type(result[1]) is float
	assert result == ('GENERO', 0.2516291673878229)

def test_getValues():
	# Data input
	d1 = [['M', 'Nino', 'URBANA'], ['M', 'Adulto Mayor', 'URBANA'], ['F', 'Nino', 'URBANA']]
	d2 = ['GENERO', 'EDAD', 'ZONA']
	d3 = 'EDAD'
	# Execution
	result = getValues(d1, d2, d3)
	# Assertions
	assert type(result) is list
	assert len(result) == 2
	assert type(result[0]) is str
	assert type(result[1]) is str
	assert result == ['Nino', 'Adulto Mayor']

def test_obtenerFilasValidas():
	# Data input
	d1 = [['M', 'Nino', 'URBANA'], ['M', 'Adulto Mayor', 'URBANA'], ['F', 'Nino', 'URBANA']]
	d2 = ['GENERO', 'EDAD', 'ZONA']
	d3 = 'EDAD'
	d4 = 'Nino'
	# Execution
	result = obtenerFilasValidas(d1, d2, d3, d4)
	# Assertions
	assert type(result) is list
	assert len(result) == 2
	assert type(result[0]) is list
	assert type(result[1]) is list
	assert result == [['M', 'URBANA'], ['F', 'URBANA']]

def test_crearArbol():
	# Data input
	fileData = g01.readCSV('./datasets/examenDataset.csv')
	attributes = fileData[0]
	data = fileData[1:]
	target = 'Compra'
	# Execution
	result = crearArbol(data, attributes, target)
	# Assertions
	assert type(result) is dict 
	assert result == {'Digital': {'__GI': 0.6099865470109874, '__PL': 'N', 'Y': 'N', 'N': {'Precio': {'__GI': 0.05642589168200307, '__PL': 'Y', '<25': {'Idioma': {'__GI': 0.10917033867559889, '__PL': 'Y', 'E': 'Y', 'I': 'Y'}}, '26_50': {'Idioma': {'__GI': 0, '__PL': 'Y', 'E': 'Y'}}, '50+': 'Y'}}}}

def test_isLeaf():
	# Data input
	tree = {'Idioma': {'__GI': 0.10917033867559889,'__PL': 'Y','E': 'Y','I': 'Y'}}
	# Execution
	result = isLeaf(tree)
	# Assertions
	assert type(result) is bool 
	assert result == True

def test_pruneTree():
	# Data input
	fileData = g01.readCSV('./datasets/examenDataset.csv')
	attributes = fileData[0]
	data = fileData[1:]
	target = 'Compra'
	# Execution
	result = crearArbol(data, attributes, target)
	pruneTree(result, 0.05)
	# Assertions
	assert type(result) is dict 
	assert result == {'Digital': {'__GI': 0.6099865470109874, '__PL': 'N', 'Y': 'N', 'N': {'Precio': {'__GI': 0.05642589168200307, '__PL': 'Y', '<25': {'Idioma': {'__GI': 0.10917033867559889, '__PL': 'Y', 'E': 'Y', 'I': 'Y'}}, '26_50': 'Y', '50+': 'Y'}}}}

def test_decisionTreePredict():
	# Data input
	tree = {'Idioma': {'__GI': 0.10917033867559889,'__PL': 'Y','E': 'Y','I': 'Y'}}
	# Execution
	result = decisionTreePredict(tree, {'Idioma':'E'})
	# Assertions
	assert type(result) is str 
	assert result == 'Y'

