from kd_trees import *

def test_construir_kd_tree():
	# Data input
	d1 = [[40,70,"PAC",1],[70,130,"PAC",2],[90,40,"PAC",3],[110,100,"PAC",4],[140,110,"PAC",5],[150,30,"PAC",6],[160,100,"PAC",7]]
	d2 = 0
	d3 = 2
	# Execution
	result = construir_kd_tree(d1, d2, d3)
	# Assertions
	assert type(result) is dict
	assert result == {'punto': [110, 100], 'left': {'punto': [40, 70], 'left': {'punto': [90, 40], 'left': None, 'right': None, 'clase': 'PAC', 'id': 3}, 'right': {'punto': [70, 130], 'left': None, 'right': None, 'clase': 'PAC', 'id': 2}, 'clase': 'PAC', 'id': 1}, 'right': {'punto': [160, 100], 'left': {'punto': [150, 30], 'left': None, 'right': None, 'clase': 'PAC', 'id': 6}, 'right': {'punto': [140, 110], 'left': None, 'right': None, 'clase': 'PAC', 'id': 5}, 'clase': 'PAC', 'id': 7}, 'clase': 'PAC', 'id': 4}

def test_calcular_distancia():
	# Data input
	d1 = [0,1]
	d2 = [1,0]
	# Execution
	result = calcular_distancia(d1, d2)
	# Assertions
	assert type(result) is float
	assert result == 1.4142135623730951

def test_distancia_mas_cercana():
	# Data input
	d1 = [6,6]
	d2 = {'punto': [90, 40]}
	d3 = {'punto': [70, 50]}
	# Execution
	result = distancia_mas_cercana(d1, d2, d3)
	# Assertions
	assert type(result) is dict
	assert result == {'punto': [70, 50], 'distancia': 77.6659513557904}

def test_kd_tree_punto_mas_cercano():
	# Data input
	d1 = {'punto': [110, 100], 'left': {'punto': [40, 70], 'left': {'punto': [90, 40], 'left': None, 'right': None, 'clase': 'PAC', 'id': 3}, 'right': {'punto': [70, 130], 'left': None, 'right': None, 'clase': 'PAC', 'id': 2}, 'clase': 'PAC', 'id': 1}, 'right': {'punto': [160, 100], 'left': {'punto': [150, 30], 'left': None, 'right': None, 'clase': 'PAC', 'id': 6}, 'right': {'punto': [140, 110], 'left': None, 'right': None, 'clase': 'PAC', 'id': 5}, 'clase': 'PAC', 'id': 7}, 'clase': 'PAC', 'id': 4}
	d2 = [140,90]
	d3 = 0
	d4 = 2
	d5 = 1
	# Execution
	result = kd_tree_punto_mas_cercano(d1, d2, d3, d4, d5)
	# Assertions
	assert type(result) is dict
	assert result == {'punto': [140, 110], 'left': None, 'right': None, 'clase': 'PAC', 'id': 5, 'distancia': 20.0}

def test_kd_predict():
	# Data input
	d1 = {'punto': [110, 100], 'left': {'punto': [40, 70], 'left': {'punto': [90, 40], 'left': None, 'right': None, 'clase': 'PAC', 'id': 3}, 'right': {'punto': [70, 130], 'left': None, 'right': None, 'clase': 'PAC', 'id': 2}, 'clase': 'PAC', 'id': 1}, 'right': {'punto': [160, 100], 'left': {'punto': [150, 30], 'left': None, 'right': None, 'clase': 'PAC', 'id': 6}, 'right': {'punto': [140, 110], 'left': None, 'right': None, 'clase': 'PAC', 'id': 5}, 'clase': 'PAC', 'id': 7}, 'clase': 'PAC', 'id': 4}
	d2 = [140,90]
	d3 = 0
	d4 = 2
	d5 = 1
	# Execution
	result = kd_predict(d1, d2, d3, d4, d5)
	# Assertions
	assert type(result) is str
	assert result == 'PAC'