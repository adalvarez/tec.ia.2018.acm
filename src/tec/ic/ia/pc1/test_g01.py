from g01 import *
import os

def test_readCSV(filename='csv_test_in.csv'):
	result = readCSV(filename)
	assert len(result) == 4
	for i in range(0,3):
		assert len(result[i]) == 3
	assert type(result) is list

def test_createCSV():
	filename = './csv_test_out.csv'
	createCSV(filename, [['col1','col2','col3'],[1,2,3],[4,5,6],[7,8,9]])
	assert os.path.exists(filename)
	test_readCSV(filename)

def test_crear_estructura_votos_cantones():
	result = crear_estructura_votos_cantones(filename='juntas_test.csv')
	assert len(result) == 3
	assert type(result[0]) is list
	assert len(result[0]) == 20
	assert type(result[1]) is list
	assert len(result[1]) == 20
	assert type(result[2]) is dict
	assert len(result[2]) == 20

def test_obtener_rangos_votos_canton():
	result = obtener_rangos_votos_canton([x for x in range(1,10)])
	assert type(result) is list
	assert len(result) == 9
	assert result == [1,3,6,10,15,21,28,36,45]

def test_obtener_rangos_votos_provincia():
	result = obtener_rangos_votos_provincia([x for x in range(1,82)], [str(x) for x in range(1,82)], "SAN JOSE")
	assert type(result) is tuple
	assert len(result) == 2
	assert type(result[0]) is list
	assert len(result[0]) == 20
	assert result[0] == [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210]
	assert type(result[1]) is list
	assert len(result[1]) == 20
	assert result[1] == [str(x) for x in range(1,21)]

def test_asignar_canton():
	result = asignar_canton(4, [3,6,10], ['A','B','C'])
	assert type(result) is str
	assert result == 'B'

def test_asignar_genero():
	result = asignar_genero(100, 1)
	assert type(result) is str
	assert result == 'M'

def test_asignar_edad():
	result = asignar_edad(1)
	assert type(result) is int
	assert result in [x for x in range(18, 65)]

def test_asignar_zona():
	result = asignar_zona(50,1)
	assert type(result) is str
	assert result == 'URBANA'

def test_asignar_dependencia():
	result = asignar_dependencia(100, 1)
	assert type(result) is str
	assert result == 'SI'

def test_asignar_por_porcentaje():
	result = asignar_por_porcentaje(50,1)
	assert type(result) is str
	assert result == 'SI'

def test_asignar_jefe_hogar():
	result = asignar_jefe_hogar(98,1,1)
	assert type(result) is str
	assert result == 'MUJER'

def test_asignar_voto():
	result = asignar_voto([x for x in range(1,2)],1)
	assert type(result) is str
	assert result == 'ACCESIBILIDAD SIN EXCLUSION'