# ODE Solver

Projeto para solucionar EDOs através de metodos numericos computacionais aprendidos no curso Metodos Numericos, if816

### Pre-requisitos

-Possuir Python instalado
	-Possuir sympy instalado
		pip install sympy
	-Possuir matplotlib instalado
		pip install matplotlib
	-Talvez seja necessario instalar o tk para rodar o matplotlib
		Utilizando seu distribuidor padrao de pacotes
		ex.:
			sudo pacman -S tk (manjaro/arch)
### Ambiente de Testes
Linux Manjaro - xfce

## Executando o Programa
-Dentro do diretório do arquivo rodar o comando
		python metodos.py	

### Exemplo de Entrada
-Todas estarão especificadas durante a execução
	-Obs.:
		O sympy se utiliza da seguinte formatação:
			x² = x**2
			x^y = x**y
	Exemplo de entrada:
		-3*t + 2*y + 1			#valor de f
		0 						#valor de t0
		1						#valor de y0
		0.05 					#valor de h
		0.4 					#valor de tf
		#seleção dos metodos
		1 						#euler
		2 						#euler invertido
		5 						#adam bashforth
		0 						#definir k = 0
		-1 						#saida do programa
### Exemplo de saída

Euler | Resultados:
y 1 :  1.1499999999999999 t 1 :  0.05
y 2 :  1.3074999999999999 t 2 :  0.10
y 3 :  1.4732499999999999 t 3 :  0.15
y 4 :  1.6480750000000000 t 4 :  0.20
y 5 :  1.8328825000000000 t 5 :  0.25
y 6 :  2.0286707499999999 t 6 :  0.30
y 7 :  2.2365378249999996 t 7 :  0.35
y 8 :  2.4576916074999997 t 8 :  0.40

Sendo seguido de um gráfico que plota os valores obtidos e identifica o metodo utilizado, a fim de comparar os resultados de diversos metodos
