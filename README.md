1.Requisitos:
	-Possuir Python instalado
	-Possuir sympy instalado
		pip install sympy
	-Possuir matplotlib instalado
		pip install matplotlib
	-Talvez seja necessario instalar o tk para rodar o matplotlib
		Utilizando seu distribuidor padrao de pacotes
		ex.:
			sudo pacman -S tk (manjaro/arch)
2.Ambiente de teste:
	-Manjaro Linux
3.Executando o programa
	-Dentro do diretório do arquivo rodar o comando
		python metodos.py
4.Entradas do programa
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
5.Saída
	-Será mostra os valores de y0 ate yf para o metodo selecionado
		Nos casos de Moulton por Previsão, apenas yf sera mostrado
	-Após a saida, será plotado um grafico comparando as resposta do metodo utilizado
