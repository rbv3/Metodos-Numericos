#teremos fn como f(tn, yn)
#h sendo a altura, vulgo (tn+1) - (tn), e tf sendo o ponto ded objetivo
#yn1 sendo o ponto objetivo e yn sendo o ponto que possuimos
#logo: a cada iteracao, devemos atualizar o valor de yn+1 do seguinte modo:
#yn1 = yn + algo
#devemos tambem lembrar de atualizr o valor de tn, sendo tn+h
#para os multi passos:
#lnyn é o last # yn, logo, l2yn eh o yn(n-2), por exemplo
#o mesmo vale pra lntn

from sympy import *
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import numpy as np


def euler(steps, tn, yn, fn, h):
	print("\nEuler | Resultados:")
	
	Vy = list()
	Vx = list()
	Vy.append(yn)
	Vx.append(tn)

	for x in range(0, steps):
		

		yn1 = yn + fn.subs([(y, yn), (t, tn)]) * h


		yn = yn1
		tn += h	
		Vy.append(yn)
		Vx.append(tn)
		print("y",x+1, ": ", "%.16f"%yn,"t", x+1, ": ","%.2f"%tn)

	line1, = plt.plot(Vx, Vy, label='Euler')

	#plt.axis([0, tn, 0, yn])
	
def eulerWontPrint(steps, tn, yn, fn, h):

	for x in range(0, steps):
		

		yn1 = yn + fn.subs([(y, yn), (t, tn)]) * h


		yn = yn1
		tn += h	

	return yn1

def eulerWontPrint1(steps, tn, yn, fn, h):
	Vy = list()
	Vx = list()
	Vy.append(yn)
	Vx.append(tn)

	for x in range(0, steps):		

		yn1 = yn + fn.subs([(y, yn), (t, tn)]) * h

		lyn = yn
		yn = yn1
		tn += h	
		Vy.append(yn)
		Vx.append(tn)

	return Vy, Vx, yn1, lyn

def eulerInv(steps, tn, yn,fn, h):
	print ("\n Euler Inverso k=0 | Resultados:")

	Vy = list()
	Vx = list()
	Vy.append(yn)
	Vx.append(tn)

	yn1 = 0
	for x in range(0, steps):
		
		yn1 = yn + fn.subs([(y, y), (t, tn)]) * h
		aux = Eq(yn1, y)

		yn1 = solve(aux, y).pop()

		yn = yn1
		tn += h	
		Vx.append(tn)		
		Vy.append(yn)

		print("y",x+1, ": ", "%.16f"%yn,"t", x+1, ": ","%.2f"%tn)

	line1, = plt.plot(Vx, Vy, label='Euler Inverso')

def eulerAprimorado(steps, tn, yn, fn, h):
	print ("\nEuler Aprimorado | Resultados:")

	Vy = list()
	Vx = list()
	Vy.append(yn)
	Vx.append(tn)

	for x in range(0, steps):
		
		yn1 = eulerWontPrint(1, tn, yn, fn, h)
		yn1 = yn +( fn.subs([(y, yn), (t, tn)]) + fn.subs([(y, yn1), (t, tn)])) * h/2

		yn = yn1
		tn += h	
		Vy.append(yn)
		Vx.append(tn)

		print("y",x+1, ": ", "%.16f"%yn,"t", x+1, ": ","%.2f"%tn)


	line1, = plt.plot(Vx, Vy, label='Euler Aprimorado')

def calcK(tn, yn, fn, h):
	K1 = fn.subs([(y, yn), (t, tn)])
	K2 = fn.subs([(y, yn+((h/2)*K1)), (t, tn+(h/2))])
	K3 = fn.subs([(y, yn+((h/2)*K2)), (t, tn+(h/2))])
	K4 = fn.subs([(y, yn+(h*K3)), (t, tn+h)])

	return K1, K2, K3, K4

def rungeKutta(steps, tn, yn, fn, h):
	print("\nRunge Kutta | Resultados:")

	Vy = list()
	Vx = list()
	Vy.append(yn)
	Vx.append(tn)

	for x in range(0, steps):
		k1, k2, k3, k4 = calcK(tn, yn, fn, h)
		yn1 = yn +( (k1+ 2*k2 + 2*k3 + k4) * h/6)


		yn = yn1
		tn += h	
		Vy.append(yn)
		Vx.append(tn)

		print("y",x+1, ": ", "%.16f"%yn,"t", x+1, ": ","%.2f"%tn)

	line1, = plt.plot(Vx, Vy, label='Runge Kutta')

def rungeKuttaWontPrint(steps, tn, yn, fn, h):
	Ly = list() # lista com valores de y
	Lt = list() # lista com valores de t
	for x in range(0, steps):
		k1, k2, k3, k4 = calcK(tn, yn, fn, h)
		yn1 = yn +( (k1+ 2*k2 + 2*k3 + k4) * h/6)

		yn = yn1
		tn += h	

		Ly.append(yn1)
		Lt.append(tn)

	return Ly, Lt

def k0Bash(steps, tn, yn, fn, h): #same as euler
	print ("\nAdam Bashford k=0 | Resultados:")
	
	Vy = list()
	Vx = list()
	Vy.append(yn)
	Vx.append(tn)

	for x in range(0, steps):		

		yn1 = yn + fn.subs([(y, yn), (t, tn)]) * h

		yn = yn1
		tn += h	
		Vy.append(yn)
		Vx.append(tn)

		print("y",x+1, ": ", "%.16f"%yn,"t", x+1, ": ","%.2f"%tn)

	line1, = plt.plot(Vx, Vy, label='Bash K0')

def k1Bash(steps, tn, yn, fn, h):
	print ("\nAdam Bashford k=1 | Resultados:")

	Vy = list()
	Vx = list()
	Vy.append(yn)
	Vx.append(tn)

	Ly, Lt = rungeKuttaWontPrint(1, tn, yn, fn, h)
	lastyn, lasttn = yn, tn
	yn, tn = Ly[0], Lt[0]

	print("Obtidos por Runge Kutta:")

	print("y 1 :  %.16f"%yn,"t 1 :  %.2f"%tn)

	print("Obtidos por Bashforth:")

	for x in range (1, steps):
		yn1 = yn + (1.5)*h*fn.subs([(y, yn), (t, tn)]) - (0.5)*h*fn.subs([(y, lastyn), (t, lasttn)])

		lastyn = yn
		yn = yn1
		lasttn = tn
		tn += h
		Vy.append(yn)
		Vx.append(tn)
		
		print("y",x+1, ": ", "%.16f"%yn,"t", x+1, ": ","%.2f"%tn)

	line1, = plt.plot(Vx, Vy, label='Bash K1')

	return yn

def k1BashWontPrint(steps, tn, yn, fn, h):
	
	Vy = list()
	Vx = list()
	Vy.append(yn)
	Vx.append(tn)

	Ly, Lt = rungeKuttaWontPrint(1, tn, yn, fn, h)
	lastyn, lasttn = yn, tn
	yn, tn = Ly[0], Lt[0]

	for x in range (1, steps):
		yn1 = yn + (1.5)*h*fn.subs([(y, yn), (t, tn)]) - (0.5)*h*fn.subs([(y, lastyn), (t, lasttn)])

		lastyn = yn
		yn = yn1
		lasttn = tn
		tn += h
		Vy.append(yn)
		Vx.append(tn)
		

	return Vy, Vx, yn, lastyn, tn-h

def k2Bash(steps, tn, yn, fn, h):
	print ("\nAdam Bashford k=2 | Resultados:")
	print("Obtidos por Runge Kutta:")

	Vy = list()
	Vx = list()
	Vy.append(yn)
	Vx.append(tn)

	Ly, Lt = rungeKuttaWontPrint(2, tn, yn, fn, h)
	l2yn, l2tn = yn, tn
	l1yn, l1tn = Ly[0], Lt[0]
	yn, tn = Ly[1], Lt[1]

	print("y 1 :  %.16f"%l1yn,"t 1 :  %.2f"%l1tn)
	print("y 2 :  %.16f"%yn,"t 2 :  %.2f"%tn)

	print("Obtidos por Bashforth:")

	for x in range(2, steps):
		yn1 = yn + (1/12)*h*(23*fn.subs([(y, yn), (t, tn)]) - 16*fn.subs([(y, l1yn), (t, l1tn)]) + 5*fn.subs([(y, l2yn), (t, l2tn)]))

		l2yn = l1yn
		l1yn = yn
		yn = yn1
		l2tn = l1tn
		l1tn = tn
		tn += h
		Vy.append(yn)
		Vx.append(tn)
		
		print("y",x+1, ": ", "%.16f"%yn,"t", x+1, ": ","%.2f"%tn)	

	line1, = plt.plot(Vx, Vy, label='Bash K2')

def k2BashWontPrint(steps, tn, yn, fn, h):

	Vy = list()
	Vx = list()
	Vy.append(yn)
	Vx.append(tn)

	Ly, Lt = rungeKuttaWontPrint(2, tn, yn, fn, h)
	l2yn, l2tn = yn, tn
	l1yn, l1tn = Ly[0], Lt[0]
	yn, tn = Ly[1], Lt[1]

	for x in range(2, steps):
		yn1 = yn + (1/12)*h*(23*fn.subs([(y, yn), (t, tn)]) - 16*fn.subs([(y, l1yn), (t, l1tn)]) + 5*fn.subs([(y, l2yn), (t, l2tn)]))

		l2yn = l1yn
		l1yn = yn
		yn = yn1
		l2tn = l1tn
		l1tn = tn
		tn += h
		Vy.append(yn)
		Vx.append(tn)
	
	return Vy, Vx, yn, l1yn, l2yn, tn-h

def k3Bash(steps, tn, yn, fn, h):
	print ("\nAdam Bashford k=3 | Resultados:")
	print("Obtidos por Runge Kutta:")

	Vy = list()
	Vx = list()
	Vy.append(yn)
	Vx.append(tn)

	Ly, Lt = rungeKuttaWontPrint(3, tn, yn, fn, h)
	l3yn, l3tn = yn, tn
	l2yn, l2tn = Ly[0], Lt[0]
	l1yn, l1tn = Ly[1], Lt[1]
	yn, tn = Ly[2], Lt[2]

	print("y 1 :  %.16f"%l2yn,"t 1 :  %.2f"%l2tn)
	print("y 2 :  %.16f"%l1yn,"t 2 :  %.2f"%l1tn)
	print("y 3 :  %.16f"%yn,"t 3 :  %.2f"%tn)

	print("Obtidos por Bashforth:")

	for x in range(3, steps):
		yn1 = yn + (1/24)*h*(55*fn.subs([(y, yn), (t, tn)]) - 59*fn.subs([(y, l1yn), (t, l1tn)]) + 37*fn.subs([(y, l2yn), (t, l2tn)]) - 9*fn.subs([(y, l3yn), (t, l3tn)]))

		l3yn = l2yn
		l2yn = l1yn
		l1yn = yn
		yn = yn1
		l3tn = l2tn
		l2tn = l1tn
		l1tn = tn
		tn += h
		Vx.append(tn)		
		Vy.append(yn)
		
		print("y",x+1, ": ", "%.16f"%yn,"t", x+1, ": ","%.2f"%tn)	

	line1, = plt.plot(Vx, Vy, label='Bash K3')

def k3BashWontPrint(steps, tn, yn, fn, h):

	Vy = list()
	Vx = list()
	Vy.append(yn)
	Vx.append(tn)

	Ly, Lt = rungeKuttaWontPrint(3, tn, yn, fn, h)
	l3yn, l3tn = yn, tn
	l2yn, l2tn = Ly[0], Lt[0]
	l1yn, l1tn = Ly[1], Lt[1]
	yn, tn = Ly[2], Lt[2]

	for x in range(3, steps):
		yn1 = yn + (1/24)*h*(55*fn.subs([(y, yn), (t, tn)]) - 59*fn.subs([(y, l1yn), (t, l1tn)]) + 37*fn.subs([(y, l2yn), (t, l2tn)]) - 9*fn.subs([(y, l3yn), (t, l3tn)]))

		l3yn = l2yn
		l2yn = l1yn
		l1yn = yn
		yn = yn1
		l3tn = l2tn
		l2tn = l1tn
		l1tn = tn
		tn += h
		Vy.append(yn)
		Vx.append(tn)
	
	return Vy, Vx, yn, l1yn, l2yn, l3yn, tn-h

def k4Bash(steps, tn, yn, fn, h):
	print ("\nAdam Bashford k=4 | Resultados:")
	print("Obtidos por Runge Kutta:")

	Vy = list()
	Vx = list()
	Vy.append(yn)
	Vx.append(tn)

	Ly, Lt = rungeKuttaWontPrint(4, tn, yn, fn, h)
	l4yn, l4tn = yn, tn
	l3yn, l3tn = Ly[0], Lt[0]
	l2yn, l2tn = Ly[1], Lt[1]
	l1yn, l1tn = Ly[2], Lt[2]
	yn, tn = Ly[3], Lt[3]

	print("y 1 :  %.16f"%l3yn,"t 1 :  %.2f"%l3tn)
	print("y 2 :  %.16f"%l2yn,"t 2 :  %.2f"%l2tn)
	print("y 3 :  %.16f"%l1yn,"t 3 :  %.2f"%l1tn)
	print("y 4 :  %.16f"%yn,"t 4 :  %.2f"%tn)

	print("Obtidos por Bashforth:")

	for x in range(4, steps):
		yn1 = yn + (1/720)*h*(1901*fn.subs([(y, yn), (t, tn)]) - 2774*fn.subs([(y, l1yn), (t, l1tn)]) + 2616*fn.subs([(y, l2yn), (t, l2tn)]) - 1274*fn.subs([(y, l3yn), (t, l3tn)]) + 251 * fn.subs([(y, l4yn), (t, l4tn)]))

		l4yn = l3yn
		l3yn = l2yn
		l2yn = l1yn
		l1yn = yn
		yn = yn1
		l4tn = l3tn
		l3tn = l2tn
		l2tn = l1tn
		l1tn = tn
		tn += h
		Vx.append(tn)		
		Vy.append(yn)
		
		print("y",x+1, ": ", "%.16f"%yn,"t", x+1, ": ","%.2f"%tn)	

	line1, = plt.plot(Vx, Vy, label='Bash K4')

def k4BashWontPrint(steps, tn, yn, fn, h):

	Vy = list()
	Vx = list()
	Vy.append(yn)
	Vx.append(tn)

	Ly, Lt = rungeKuttaWontPrint(4, tn, yn, fn, h)
	l4yn, l4tn = yn, tn
	l3yn, l3tn = Ly[0], Lt[0]
	l2yn, l2tn = Ly[1], Lt[1]
	l1yn, l1tn = Ly[2], Lt[2]
	yn, tn = Ly[3], Lt[3]

	for x in range(4, steps):
		yn1 = yn + (1/720)*h*(1901*fn.subs([(y, yn), (t, tn)]) - 2774*fn.subs([(y, l1yn), (t, l1tn)]) + 2616*fn.subs([(y, l2yn), (t, l2tn)]) - 1274*fn.subs([(y, l3yn), (t, l3tn)]) + 251 * fn.subs([(y, l4yn), (t, l4tn)]))

		l4yn = l3yn
		l3yn = l2yn
		l2yn = l1yn
		l1yn = yn
		yn = yn1
		l4tn = l3tn
		l3tn = l2tn
		l2tn = l1tn
		l1tn = tn
		tn += h
		Vy.append(yn)
		Vx.append(tn)
		
	return Vy, Vx, yn, l1yn, l2yn, l3yn, l4yn, tn-h

def k5Bash(steps, tn, yn, fn, h):
	print ("\nAdam Bashford k=5 | Resultados:")
	print("Obtidos por Runge Kutta:")

	Vy = list()
	Vx = list()
	Vy.append(yn)
	Vx.append(tn)
	
	p = steps
	if p < 5:
		p = p
		Ly, Lt = rungeKuttaWontPrint(p, tn, yn, fn, h)

		for i in range(0, p):
			print("y",i+1, ": ", "%.16f"%Ly[i],"t", i+1, ": ","%.2f"%Lt[i])

	else:
		p = 5
		Ly, Lt = rungeKuttaWontPrint(5, tn, yn, fn, h)
		l5yn, l5tn = yn, tn
		l4yn, l4tn = Ly[0], Lt[0]
		l3yn, l3tn = Ly[1], Lt[1]
		l2yn, l2tn = Ly[2], Lt[2]
		l1yn, l1tn = Ly[3], Lt[3]
		yn, tn = Ly[4], Lt[4]

		for i in range(0, p):
			print("y",i+1, ": ", "%.16f"%Ly[i],"t", i+1, ": ","%.2f"%Lt[i])


	print("Obtidos por Bashforth:")

	for x in range(5, steps):
		yn1 = yn + (1/1440)*h*(4277*fn.subs([(y, yn), (t, tn)]) - 7923*fn.subs([(y, l1yn), (t, l1tn)]) + 9982*fn.subs([(y, l2yn), (t, l2tn)]) - 7298*fn.subs([(y, l3yn), (t, l3tn)]) + 2877 * fn.subs([(y, l4yn), (t, l4tn)]) - 475*fn.subs([(y, l4yn), (t, l4tn)]))

		l5yn = l4yn
		l4yn = l3yn
		l3yn = l2yn
		l2yn = l1yn
		l1yn = yn
		yn = yn1
		l5tn = l4tn
		l4tn = l3tn
		l3tn = l2tn
		l2tn = l1tn
		l1tn = tn
		tn += h
		Vx.append(tn)		
		Vy.append(yn)
		
		print("y",x+1, ": ", "%.16f"%yn,"t", x+1, ": ","%.2f"%tn)

	line1, = plt.plot(Vx, Vy, label='Bash K5')			

def k5BashWontPrint(steps, tn, yn, fn, h):

	Vy = list()
	Vx = list()
	Vy.append(yn)
	Vx.append(tn)

	p = steps
	if p < 5:
		p = p
		Ly, Lt = rungeKuttaWontPrint(p, tn, yn, fn, h)

		for i in range(0, p):
			print("y",i+1, ": ", "%.16f"%Ly[i],"t", i+1, ": ","%.2f"%Lt[i])

	else:
		p = 5
		Ly, Lt = rungeKuttaWontPrint(5, tn, yn, fn, h)
		l5yn, l5tn = yn, tn
		l4yn, l4tn = Ly[0], Lt[0]
		l3yn, l3tn = Ly[1], Lt[1]
		l2yn, l2tn = Ly[2], Lt[2]
		l1yn, l1tn = Ly[3], Lt[3]
		yn, tn = Ly[4], Lt[4]


	for x in range(5, steps):
		yn1 = yn + (1/1440)*h*(4277*fn.subs([(y, yn), (t, tn)]) - 7923*fn.subs([(y, l1yn), (t, l1tn)]) + 9982*fn.subs([(y, l2yn), (t, l2tn)]) - 7298*fn.subs([(y, l3yn), (t, l3tn)]) + 2877 * fn.subs([(y, l4yn), (t, l4tn)]) - 475*fn.subs([(y, l4yn), (t, l4tn)]))

		l5yn = l4yn
		l4yn = l3yn
		l3yn = l2yn
		l2yn = l1yn
		l1yn = yn
		yn = yn1
		l5tn = l4tn
		l4tn = l3tn
		l3tn = l2tn
		l2tn = l1tn
		l1tn = tn
		tn += h
		Vy.append(yn)
		Vx.append(tn)

	return Vy, Vx, yn, l1yn, l2yn, l3yn, l4yn, l5tn, tn-h				

def k0MoultonP(steps, tn, yn, fn, h): #same as backward euler
	print ("\nAdam Moulton Previsao k=0 | Resultados:")
		
	Vy, Vx, yn1, yn = eulerWontPrint1(steps, tn, yn, fn, h)
	tn = tn + (steps)*h

	yn1 = yn + fn.subs([(y, yn1), (t, tn)]) * h

	Vy[len(Vy)-1] = yn1


	print("y",steps, ": ", "%.16f"%yn1,"t", steps, ": ","%.2f"%tn)

	line1, = plt.plot(Vx, Vy, label='Moulton Prev K0')	

def k1MoultonP(steps, tn, yn, fn, h): 
	print ("\nAdam Moulton Previsao k=1 | Resultados:")

	Vy, Vx, yn1, yn, tn = k1BashWontPrint(steps, tn, yn, fn, h)

	yn1 = yn + (0.5)*h*fn.subs([(y, yn1), (t, tn+h)]) + (0.5)*h*fn.subs([(y, yn), (t, tn)])

	Vy[len(Vy)-1] = yn1
	
	print("y",steps, ": ", "%.16f"%yn1,"t", steps, ": ","%.2f"%(tn+h)) 
	
	line1, = plt.plot(Vx, Vy, label='Moulton Prev K1')		

def k2MoultonP(steps, tn, yn, fn, h): 
	print ("\nAdam Moulton Previsao k=2 | Resultado:")
	
	Vy, Vx, yn1, yn, l1yn, tn = k2BashWontPrint(steps, tn, yn, fn, h)
	yn1 = yn + ((1/12.0)*h*(5*fn.subs([(y, yn1), (t, tn+h)]) + 8*fn.subs([(y, yn), (t, tn)]) - 1*fn.subs([(y, l1yn), (t, tn-h)])))

	Vy[len(Vy)-1] = yn1

	print("y",steps, ": ", "%.16f"%yn1,"t", steps, ": ","%.2f"%(tn+h))

	line1, = plt.plot(Vx, Vy, label='Moulton Prev K2')	

def k3MoultonP(steps, tn, yn, fn, h): 
	print ("\nAdam Moulton Previsao k=3 | Resultado:")
	
	Vy, Vx, yn1, yn, l1yn, l2yn, tn = k3BashWontPrint(steps, tn, yn, fn, h)
	yn1 = yn + ((1/24.0)*h*(9*fn.subs([(y, yn1), (t, tn+h)]) + 19*fn.subs([(y, yn), (t, tn)]) - 5*fn.subs([(y, l1yn), (t, tn-h)]) + 1*fn.subs([(y, l2yn), (t, tn-(2*h))])))

	Vy[len(Vy)-1] = yn1

	print("y",steps, ": ", "%.16f"%yn1,"t", steps, ": ","%.2f"%(tn+h))

	line1, = plt.plot(Vx, Vy, label='Moulton Prev K3')	

def k4MoultonP(steps, tn, yn, fn, h): 
	print ("\nAdam Moulton Previsao k=4 | Resultado:")
	
	Vy, Vx, yn1, yn, l1yn, l2yn, l3yn, tn = k4BashWontPrint(steps, tn, yn, fn, h)
	yn1 = yn + ((1/720.0)*h*(251*fn.subs([(y, yn1), (t, tn+h)]) + 646*fn.subs([(y, yn), (t, tn)]) - 264*fn.subs([(y, l1yn), (t, tn-h)]) + 106*fn.subs([(y, l2yn), (t, tn-(2*h))]) - 19*fn.subs([(y, l3yn), (t, tn-(3*h))])))

	Vy[len(Vy)-1] = yn1

	print("y",steps, ": ", "%.16f"%yn1,"t", steps, ": ","%.2f"%(tn+h))

	line1, = plt.plot(Vx, Vy, label='Moulton Prev K4')	

def k5MoultonP(steps, tn, yn, fn, h): 
	print ("\nAdam Moulton Previsao k=5 | Resultado:")
	
	Vy, Vx, yn1, yn, l1yn, l2yn, l3yn, l4yn, tn = k5BashWontPrint(steps, tn, yn, fn, h)
	yn1 = yn + ((1/1440.0)*h*(475*fn.subs([(y, yn1), (t, tn+h)]) + 1427*fn.subs([(y, yn), (t, tn)]) - 798*fn.subs([(y, l1yn), (t, tn-h)]) + 482*fn.subs([(y, l2yn), (t, tn-(2*h))]) - 173*fn.subs([(y, l3yn), (t, tn-(3*h))]) + 27*fn.subs([(y, l4yn), (t, tn-(4*h))])))

	Vy[len(Vy)-1] = yn1

	print("y",steps, ": ", "%.16f"%yn1,"t", steps, ": ","%.2f"%(tn+h))

	line1, = plt.plot(Vx, Vy, label='Moulton Prev K5')	

def k0Moulton(steps, tn, yn, fn, h): #same as backward euler
	print ("\nAdam Moulton k=0 | Resultados:")

	Vy = list()
	Vx = list()
	Vy.append(yn)
	Vx.append(tn)

	yn1 = 0
	for x in range(0, steps):
		
		yn1 = yn + fn.subs([(y, y), (t, tn)]) * h
		aux = Eq(yn1, y)

		yn1 = solve(aux, y).pop()

		yn = yn1
		tn += h	
		Vx.append(tn)		
		Vy.append(yn)

		print("y",x+1, ": ", "%.16f"%yn,"t", x+1, ": ","%.2f"%tn)

	line1, = plt.plot(Vx, Vy, label='Moulton K0')

def k1Moulton(steps, tn, yn, fn, h): 
	print ("\nAdam Moulton k=1 | Resultados:")

	print("Obtidos por Runge Kutta:")

	Vy = list()
	Vx = list()
	Vy.append(yn)
	Vx.append(tn)

	Ly, Lt = rungeKuttaWontPrint(1, tn, yn, fn, h)
	l1yn, l1tn = yn, tn
	yn, tn = Ly[0], Lt[0]

	print("y 1 :  %.16f"%yn,"t 1 :  %.2f"%tn)

	print("Obtidos por Moulton:")

	for x in range(1, steps):

		yn1 = yn + ((1/2.0)*h*(fn.subs([(y, y), (t, tn+h)]) + fn.subs([(y, yn), (t, tn)])))

		aux = Eq(yn1, y)

		yn1 = solve(aux, y).pop()

		l2yn = l1yn
		l1yn = yn
		yn = yn1
		l2tn = l1tn
		l1tn = tn
		tn += h
		Vx.append(tn)		
		Vy.append(yn)

		print("y",x+1, ": ", "%.16f"%yn,"t", x+1, ": ","%.2f"%tn)

	line1, = plt.plot(Vx, Vy, label='Moulton K1')

def k2Moulton(steps, tn, yn, fn, h): 
	print ("\nAdam Moulton k=2 | Resultados:")

	print("Obtidos por Runge Kutta:")

	Vy = list()
	Vx = list()
	Vy.append(yn)
	Vx.append(tn)

	Ly, Lt = rungeKuttaWontPrint(2, tn, yn, fn, h)
	l2yn, l2tn = yn, tn
	l1yn, l1tn = Ly[0], Lt[0]
	yn, tn = Ly[1], Lt[1]

	print("y 1 :  %.16f"%l1yn,"t 1 :  %.2f"%l1tn)
	print("y 2 :  %.16f"%yn,"t 2 :  %.2f"%tn)

	print("Obtidos por Moulton:")

	for x in range(2, steps):

		yn1 = yn + ((1/12.0)*h*(5*fn.subs([(y, y), (t, tn+h)]) + 8*fn.subs([(y, yn), (t, tn)]) - 1*fn.subs([(y, l1yn), (t, l1tn)])))

		aux = Eq(yn1, y)

		yn1 = solve(aux, y).pop()

		l2yn = l1yn
		l1yn = yn
		yn = yn1
		l2tn = l1tn
		l1tn = tn
		tn += h
		Vx.append(tn)		
		Vy.append(yn)

		print("y",x+1, ": ", "%.16f"%yn,"t", x+1, ": ","%.2f"%tn)

	line1, = plt.plot(Vx, Vy, label='Moulton K2')

def k3Moulton(steps, tn, yn, fn, h): 
	print ("\nAdam Moulton k=3 | Resultados:")

	print("Obtidos por Runge Kutta:")

	Vy = list()
	Vx = list()
	Vy.append(yn)
	Vx.append(tn)

	Ly, Lt = rungeKuttaWontPrint(3, tn, yn, fn, h)
	l3yn, l3tn = yn, tn
	l2yn, l2tn = Ly[0], Lt[0]
	l1yn, l1tn = Ly[1], Lt[1]
	yn, tn = Ly[2], Lt[2]

	print("y 1 :  %.16f"%l2yn,"t 1 :  %.2f"%l2tn)
	print("y 2 :  %.16f"%l1yn,"t 2 :  %.2f"%l1tn)
	print("y 3 :  %.16f"%yn,"t 3 :  %.2f"%tn)

	print("Obtidos por Moulton:")

	for x in range(3, steps):

		yn1 = yn + ((1/24.0)*h*(9*fn.subs([(y, y), (t, tn+h)]) + 19*fn.subs([(y, yn), (t, tn)]) - 5*fn.subs([(y, l1yn), (t, l1tn)]) + 1*fn.subs([(y, l2yn), (t, l2tn)])))

		aux = Eq(yn1, y)

		yn1 = solve(aux, y).pop()

		l3yn = l2yn
		l2yn = l1yn
		l1yn = yn
		yn = yn1
		l3tn = l2tn
		l2tn = l1tn
		l1tn = tn
		tn += h
		Vx.append(tn)		
		Vy.append(yn)

		print("y",x+1, ": ", "%.16f"%yn,"t", x+1, ": ","%.2f"%tn)

	line1, = plt.plot(Vx, Vy, label='Moulton K3')

def k4Moulton(steps, tn, yn, fn, h):
	print ("\nAdam Moulton k=4 | Resultados:")
	print("Obtidos por Runge Kutta:")

	Vy = list()
	Vx = list()
	Vy.append(yn)
	Vx.append(tn)

	Ly, Lt = rungeKuttaWontPrint(4, tn, yn, fn, h)
	l4yn, l4tn = yn, tn
	l3yn, l3tn = Ly[0], Lt[0]
	l2yn, l2tn = Ly[1], Lt[1]
	l1yn, l1tn = Ly[2], Lt[2]
	yn, tn = Ly[3], Lt[3]

	print("y 1 :  %.16f"%l3yn,"t 1 :  %.2f"%l3tn)
	print("y 2 :  %.16f"%l2yn,"t 2 :  %.2f"%l2tn)
	print("y 3 :  %.16f"%l1yn,"t 3 :  %.2f"%l1tn)
	print("y 4 :  %.16f"%yn,"t 4 :  %.2f"%tn)

	print("Obtidos por Moulton:")

	for x in range(4, steps):
		
		yn1 = yn + ((1/720.0)*h*(251*fn.subs([(y, y), (t, tn+h)]) + 646*fn.subs([(y, yn), (t, tn)]) - 264*fn.subs([(y, l1yn), (t, l1tn)]) + 106*fn.subs([(y, l2yn), (t, l2tn)]) - 19*fn.subs([(y, l3yn), (t, l3tn)])))

		aux = Eq(yn1, y)

		yn1 = solve(aux, y).pop()

		l4yn = l3yn
		l3yn = l2yn
		l2yn = l1yn
		l1yn = yn
		yn = yn1
		l4tn = l3tn
		l3tn = l2tn
		l2tn = l1tn
		l1tn = tn
		tn += h
		
		Vx.append(tn)		
		Vy.append(yn)

		print("y",x+1, ": ", "%.16f"%yn,"t", x+1, ": ","%.2f"%tn)
	
	line1, = plt.plot(Vx, Vy, label='Moulton K4')

def k5Moulton(steps, tn, yn, fn, h):
	print ("\nAdam Bashford k=5 | Resultados:")
	print("Obtidos por Runge Kutta:")

	Vy = list()
	Vx = list()
	Vy.append(yn)
	Vx.append(tn)

	p = steps
	if p < 5:
		p = p
		Ly, Lt = rungeKuttaWontPrint(p, tn, yn, fn, h)

		for i in range(0, p):
			print("y",i+1, ": ", "%.16f"%Ly[i],"t", i+1, ": ","%.2f"%Lt[i])

	else:
		p = 5
		Ly, Lt = rungeKuttaWontPrint(5, tn, yn, fn, h)
		l5yn, l5tn = yn, tn
		l4yn, l4tn = Ly[0], Lt[0]
		l3yn, l3tn = Ly[1], Lt[1]
		l2yn, l2tn = Ly[2], Lt[2]
		l1yn, l1tn = Ly[3], Lt[3]
		yn, tn = Ly[4], Lt[4]

		for i in range(0, p):
			print("y",i+1, ": ", "%.16f"%Ly[i],"t", i+1, ": ","%.2f"%Lt[i])


	print("Obtidos por Moulton:")

	for x in range(5, steps):
		yn1 = yn + ((1/1440.0)*h*(475*fn.subs([(y, y), (t, tn+h)]) + 1427*fn.subs([(y, yn), (t, tn)]) - 798*fn.subs([(y, l1yn), (t, l1tn)]) + 482*fn.subs([(y, l2yn), (t, l2tn)]) - 173*fn.subs([(y, l3yn), (t, l3tn)]) + 27*fn.subs([(y, l4yn), (t, l4tn)])))

		aux = Eq(yn1, y)

		yn1 = solve(aux, y).pop()

		l5yn = l4yn
		l4yn = l3yn
		l3yn = l2yn
		l2yn = l1yn
		l1yn = yn
		yn = yn1
		l5tn = l4tn
		l4tn = l3tn
		l3tn = l2tn
		l2tn = l1tn
		l1tn = tn
		tn += h
		
		Vx.append(tn)		
		Vy.append(yn)

		print("y",x+1, ": ", "%.16f"%yn,"t", x+1, ": ","%.2f"%tn)
	
	line1, = plt.plot(Vx, Vy, label='Moulton K5')		

def k0Backward(steps, tn, yn, fn, h):
	print ("\n Backwardk=0 | Resultados:")

	Vy = list()
	Vx = list()
	Vy.append(yn)
	Vx.append(tn)

	yn1 = 0
	for x in range(0, steps):
		
		yn1 = yn + fn.subs([(y, y), (t, tn)]) * h
		aux = Eq(yn1, y)

		yn1 = solve(aux, y).pop()

		yn = yn1
		tn += h	
		Vx.append(tn)		
		Vy.append(yn)

		print("y",x+1, ": ", "%.16f"%yn,"t", x+1, ": ","%.2f"%tn)

	line1, = plt.plot(Vx, Vy, label='Backward K0')

def k1Backward(steps, tn, yn, fn, h):
	print ("\n Backward k=1 | Resultados:")

	print("Obtidos por Runge Kutta:")

	Vy = list()
	Vx = list()
	Vy.append(yn)
	Vx.append(tn)

	Ly, Lt = rungeKuttaWontPrint(1, tn, yn, fn, h)
	l1yn, l1tn = yn, tn
	yn, tn = Ly[0], Lt[0]

	print("y 1 :  %.16f"%yn,"t 1 :  %.2f"%tn)

	print("Obtidos por Backward:")
	for x in range(1, steps):

		yn1 = ((1/3.0)*(2*h*fn.subs([(y, y), (t, tn+h)]) + 4*yn - l1yn))

		aux = Eq(yn1, y)

		yn1 = solve(aux, y).pop()

		l1yn = yn
		yn = yn1
		l1tn = tn
		tn += h
		Vx.append(tn)		
		Vy.append(yn)

		print("y",x+1, ": ", "%.16f"%yn,"t", x+1, ": ","%.2f"%tn)

	line1, = plt.plot(Vx, Vy, label='Backward K1')

def k3Backward(steps, tn, yn, fn, h):
	print ("\n Backward k=3 | Resultados:")

	print("Obtidos por Runge Kutta:")

	Vy = list()
	Vx = list()
	Vy.append(yn)
	Vx.append(tn)

	Ly, Lt = rungeKuttaWontPrint(3, tn, yn, fn, h)
	l3yn, l3tn = yn, tn
	l2yn, l2tn = Ly[0], Lt[0]
	l1yn, l1tn = Ly[1], Lt[1]
	yn, tn = Ly[2], Lt[2]

	print("y 1 :  %.16f"%l2yn,"t 1 :  %.2f"%l2tn)
	print("y 2 :  %.16f"%l1yn,"t 2 :  %.2f"%l1tn)
	print("y 3 :  %.16f"%yn,"t 3 :  %.2f"%tn)

	print("Obtidos por Backward:")
	for x in range(3, steps):

		yn1 = ((1/25.0)*(12*h*fn.subs([(y, y), (t, tn+h)]) + 48*yn -36*l1yn +16*l2yn -3*l3yn))

		aux = Eq(yn1, y)

		yn1 = solve(aux, y).pop()

		l3yn = l2yn
		l2yn = l1yn
		l1yn = yn
		yn = yn1
		l3tn = l2tn
		l2tn = l1tn
		l1tn = tn
		tn += h
		Vx.append(tn)		
		Vy.append(yn)

		print("y",x+1, ": ", "%.16f"%yn,"t", x+1, ": ","%.2f"%tn)

	line1, = plt.plot(Vx, Vy, label='Backward K3')

def exata(steps, tn, yn, str2, h):

	fn = sympify(str2)

	Vy = list()
	Vx = list()
	Vy.append(yn)
	Vx.append(tn)

	t = symbols("t")
	print("\n Função Exata: Resultados:")
	for x in range(0, steps+1):

		yn = fn.subs(t, tn)

		print("y",x+1, ": ", "%.16f"%yn,"t", x+1, ": ","%.2f"%tn)
		Vx.append(tn)		
		Vy.append(yn)

		tn += h


	line1, = plt.plot(Vx, Vy, label='Exata')



#lembrar de apagar esse hardcode dps e descomentar os input()
#tn = 0
#yn = 1
#h = 0.05
#tf = 0.4
print("\nOla, por favor insira a função exata, vulgo y")
str2 = input()
print("\nAgora, por favor insira f, vulgo y':")
str1 = input()
print("\nAgora, insira o valor inicial de t, t0:")
tn = input()
tn = float(tn)
print("\nO valor incial de y, y0:")
yn = input()
yn = float(yn)
print("\nO tamanho do passo, h:")
h = input()
h = float(h)
print("\nE nosso ponto de aplicacao objetivo, tf:")
tf = input()
tf = float(tf)
print("\nPor ultimo, por favor, determine qual o metodo:(no max 5 metodos diferentes)")

y, t = symbols("y t")
#str1 = "-3*t + 2*y + 1"
fn = sympify(str1)
Fn = sympify(str2)
steps = (tf-tn)/h
steps = int(steps)

counter = 0;
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
styles = ('b-', 'r-', 'r-', 'c-', 'm-', 'y-', 'k-')

while counter < 5:
	print("\n-1 - Para sair e plotar o grafico")
	print("1 - Euler Simples")
	print("2 - Euler Inverso")
	print("3 - Euler Aprimorado")
	print("4 - Runge Kutta")
	print("5 - Adam Bashforth")
	print("6 - Adam Moulton")
	print("7 - Adam Moulton com Previsao e Correcao")
	print("8 - Formula Inversa")
	op = input()
	op = int(op)

	if op == -1:
		break

	if op == 1:
		euler(steps, tn, yn, fn, h)
		exata(steps, tn, yn, str2, h)
	elif op == 2:
		eulerInv(steps, tn, yn, fn, h)
		exata(steps, tn, yn, str2, h)
	elif op == 3:
		eulerAprimorado(steps, tn, yn, fn, h)
		exata(steps, tn, yn, str2, h)
	elif op == 4:
		rungeKutta(steps, tn, yn, fn, h)
		exata(steps, tn, yn, str2, h)
	elif op == 5:
		print("Digite o valor do K, de 0 a 5")
		op1 = input()
		op1 = int(op1)
		if op1 == 0:
			k0Bash(steps, tn, yn, fn, h)	
			exata(steps, tn, yn, str2, h)		
		if op1 == 1:
			k1Bash(steps, tn, yn, fn, h)
			exata(steps, tn, yn, str2, h)
		if op1 == 2:
			k2Bash(steps, tn, yn, fn, h)
			exata(steps, tn, yn, str2, h)
		if op1 == 3:
			k3Bash(steps, tn, yn, fn, h)
			exata(steps, tn, yn, str2, h)
		if op1 == 4:
			k4Bash(steps, tn, yn, fn, h)
			exata(steps, tn, yn, str2, h)
		if op1 == 5:
			k5Bash(steps, tn, yn, fn, h)
			exata(steps, tn, yn, str2, h)
	elif op == 6:
		print("Digite o valor do K, de 0 a 5")
		op1 = input()
		op1 = int(op1)
		if op1 == 0:
			k0Moulton(steps, tn, yn, fn, h)
			exata(steps, tn, yn, str2, h)
		if op1 == 1:
			k1Moulton(steps, tn, yn, fn, h)
			exata(steps, tn, yn, str2, h)
		if op1 == 2:
			k2Moulton(steps, tn, yn, fn, h)
			exata(steps, tn, yn, str2, h)
		if op1 == 3:
			k3Moulton(steps, tn, yn, fn, h)
			exata(steps, tn, yn, str2, h)
		if op1 == 4:
			k4Moulton(steps, tn, yn, fn, h)
			exata(steps, tn, yn, str2, h)
		if op1 == 5:
			k5Moulton(steps, tn, yn, fn, h)
			exata(steps, tn, yn, str2, h)
	elif op == 7:
		print("Digite o valor do K, de 0 a 5")
		op1 = input()
		op1 = int(op1)
		if op1 == 0:
			k0MoultonP(steps, tn, yn, fn, h)
			exata(steps, tn, yn, str2, h)
		if op1 == 1:
			k1MoultonP(steps, tn, yn, fn, h)
			exata(steps, tn, yn, str2, h)
		if op1 == 2:
			k2MoultonP(steps, tn, yn, fn, h)
			exata(steps, tn, yn, str2, h)
		if op1 == 3:
			k3MoultonP(steps, tn, yn, fn, h)
			exata(steps, tn, yn, str2, h)
		if op1 == 4:
			k4MoultonP(steps, tn, yn, fn, h)
			exata(steps, tn, yn, str2, h)
		if op1 == 5:
			k5MoultonP(steps, tn, yn, fn, h)
			exata(steps, tn, yn, str2, h)
	elif op == 8:
		print("Digite o valor do K, entre 0, 1 e 3")
		op1 = input()
		op1 = int(op1)
		if op1 == 0:
			k0Backward(steps, tn, yn, fn, h)
			exata(steps, tn, yn, str2, h)
		if op1 == 1:
			k1Backward(steps, tn, yn, fn, h)
			exata(steps, tn, yn, str2, h)
		if op1 == 3:
			k3Backward(steps, tn, yn, fn, h)
			exata(steps, tn, yn, str2, h)

	counter+=1

plt.legend(loc='upper left')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Metodos Numericos')

plt.show()