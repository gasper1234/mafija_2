from scipy.integrate import solve_ivp
from scipy.integrate import cumtrapz
import numpy as np
import matplotlib.pyplot as plt
from data import *

C_list = np.linspace(3000, 10000, 200)


iconds=np.array([1., 0.])
def gen(r, state):

    dydt=np.zeros_like(state)
    dydt[0]=state[1]
    dydt[1]=-C*state[0]*(1.-r**2)-state[1]/r
    return dydt

def eval(C):
	iconds=np.array([1., 0.])

	def gen_ivp(r, state):

	    dydt=np.zeros_like(state)
	    dydt[0]=state[1]
	    dydt[1]=-C*state[0]*(1.-r**2)-state[1]/r
	    return dydt

	res1 = solve_ivp(gen_ivp, [10**(-5), 1], iconds)
	#solve_ivp(gen_ivp, [10**(-5), 1], iconds,method='DOP853',rtol=1e-12,atol=1e-12)
	return res1.y[0][-1]


def bisec(f, x1, x2, y1, y2):
	for i in range(25):
		x_mid = (x1 + x2)/2
		y_mid = f(x_mid)
		if y_mid*y1 < 0:
			x2 = x_mid
			y2 = y_mid
		else:
			x1 = x_mid
			y1 = y_mid
	return [x1, x2]

last_val = []
'''
#grobo določi
for i in range(len(C_list)):
	C = C_list[i]
	res1 = solve_ivp(gen, [10**(-8), 1], iconds,method='DOP853',rtol=1e-14,atol=1e-14)
	last_val.append(res1.y[0][-1])

intervali_nicel = []

for i in range(len(last_val)-1):
	if last_val[i]*last_val[i+1] < 0:
		intervali_nicel.append([C_list[i], C_list[i+1]])

#natačno določi
for i in range(len(intervali_nicel)):
	x = bisec(eval, intervali_nicel[i][0], intervali_nicel[i][1], eval(intervali_nicel[i][0]), eval(intervali_nicel[i][1]))
	print('interval', x)
	intervali_nicel[i] = (x[0]+x[1])/2
'''

for i in range(len(intervali_nicel)):
	x = intervali_nicel[i]
	intervali_nicel[i] = (x[0]+x[1])/2


#za že narejen seznam

func_list = np.array([[0. for _ in range(100)] for _ in range(len(intervali_nicel))])
t_list = np.linspace(10**(-8), 1, 100)
print(intervali_nicel)

#najde rešitve
for i in range(len(intervali_nicel)):
	C = intervali_nicel[i]
	res = solve_ivp(gen, [10**(-8), 1], iconds, t_eval=t_list, method='DOP853',rtol=1e-14,atol=1e-14)
	func_list[i] += res.y[0]
	#plt.plot(res.t, res.y[0], label=intervali_nicel[i])
#plt.legend()
#plt.show()

#preveri ortogonalnost
for i in range(len(func_list)):
	#plt.plot(t_list, func_list[1]*func_list[i]*t_list*(1-t_list**2))
	a = cumtrapz(func_list[0]*func_list[0]*t_list*(1-t_list**2), t_list)[-1]
	print('ortogonal', cumtrapz(func_list[0]*func_list[i]*t_list*(1-t_list**2), t_list)[-1]/a)
#plt.show()

T_koef = [0 for i in range(len(intervali_nicel))]

for i in range(len(intervali_nicel)):
	norm = cumtrapz(func_list[i]*func_list[i]*t_list*(1-t_list**2), t_list)[-1]
	T_koef[i] = cumtrapz(func_list[i]*t_list*(1-t_list**2), t_list)[-1] / norm

print(T_koef)

bound = np.array([0. for _ in range(len(func_list[0]))])
for i in range(len(T_koef)):
	bound += func_list[i]*T_koef[i]

plt.plot(t_list, bound)
plt.show()


'''
def bisec(f, x1, x2, y1, y2, tol):
	print(1)
	if abs(x1-x2) < tol:
		return [x1, x2]
	x_mid = x1 + x2
	y_mid = f(x_mid)
	print(x1, x2)
	if y_mid*y1 < 0:
		return bisec(f, x1, x_mid, y1, y_mid, tol)
	else:
		return bisec(f, x_mid, x2, y_mid, y2, tol)
		'''