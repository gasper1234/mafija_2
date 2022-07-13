from scipy.integrate import solve_ivp
from scipy.integrate import cumtrapz
import numpy as np
import matplotlib.pyplot as plt

C_list = np.linspace(1, 300, 300//10)


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
	for i in range(15):
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

for i in range(len(C_list)):
	C = C_list[i]
	res1 = solve_ivp(gen, [10**(-8), 1], iconds,method='DOP853',rtol=1e-14,atol=1e-14)
	last_val.append(res1.y[0][-1])

intervali_nicel = []

for i in range(len(last_val)-1):
	if last_val[i]*last_val[i+1] < 0:
		intervali_nicel.append([C_list[i], C_list[i+1]])

for i in range(len(intervali_nicel)):
	x = bisec(eval, intervali_nicel[i][0], intervali_nicel[i][1], eval(intervali_nicel[i][0]), eval(intervali_nicel[i][1]))
	print(x)
	intervali_nicel[i] = (x[0]+x[1])/2

func_list = np.array([[0. for _ in range(100)] for _ in range(len(intervali_nicel))])
t_list = np.linspace(10**(-8), 1, 100)
print(intervali_nicel)
for i in range(len(intervali_nicel)):
	C = intervali_nicel[i]
	res = solve_ivp(gen, [10**(-8), 1], iconds, t_eval=t_list, method='DOP853',rtol=1e-14,atol=1e-14)
	func_list[i] += res.y[0]
	#plt.plot(res.t, res.y[0], label=intervali_nicel[i])
#plt.legend()
#plt.show()

for i in range(len(func_list)):
	print(1)
	plt.plot(t_list, func_list[0]*func_list[i]*t_list)
	print(cumtrapz(func_list[0]*func_list[i]*t_list, t_list))
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