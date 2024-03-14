from math import *
from scipy.integrate import odeint
from models.agent import *
import random

class Car(Agent):

	def __init__(self, size, velocity, k):
		self.size = size
		self.velocity = velocity
		self.k = k

	def model(self, q, t, u):
		x, y, theta = q
		v, w = u

		dxdt = v * cos(theta)
		dydt = v * sin(theta)
		dthetadt = w
		return [dxdt, dydt, dthetadt]

	def controller(self, q, qref, uref):
		x, y, theta = q
		xref, yref, thetaref = qref
		vref, wref = uref
		k1, k2, k3 = self.k

		xe = cos(theta) * (xref - x) + sin(theta) * (yref - y)
		ye = -sin(theta) * (xref - x) + cos(theta) * (yref - y)
		thetae = thetaref - theta
		v = vref * cos(thetae) + k1 * xe
		w = wref + vref * (k2 * ye + k3 * sin(thetae))

		return [v, w]

	def bloating(self, n):
		return 0
		k1, k2, k3 = self.k
		if n != 0:
			return sqrt(4 * n / k2)
		else:
			return sqrt(4 / k2)

	def run_model(self, q0, t, qref, uref):
		q = [q0]
		u0 = [0, 0]
		for i in range(0,len(t)):
			t_step = [t[i-1], t[i]]

			q1 = odeint(self.model, q0, t_step, args = (u0,)) # + [random.uniform(-0.005, 0.005),
															 #	 random.uniform(-0.005, 0.005),
											 				#	 random.uniform(-0.005, 0.005),]
			q0 = q1[1]
			q.append(q0)
			u0 = self.controller(q0, qref[i], uref[i])
		return q