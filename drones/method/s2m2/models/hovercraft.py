# 3d Kinematic Model
# Written by: Kristina Miller

from math import *
from scipy.integrate import odeint
from models.agent import *

class Hovercraft(Agent):

	def __init__(self, size, velocity, k):
		self.size = size
		self.velocity = velocity
		self.k = k

	def model(self, q, t, u):
		x, y, z, theta = q
		v, vz, w = u
		dxdt = v*cos(theta)
		dydt = v*sin(theta)
		dzdt = vz
		dthetadt = w
		return [dxdt, dydt, dzdt, dthetadt]

	def controller(self, q, qref, uref):

		k1, k2, k3, kp = self.k
		xref, yref, zref, rollref, pitchref, yawref = qref
		x, y, z, theta = q

		vxyref = uref[0]*cos(pitchref)
		vzref = uref[0]*sin(pitchref)
		_, _, uyaw_ref = uref[1]

		xe = cos(theta) * (xref - x) + sin(theta) * (yref - y)
		ye = -sin(theta) * (xref - x) + cos(theta) * (yref - y)
		ze = zref - z
		thetae = pitchref - theta

		vxy = vxyref * cos(thetae) + k1 * xe
		w = uyaw_ref + vxyref * (k2 * ye + k3 * sin(thetae))
		vz = vzref + kp * ze

		return [vxy, vz, w]

	def bloating(self, n):
		return 0.15
		k1, k2, k3, kp = self.k
		if n != 0:
			return sqrt(4*n/k2)
		else:
			return 0

	def run_model(self, q0, t, qref, uref):
		q = [q0]
		u0 = [0, 0, 0]
		for i in range(0,len(t)):
			t_step = [t[i-1], t[i]]
			q1 = odeint(self.model, q0, t_step, args = (u0,))
			q0 = q1[1]
			# print(q0)
			q.append(q0)
			u0 = self.controller(q0, qref[i], uref[i])
		return q
