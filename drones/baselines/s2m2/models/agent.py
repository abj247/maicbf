
class Agent:
	def __init__(self, size, velocity, k):
		self.size = size
		self.velocity = velocity
		self.k = k

	def model(self, q, t, u):
		pass

	def controller(self, q, qref, uref):
		pass

	def bloating(self, n):
		pass

	def run_model(self, q0, t, qref, uref):
		pass