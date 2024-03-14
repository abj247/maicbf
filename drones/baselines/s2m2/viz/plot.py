from viz.util import *
from algs.collision import *
import os

def plot_results(models, limits, Obstacles, Thetas, Goals, ma_segs, name, MO = [], refs = None):
	plt.close('all')
	agent_num = len(models)
	fig, axes = plot_env(limits, Obstacles)
	# plot_goals(Goals)
	# plot_thetas(Thetas)

	# for mo in MO:
	# 	for obs in mo:
	# 		A = obs[2]
	# 		b = np.array([[e] for e in obs[3]])
	# 		plot_polytope_3d(A, b, axes)

	paths = extract_paths(models, Thetas, ma_segs)
	for idx in range(agent_num):
		if len(Thetas[idx]) == 2:
			ref_x, ref_y, _, tru_x, tru_y, _, _ = paths[idx]
			ref_x, ref_y, _, tru_x, tru_y, _, _ = paths[idx]
			# axes.plot(ref_x, ref_y, color='k', linewidth = 2, linestyle = 'dashed')
			# axes.plot(tru_x, tru_y, color='purple', linewidth = 3, linestyle='dashed',)
			if refs is not None:
				x = [p[1] for p in refs[idx]]
				y = [p[2] for p in refs[idx]]
				axes.plot(x, y,color='k', linestyle = 'dashed', linewidth = 2, marker = 'o')
		elif len(Thetas[idx]) == 3:
			ref_x, ref_y, ref_z, tru_x, tru_y, tru_z, _, _, _, _ = paths[idx]
			axes.plot(ref_x, ref_y, ref_z, color='k')
			axes.plot(tru_x, tru_y, tru_z, 'r--')

	fig.show()
	path = os.path.abspath("results/%s.svg" % (name))
	fig.savefig(path, bbox_inches='tight', pad_inches = 0)
	return None

