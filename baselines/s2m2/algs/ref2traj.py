# Ref2Traj

import numpy as np
from numpy.linalg import norm

def get_xref(p1, p2, times):

	dim = len(p1)
	t1, t2 = times[0], times[-1]

	if dim == 2:
		mx = p2[0] - p1[0]
		bx = p1[0]

		my = p2[1] - p1[1]
		by = p1[1]

		theta = np.arctan2((np.array(p2) - np.array(p1))[1], (np.array(p2) - np.array(p1))[0])

		qref = [] # append the states to this list
		for time in times:
			xref = mx*((time - t1) / (t2 - t1)) + bx
			yref = my*((time - t1) / (t2 - t1)) + by
			qref.append([xref, yref, theta])

	elif dim == 3:
		mx = p2[0] - p1[0]
		bx = p1[0]

		my = p2[1] - p1[1]
		by = p1[1]

		mz = p2[2] - p1[2]
		bz = p1[2]

		roll = 0
		pitch = np.arctan2((np.array(p2) - np.array(p1))[2], norm(np.array(p2)[0:2] - np.array(p1)[0:2]))
		yaw = np.arctan2((np.array(p2) - np.array(p1))[1], (np.array(p2) - np.array(p1))[0])

		qref = []
		for time in times:
			xref = mx * ((time - t1) / (t2 - t1)) + bx
			yref = my * ((time - t1) / (t2 - t1)) + by
			zref = mz * ((time - t1) / (t2 - t1)) + bz
			qref.append([xref, yref, zref, roll, pitch, yaw])

	else:
		print('Nodes not in workspace!')

	return qref

def get_uref(p1, p2, times):
	dist = norm(np.array(p2) - np.array(p1))
	v_ref = dist / (times[-1] - times[0])
	dim = len(p1)
	if dim == 2:
		uref = []
		for _ in times:
			vref = v_ref
			wref = 0
			uref.append([vref, wref])
	elif dim == 3:
		uref = []
		for _ in times:
			vref = v_ref
			roll_ref = 0
			pitch_ref = 0
			yaw_ref = 0
			uref.append([vref, [roll_ref, pitch_ref, yaw_ref]])
	else:
		print('Nodes not in workspace!')

	return uref

def ref2traj(ma_nodes):
	dim = len(ma_nodes[0][0])-1
	agent_num = len(ma_nodes)

	# compute step size
	step_num = 10000
	max_t = max([ma_nodes[idx][-1][0] for idx in range(agent_num)])

	step_size = max_t / step_num

	ref_trajs = []
	# calculate controller
	for idx in range(agent_num):
		ref_traj = []
		nodes = ma_nodes[idx]
		for j in range(len(nodes) - 1):
			s1, s2 = nodes[j:j+2]
			t1, t2 = s1[0], s2[0]
			p1, p2 = s1[1:], s2[1:]
			times = np.arange(t1, t2, step_size)
			if times != []:
				qref = get_xref(p1, p2, times)
				uref = get_uref(p1, p2, times)
				ref_traj.append([times, qref, uref])
		# keep still
		times = np.arange(ma_nodes[idx][-1][0], max_t, step_size)
		p = ma_nodes[idx][-1][1:]
		if times != []:
			qref = get_xref(p, p, times)
			uref = get_uref(p, p, times)
			ref_traj.append([times, qref, uref])
		ref_trajs.append(ref_traj)

	return ref_trajs
