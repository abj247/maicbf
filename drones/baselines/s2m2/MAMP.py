# Run benchmarks 

import argparse
import sys
from algs.centralized import centralized_algo
from algs.decentralized import decentralized_algo

parser = argparse.ArgumentParser()
parser.add_argument("models", nargs='+')
parser.add_argument("--env")
parser.add_argument("--min_segs")
parser.add_argument("--max_segs")
parser.add_argument("--method")
parser.add_argument("--viz")
args = parser.parse_args()

# check method (Default = Centralized)
if args.method == None: method = "decentralized"
else: method = args.method

# check the number of line segments
if args.max_segs == None: max_segs = 100
else: max_segs = int(args.max_segs)
if args.min_segs == None: min_segs = 3
else: min_segs = int(args.min_segs)

# Plot the problem and the test runs
if __name__ == '__main__':
	if method == "centralized":
		trajs = centralized_algo(args.models, args.env, min_segs, max_segs)
	elif method == "decentralized":
		trajs = decentralized_algo(args.models, args.env, min_segs, max_segs)
	if bool(args.plot) == True:
		True
	else: sys.exit()