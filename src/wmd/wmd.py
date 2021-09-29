# this is the file for the python3 compatible earth mover's distance
import pdb, sys, numpy as np, pickle
from multiprocessing import Pool
import pickle

import os
from datetime import datetime

sys.path.append('../util')
sys.path.append('../preprocessing')


from scipy.spatial.distance import pdist, squareform
from pyemd import emd


# Calculate the similarity (WMD/EMD) between two pickle files
# requires:
#	[comparisonpair]			: [(word_vectors1, word_weights1), (word_vectors2, word_weights2)]
#	[distance_function_string]	: <str>: 'cosine' for cosine distance, or 'euclidean' for euclidean distance based WMD
def calc_similarity(comparisonpair, distance_function_string = "cosine"):
	# load pickle files X, BOW_X = (word_vector_arrays, BOW-features)
	word_vectors1, word_weights1 = comparisonpair[0]
	word_vectors2, word_weights2 = comparisonpair[1]

	# word_weights1 and word_weights2 correspond to histograms in the pyemd library

	# CAVEAT! There seems to be an issue with the correctness of the distance metric
	# Also the current implementation foresees that the signature sizes are equal!
	# !!!https://github.com/wmayner/pyemd/issues/32
	# YET it seems as this is the only python3 version ... for multi-dimensional vectors (for 1D refer to scipy's wasserstein distance)
	# check if both files users are identical
	if (word_vectors1 == word_vectors2) and (word_weights1 == word_weights2):
		return 1.0
	# else
	else:
		# calculate the earth mover's distance (EMD) between two 'signatures' (generalized distributions)
		# signature format: (list of vectors [number of vectors x embedding dimension], list of their weights)
		# with the cosine distance
		# cast to narray
		word_vectors1 = np.array(word_vectors1)
		word_vectors2 = np.array(word_vectors2)

		# currently the pyemd library only accepts signatures of the same size! this is not a problem though
		# we first build a matrix of np.concatenate(word_vectors1, word_vectors2) on the zero-axis
		# [word_vectors1,
		#  word_vectors2] as a numpy array
		# then, the resulting distance matrix contains all pairwise distances, we are only interested in those between word_vectors1 and word_vectors2
		concat_word_vectors = np.concatenate([word_vectors1, word_vectors2])

		# the weight_vectors have to be adjusted accordingly
		# word_weights1 --> [word_weights1 , zeros for wordweights2]
		# word_weights2 --> [zeros for wordweights1, word_weights2]
		concat_word_weights1 = np.concatenate([word_weights1								, [0 for _ in range(len(word_weights2))]])
		concat_word_weights2 = np.concatenate([[0 for _ in range(len(word_weights1))]	, word_weights2])

		# create the distance matrix as a matrix in squareform
		# distances are measured with respect to the EUCLIDEAN DISTANCE (!)
		# for alternatives see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist
		distance_matrix = squareform(pdist(concat_word_vectors, metric = distance_function_string))

		emd_distance = emd(concat_word_weights1, concat_word_weights2, distance_matrix)

		# convert EMD 'distances' into 'similarities' in the interval [0,1]
		# high distance -> low similarity; short distance -> high similarity
		similarity = 1-emd_distance/2
		# REMARK 1:
		# if distance_function_string == "cosine": the <word_weights1> and <word_weights_2> have to sum up to 1.0 respectively
		# 							in this case: emd_distance is in the interval [0,2]
		# REMARK 2:
		# if distance_function_string == "euclidean": the <word_weights1> and <word_weights_2> have to sum up to 1.0 respectively
		#											AND the vectors in <word_vectors1> and <word_vectors2> all have euclidean norm 1.0
		#								 in this case: emd_distance is in the interval [0,2]

		return similarity

# use this in order to specify a distance function (for parallelization) in calcPairwiseDist
def calc_similarity_cosine(comparisonpair):
	return calc_similarity(comparisonpair, distance_function_string = "cosine")
# use this in order to specify a distance function (for parallelization) in calcPairwiseDist
def calc_similarity_euclidean(comparisonpair):
	return calc_similarity(comparisonpair, distance_function_string = "euclidean")



# load a pickle file from a pickle_path
def load_pickle(pickle_path):
    # load both users' pickle files
    with open(pickle_path, 'rb') as f:
        pickle_load = pickle.load(f)
    return pickle_load



def make_pairs(files, files2 = None):
	""" For either one (!) OR two (!) lists create pairs of files.
	If:
	(a) only one list is given:
				calculate pairs as the upper triangle of the pair matrix [(p_i, p_j)]_{i,j} for files files f_i, f_j in files
	(b) both comparison files are given:
				Calculate pairs between comparison_files and comparison_files2 list
	"""
	pairs = []
	if files2:
		N = len(files)
		M = len(files2)
		for i in range(N):
			for j in range(M):
				pairs.append((files[i], files2[j]))

	else:
		N = len(files)
		for i in range(N):
			for j in range(i+1,N):
				pairs.append((files[i], files[j]))
	return pairs

def fetch_pickle_paths(pickle_dir):
	""" Fetch all pickle (*.pk) files from the pickle_dir as a <list> of <str>"""
	pickle_paths = []
	for root, dirs, files in os.walk(pickle_dir):
		for name in files:
			if os.path.splitext(name)[1] == '.pk':
				pickle_paths.append(os.path.join(root,name))
	return pickle_paths

def calc_similarities(comparison_pairs, distance_function_string = "cosine", processes = 4):
	""" Calculate pairwise similarities for a list of comparison pairs (pair of signatures) in parallel."""
	pool = Pool(processes=processes)
	num_tasks = len(comparison_pairs)

	similarities = []
	# switch between "cosine" and "euclidean" distance

	if distance_function_string == "cosine":
		run_calc = calc_similarity_cosine
	elif distance_function_string == "euclidean":
		run_calc = calc_similarity_euclidean
	else:
		print("unknown distance function string")
		run_calc = None

	for  i, sim in enumerate(pool.map(run_calc, comparison_pairs), 1):
		sys.stderr.write('\rCalculated {}/{}({})% of all similarities'.format(i,num_tasks,round(i/num_tasks*100),2)) #0.%
		similarities.append(sim)
	print('')
	return similarities


def calcPairwiseDist(pickle_dir, similarity_dir, pickle_dir2 = None, distance_function_string = "cosine", from_dir = True, experiment_name = "",
		processes = 4, output_format = "gephi", make_pickle = False):
	"""# export is a (Gephi) edge graph
	# requires :
		[pickle_dir] folder with the corresponding pickle files to compair pairwise: if from_dir == True// else it is a list of paths to pickle files
		[similarity_dir]
		[distance_function_string]
		[from_dir]
		[experiment_name]
		[processes]
		[output_format] : str, one of ["gephi", "sim_dict", None]
	"""
	# 0. find all files in 'pickle_dir' (this works recursively - and all subfolders are searched) and 'pickle_dir2' if specified
	# if from dir:
	if from_dir == True:
		pickle_paths   = fetch_pickle_paths(pickle_dir)
		if pickle_dir2:
			pickle_paths2 = fetch_pickle_paths(pickle_dir2)
	# else: pickle_dir
	else:
		pickle_paths = pickle_dir
		pickle_paths2 = pickle_dir2

	#print('pickle_paths', pickle_paths)

	# 1. load comparison files
	# comparison_files format : [<list> of <list> as a list of vectors, <list> of <float>s as a list of weights]
	comparison_files = [load_pickle(pickle_path) for pickle_path in pickle_paths]
	if pickle_paths2:
		comparison_files2 = [load_pickle(pickle_path) for pickle_path in pickle_paths2]
	else:
		comparison_files2 = None

	# 2. Generate comparison pairs and pickle path pairs
	comparison_pairs = make_pairs(comparison_files, files2 = comparison_files2)

	# NOTE: If you want to parallelize this, you will have to add the fileIDs to the similarity output to the 'calc_similarity' method [fuse picklepath_pairs and comparison_pairs]
	# 3. For all comparison pairs run emd (Earth Mover's Distance)
	similarities = calc_similarities(comparison_pairs, distance_function_string = distance_function_string, processes = processes) #### ADD

	# 4.Generate pickle path pairs (as node IDs)
	picklepath_pairs = make_pairs(pickle_paths, files2 = pickle_paths2)

	# 5. Convert to gephi edge-graph format
	# source, target, weight
	# ID1, ID2, weight1-2
	# ID1, ID3, weight1-3
    # ...

	if output_format is None:
		result = str(list(zip(picklepath_pairs, similarities)))[1:-1]
	else:
		if output_format == "gephi":
			result = "source,target,weight\n"
			for picklepath_pair, similarity in zip(picklepath_pairs, similarities):
				filename1 = os.path.splitext(os.path.basename(str(picklepath_pair[0])))[0]
				filename2 = os.path.splitext(os.path.basename(str(picklepath_pair[1])))[0]

				line   = filename1 + ',' + filename2 + ',' + str(similarity)+'\n'
				result += line

		elif output_format == "sim_dict":
			# sim_dict format: { "peer1-peer2" : sim(peer1, peer2), ...}
			result = {}

			for picklepath_pair, similarity in zip(picklepath_pairs, similarities):
				filename1 = os.path.splitext(os.path.basename(str(picklepath_pair[0])))[0]
				filename2 = os.path.splitext(os.path.basename(str(picklepath_pair[1])))[0]

				key = filename1 + '-' + filename2
				value = float(similarity)
				result[key] = value

	# define a timesting (hour_minute_second) in order to specify the time the sim_file has been generated, NOT WORKING CURRENTLY
	time_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
	out_file_name = os.path.join(similarity_dir, time_string + '_' + experiment_name + '_' + distance_function_string + '_sims.pk')

	if make_pickle == False:
		with open(out_file_name, mode = 'w') as f:
			try:
				f.write(result)
				print('Wrote a file with pairwise similarities')
			except TypeError:
				print('Need type string, not {} of <result>. Please choose for instance output_format = "gephi" '.format(type(result)))
	else:
		# Pickle the result
		with open(out_file_name, 'wb') as f:
			pickle.dump(result, f)
			f.close()


	return out_file_name



def main():
	"""
	# calculate all pairwise distances of pickle files in a pickle_dir
    pickle_dir     = sys.argv[1]
    similarity_dir = '../../data/similarityfiles/'

    # calculate all pairwise distances between pickle files (prepared by get_word_vectors.py --> should probably be renamed)
    calcPairwiseDist(pickle_dir, similarity_dir)
	"""

	"""Calculate the similarity between two word-vector, word-weight pairs."""
	"""
	word_vectors1 = [[-0.2,0.1,0.55],[0.11, -0.57, 0.49]]
	word_vectors2 = [[-0.25,0.5,0.75],[0.23,0.51,0.6]]
	word_weights1 = [6/11, 5/11]
	word_weights2 = [8/13, 5/13]
	distance = cosine_distance
	print(emd( (word_vectors1, word_weights1), (word_vectors2, word_weights2), distance))
	"""
	pickle_dir     	= '../../data/picklefiles/test'
	similarity_dir 	= '../../data/similarityfiles/'


	#t = calcPairwiseDist(pickle_dir, similarity_dir, distance_function_string = "cosine", from_dir = True, experiment_name = "", processes = 4)

	print(make_pairs([1,2,3], files2= [4,5]))



if __name__ == "__main__":
	main()
