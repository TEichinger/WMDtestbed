import sys, os
sys.path.append('./')
sys.path.append('../')
from util.utilities import load_pickle,log_classifier_accuracy, get_file_paths, get_labels, get_current_timestamp
from wmd.wmd import calc_similarities, load_pickle, calcPairwiseDist
from preprocessing.preprocess import make_pickle_files
from evaluation.evaluate_classifier import evaluate_classifier_log


import pandas as pd

import argparse
import subprocess
	
	
# log the to_label_file_path, and its closest labels in the format
# <to_label_file_path>:<1st_label> <2nd_label> ... <keep_top-th_label>
def log_sorted_labels(log_path, to_label_file_path, sorted_labels):
	log_line = to_label_file_path + ":"
	for sorted_label in sorted_labels:
		log_line += " "
		log_line += str(sorted_label)
		
	with open(log_path, mode = "w") as f:
		f.write(log_line)
	
	return



	
	
# predict labels fast by calculating a large matrix of distances instead of columns in a for loop
# predict labels for a list of 'to_label_file_paths' via knowledge points
# [to_label_file_paths]		: <list> of file paths to text files to infer a label on
# [knowledge_signatures]	: <list> of signatures (see extract_signatures)
# [knowledge_labels]		: <list> of integers (1: positive, 0: negative, -1: unknown)
# [n_neighbors]				: <int> number of neighbors to consider for label inference
# [we_model_path]			: <str> path to fasttext model (.bin)
# [sigSize]					: <int>
# [minDf]					: <float>
# [maxDf]					: <float>
# [sigVecStrat]				: <str>
# [sigWeightStrat]			: <str>
# [vectorNormalization]		: <bool>
# [weightNormalization]		: <str>
# [verbose]					: print details if True
# [processes]				: <int> number of processes to consider for parallelization
# [batch_size]				: <int> batch size for WMD distance calculation (every batch consists of len(knowledge_signatures) x batch_size elements)
# [keep_top]				: <int> number of most similar labels to keep for every to_label_file_path
# output: output_labels (relative to n_neighbors)
#			- result log that has to be parsed after all the inference has happened, this should be quick however, and also be feasible with say target
#			- files to infer a label on
def predict_labels(to_label_file_paths, knowledge_file_paths, knowledge_labels, pickle_dir, log_path, tfidf_triple_name, n_neighbors, we_model_path, sigSize, minDf, maxDf, \
					distance_function_string, sigVecStrat, sigWeightStrat, vectorNormalization, weightNormalization, verbose = False, processes = 4, batch_size = 1000, keep_top = 55, skip_pickling = False, precalculated_dir = None):
	m = len(knowledge_file_paths)
	n = len(to_label_file_paths)
	print("m,n:{}, {}".format(m,n))
	print("Clear the classifier log..")
	with open(log_path, mode = "w") as f:
		pass
		
	# MAKE PICKLE FILES FOR KNOWLEDGE FILES AND TO_LABEL_FILES
	##########################################################
	# make pickle files for knowledge files
	knowledge_pickle_paths = make_pickle_files(knowledge_file_paths, pickle_dir, we_model_path, sig_size = sigSize, sigvec_strat = sigVecStrat , sigweight_strat = sigWeightStrat, min_df = minDf , max_df = maxDf, vector_normalization = vectorNormalization)
	# make pickle files for to_label_files
	to_label_pickle_paths  = make_pickle_files(to_label_file_paths, pickle_dir, we_model_path, sig_size = sigSize, sigvec_strat = sigVecStrat , sigweight_strat = sigWeightStrat, min_df = minDf , max_df = maxDf, vector_normalization = vectorNormalization)

	
	# create batches of to_label_pickle_paths
	if len(to_label_pickle_paths)%batch_size == 0:
		num_batches = len(to_label_pickle_paths)//batch_size
	else:
		num_batches = (len(to_label_pickle_paths)//batch_size)+1

	to_label_pickle_batches = [to_label_pickle_paths[i*batch_size : (i+1)*batch_size] for i in range(num_batches)]
	
	# extract knowledge signatures
	knowledge_signatures = [load_pickle(el) for el in knowledge_pickle_paths]
	
	for i,batch in enumerate(to_label_pickle_batches, 1):
		print("Calculate {}/{} batch.".format(i, len(to_label_pickle_batches)))
		# extract to_label_signature
		to_label_signatures_batch = [load_pickle(el) for el in batch]
		
		# create comparison pairs of loaded pickle files (signatures)
		# the comparison pairs will be indices of the form [(0,0), (1,0), (2,0),...], therefore the distances will be sorted by to_label_file
		# such that every m pairs, a new to_label_file is considered for inference
		comparisonpairs = []
		for to_label_signature in to_label_signatures_batch:
			for knowledge_signature in knowledge_signatures:
				comparisonpairs.append((to_label_signature, knowledge_signature))
		
		print("comparisonpairs",len(comparisonpairs))
		# calculate all similarities
		# note that we have to go for batches here [when a single pickle file has about 350 kb
		# and we have to calculate m x n distances, then we need m x n x 350 x 2 kb.
		# for a fixed m = 100, say, and a batch size of 1000 instead of n, we have a RAM requirement of
		# about 35 GB
		# here it is up for future experiments to see if this is tractable or not
		# with a lower signature size, we also have lower RAM requirements, ergo we can increase the batch size
		similarities = calc_similarities(comparisonpairs, distance_function_string = distance_function_string, processes = processes)

		# keep only the keep_top elements [ m x batch_size ] --> [keep_top x batch_size] similarities
		#########################
		# the resulting top closest labels should be output into a file
		per_to_label_file_similarities = [similarities[i*m:(i+1)*m] for i in range((len(similarities)//m) )]

		# for every to_label_file
		for to_label_file_path, similarities in zip (batch, per_to_label_file_similarities):
			# sort the knowledge files with labels according to the calculated similarities
			similarities_sorted_withlabels = sorted(zip(similarities, knowledge_labels), reverse = True)
			# drop all but the top 'keep_top'
			sorted_labels = [el[1] for el in similarities_sorted_withlabels][:keep_top]
			# log the labels only to a temporary file
			log_sorted_labels(log_path, to_label_file_path, sorted_labels)
	
	print("Logged all results")
	
	return
	
	
# log the sorted top labels according to the instance-based
# [log_path]			:
# [to_label_file_path]	:
# [sorted_labels]		:
# format: 'to_label_file_path' 'pred_label_1' 'pred_label_1' 'pred_label_1' 'pred_label_1'
def log_sorted_labels(log_path, to_label_file_path, sorted_labels, append = True):
	# clear the log_path, if append is not True
	if not append:
		with open(log_path, mode = "w") as f:
			pass
		
	# create a log line
	# start the line with the to_label_file_path (without extension)
	to_label_file_name = os.path.split(to_label_file_path)[-1]
	line = os.path.splitext(to_label_file_name)[0] + " "
	# add blank-separated prediction labels
	for sorted_label in sorted_labels:
		line += str(sorted_label)
		line += " "
	line += "\n"
	
	# append a line stating the calculated predictions
	with open(log_path, mode = "a") as f:
		f.write(line)
		
	return
		

		
def create_label_dir(path_list, label_dir):
	""" Tranform a list of filepaths to assign a label/calculate a distance to into a directory holding the respective files
	to run the main function on.
	requires:
		[path_list]: <list> of <str> paths to the files to assign a label/calculate a distance to
		[label_dir]: <str> directory path to fill
	"""
	# make label_dir directory, if it does not yet exist
	if not os.path.isdir(label_dir) :
		os.mkdir(label_dir)
	
	# for all paths in path_list
	for source_path in path_list:
		target_path = os.path.join(label_dir, os.path.split(source_path)[1])
		# copy the path into the label_dir
		subprocess.call(["cp", source_path, target_path])
	print("Copied files to {}.".format(label_dir))
	return
	
def read_histogram_file(histogram_file_path, histories_dir):
	""" Read a histogram file created e.g. by ../db_interaction/mongo.py -- <calculate_distance_distribution> function and extract all reviewer IDs in
	the index column.
	Format: 
		, cosine_sim_to_IDX
	ID1, 0.5141243
	ID2, 0.5131241
	...
	requires:
		[histogram_file_path]:
		[histories_dir]: directory that holds the .txt files corresponding to the IDs specified in 'histogram_file_path' (without file extension)
	"""
	df = pd.read_csv(histogram_file_path, index_col = 0)
	# extract indices
	indices = list(df.index)
	# assemble path list [histories_dir + index + ".txt"]
	path_list = [os.path.join(histories_dir, index) + ".txt" for index in indices]
	
	return path_list
	

def main():
	# PARSE ARGS
	#############
	parser = argparse.ArgumentParser()
	parser.add_argument("knowledge_dir", help="path to the documents to be used for the classifier") 	
	parser.add_argument("label_dir", help="path to the documents to be used to be labeled/or to which WMD distances are to be calculated if unsupervised is '1'") 			
	parser.add_argument("model"       , help="path to the binary (.bin) word embedding model to use" 											, action = "store")
	parser.add_argument("processes"   , help="number of parallel pool processes"																, action = "store")
	parser.add_argument("sig_size"    , help="signature size for the WMD"	                													, action = "store")
	parser.add_argument("sig_vec_strat"    , help="string specifying the signature vector strategy; first k by default"							, action = "store")
	parser.add_argument("sig_weight_strat" , help="string specifying the signature weight strategy; 1/k by default, where n the signature size"	, action = "store")
	parser.add_argument("classifier_log" , help="path to the log files"	, action = "store")
	parser.add_argument("positive_dir" , help="path to the positive files"	, action = "store")
	parser.add_argument("negative_dir" , help="path to the negative files"	, action = "store")
	
	parser.add_argument("--experiment_name" , help="MAY NOT CONTAIN '/'!give your experiment a name to attach to all outputs such as similarityfiles"	, action = "store")
	parser.add_argument("--unsupervised" , help="'1': do not load a simmat in order to derive kNN accuracy; will not plot even if specified"	, action = "store")
	parser.add_argument("--label_dir_from_file" , help="path to a histogram file - created e.g. by ../db_interaction/mongo.py -- <calculate_distance_distribution> function"	, action = "store")
	parser.add_argument("--tfidf_triple_name" , help="name of the mongoDB collections corresponding to the data set to consider"	, action = "store")
	parser.add_argument("--rest"      , help="'1': take all other files for the same category", action = "store")
	parser.add_argument("--n_neighbors_list" , help="semi-colon separated integers to use for evaluation the classification on, e.g. '1;3;5;7;9'"	, action = "store")
	parser.add_argument("--euclideanWMD" , help="1: Use WMD equipped with euclidean distance instead of cosine distance"	, action = "store")
	parser.add_argument("--min_df" , help="min_df for tfidf"	, action = "store")
	parser.add_argument("--max_df" , help="max_df for tfidf"	, action = "store")
	parser.add_argument("--vector_normalization"      , help="'1': normalize the word vectors before passing to the WMD", action = "store")
	parser.add_argument("--precalculated_dir"      , help="path to a directory holding precalculated pickle files - no pickle files generation overhead", action = "store")


	args = parser.parse_args()
		

	
	knowledge_dir   = args.knowledge_dir
	label_dir       = args.label_dir
	we_model_path	= args.model
	processes		= int(args.processes)
	# DEFAULT PARAMETERS: 	sig_size = 50, min_df = 0.0, max_df = 1.0, distance_function = euclidean_distance, \
	sigSize			= int(args.sig_size)
	sigVecStrat		= args.sig_vec_strat
	sigWeightStrat	= args.sig_weight_strat
	classifier_log	= args.classifier_log
	positive_dir 	= args.negative_dir
	negative_dir 	= args.positive_dir
	
	experiment_name = args.experiment_name
	unsupervised	= args.unsupervised
	label_dir_from_file = args.label_dir_from_file
	tfidf_triple_name= args.tfidf_triple_name
	rest			= args.rest
	n_neighbors_list= args.n_neighbors_list
	euclidean		= args.euclideanWMD
	minDf			= args.min_df
	maxDf			= args.max_df
	
	vectorNormalization = args.vector_normalization
	if vectorNormalization == "1":
		vectorNormalization = True
	else:
		vectorNormalization = False	
		

	precalculated_dir = args.precalculated_dir
	# if precalculated_dir not None
	if precalculated_dir:
		skip_pickling = True
	else:
		skip_spickling = False
	
	if unsupervised == "1":
		unsupervised = True		
	
	# check for rest
	if rest == "1":
		rest = True
		label_path		= "rest"
	else:
		rest = False
		label_path = label_dir
	
	# extract n_neighbors_list
	if n_neighbors_list == None:
		n_neighbors_list = [1,3,5,7,9,11,13,15,17,19,25,35,45,55]
	else:
		n_neighbors_list = [int(el) for el in n_neighbors_list.split(";")]
	n_neighbors = 1 # some default?
	
	# choose cosine distance as default distance metric for WMD
	if euclidean == "1":
		distance_function_string = "euclidean"
	else:
		distance_function_string = "cosine"
		
	# set default values for minDf and maxDf
	if minDf == None:
		minDf = 0.0
	else:	
		minDf = float(minDf)
	if maxDf == None:
		maxDf = 1.0
	else:	
		maxDf = float(maxDf)

		
	# check if experiment_name contains '/'
	if "/" in experiment_name:
		print("Experiment Name may not contain '/', '/' has been removed")
		experiment_name = experiment_name.replace("/","")
		
	# set pickle_dir
	pickle_dir		= "../../data/picklefiles/"
	similarity_dir	= "../../data/similarityfiles"
		
	####################################################################################################################
	# ACQUIRE KNOWLEDGE FOR CLASSIFICATION
	#######################################
	# load knowledge_file_paths
	knowledge_file_paths = get_file_paths(knowledge_dir, ext = ".txt")
	# get 'knowledge_file_signatures'' true labels [1: positive, 0: negative] (includes unknown labels -1)
	# calculate labels in the input_path according to 'positive_dir' and 'negative_dir'
	# the label file name is by default 'label_file' and written into the input directory [in this case 'knowledge_dir']
	if not unsupervised:
		knowledge_labels = get_labels(knowledge_dir, positive_dir, negative_dir, as_list = True)

	# PREPARE FILES TO LABEL
	# if not unsupervised: assign labels
	# if unsupervised: calculate pairwise distances between knowledge_files and label_files
	########################
	# get label file paths from label_dir, if rest is False
	if not rest:
		# here you might need to create the label_dir first, e.g. by extracting the file_paths from a histogram file
		if label_dir_from_file:
			# extract file paths to compare the knowledge files to from the 'label_dir_from_file' path by looking up the .txt files in label_dir
			to_label_file_paths = read_histogram_file(label_dir_from_file, label_dir)
		else:
			to_label_file_paths = get_file_paths(label_dir, ext = ".txt")
		
	else:
		# ../../data/experiments/IMDB_supervised_small/IMDB_supervised_small_1 --> IMDB_supervised_small_1
		knowledge_folder = os.path.split(knowledge_dir)[-1]
		# knowledge_super_directory
		knowledge_super_dir = os.path.split(knowledge_dir)[:-1][0]
		
		to_label_file_paths = []
		for root, dir, files in os.walk(knowledge_super_dir):
			for file_name in files:
				# is *.txt file
				isTxtFile = os.path.splitext(file_name)[1] == '.txt'
				file_path = os.path.join(root,file_name)
				if (isTxtFile) & (file_path not in knowledge_file_paths):
					to_label_file_paths.append(file_path)
					
	# PREDICT LABELS
	################
	if not unsupervised:
		predict_labels(to_label_file_paths, knowledge_file_paths, knowledge_labels, pickle_dir, classifier_log, tfidf_triple_name, n_neighbors, we_model_path, sigSize, minDf, maxDf, distance_function_string,\
						sigVecStrat, sigWeightStrat, vectorNormalization, weightNormalization, verbose = False, processes = processes, skip_pickling = skip_pickling, precalculated_dir = precalculated_dir)	
		# evaluate the accuracy
		for n_neighbors in n_neighbors_list:
			accuracy = evaluate_classifier_log(classifier_log, n_neighbors, positive_dir, negative_dir)
			knowledge_path 	= knowledge_dir
			output_path 	= "result.log"
			log_classifier_accuracy(output_path, knowledge_path, label_path, we_model_path,  sigVecStrat, sigWeightStrat, minDf, maxDf, sigSize, distance_function_string, vectorNormalization, n_neighbors, accuracy)
	else:
		# create pickle files for both knowledge and label files
		knowledge_pickle_paths = make_pickle_files(knowledge_file_paths, pickle_dir, we_model_path , sig_size = sigSize,
													sigvec_strat = sigVecStrat , sigweight_strat = sigWeightStrat, min_df = minDf ,
													max_df = maxDf, vector_normalization = vectorNormalization,
													verbose = False)
		to_label_pickle_paths = make_pickle_files(to_label_file_paths, pickle_dir, we_model_path, sig_size = sigSize,\
													sigvec_strat = sigVecStrat , sigweight_strat = sigWeightStrat, min_df = minDf ,\
													max_df = maxDf, vector_normalization = vectorNormalization,
													verbose = False)
		# create pickle_dirs
		knowledge_label_dir = os.path.join(pickle_dir,get_current_timestamp()+"_knowledge")
		create_label_dir(knowledge_pickle_paths, knowledge_label_dir)
		# create to_label_labelpickle_dirs
		to_label_label_dir = os.path.join(pickle_dir,get_current_timestamp()+"_to_label")
		create_label_dir(to_label_pickle_paths, to_label_label_dir)
							
		# calculate pairwise distances between knowledge and label files (via pickle files)
		calcPairwiseDist(knowledge_label_dir, similarity_dir, pickle_dir2 = to_label_label_dir, distance_function_string = distance_function_string, from_dir = True, experiment_name = experiment_name, processes = 4)
	
	# get label file 'true labels'
	#print(label_file_paths, knowledge_file_paths, knowledge_signatures, labels)
	

	# RUN CLASSIFICATION ON GRID
	#####################################################	
	#predicted_labels_lists, true_labels_list = run_grid(to_label_file_paths, we_model_path, knowledge_file_paths, knowledge_dir, label_dir, positive_dir, negative_dir, log_path, sequential = True, max_dfs = [0.95, 1.0])#,\
	#	sig_sizes = [50], min_dfs = [0.0], max_dfs = [1.0], distance_functions = [cosine_distance], sigvec_strats = "tfidf", sigweight_starts = "tfidf", weight_normalizations=["histogram"], processes = 4)	
	
	#run_grid(to_label_file_paths, knowledge_signatures, knowledge_labels, knowledge_dir, label_dir, we_model_path, positive_dir, negative_dir, log_path, n_neighbors_list, we_model)#, \
	#sigSizes, minDfs, maxDfs, distance_functions, sigVecStrats, sigWeightStrats, weightNormalizations, verbose = False, processes = 4)


if __name__ == "__main__":
	main()



















	
