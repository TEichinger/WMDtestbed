import sys, os
sys.path.append('./')
sys.path.append('../')
sys.path.append('../wmd/python-emd-master')
from util.utilities import cosine_distance, euclidean_distance, load_we_model, get_file_paths, get_labels
import argparse
from classifier import run_grid, load_knowledge

############################################################################
# NOT FUNCTIONAL AS OF NOW
############################################################################

# this python file implements cross validation for the classifiers defined in classifier.py


"""
def cross_validate(experiment_folders (or experiment_file_paths as list of list?), parameters for classifier):
	for every sublist:
		create a classifier
		run classifier on all other experiment folders
		collect accuracy
		derive overall accuracy
		
	return individual_accuracies, overall_accuracy, best_model # where best model is a set of reviews and a parameter set (just regarding the signatures is probably not legitimate)
"""

def main():
# PARSE ARGS
	#############
	parser = argparse.ArgumentParser()
	parser.add_argument("experiment_dir", help="path to directory holding the split folders for cross validation") 	
	parser.add_argument("model"       , help="path to the binary (.bin) word embedding model to use" 											, action = "store")
	parser.add_argument("processes"   , help="number of parallel pool processes"																, action = "store")
	parser.add_argument("sig_size"    , help="signature size for the WMD"	                													, action = "store")
	parser.add_argument("sig_vec_strat"    , help="string specifying the signature vector strategy; first k by default"							, action = "store")
	parser.add_argument("sig_weight_strat" , help="string specifying the signature weight strategy; 1/k by default, where n the signature size"	, action = "store")
	parser.add_argument("log_path" , help="path to the log files"	, action = "store")
	parser.add_argument("positive_dir" , help="path to the positive files"	, action = "store")
	parser.add_argument("negative_dir" , help="path to the negative files"	, action = "store")
	
	parser.add_argument("--n_neighbors_list" , help="semi-colon separated integers to use for evaluation the classification on, e.g. '1;3;5;7;9'"	, action = "store")
	parser.add_argument("--euclideanWMD" , help="1: Use WMD equipped with euclidean distance instead of cosine distance"	, action = "store")
	parser.add_argument("--min_df" , help="min_df for tfidf"	, action = "store")
	parser.add_argument("--max_df" , help="max_df for tfidf"	, action = "store")


	args = parser.parse_args()
		
	positive_dir = args.negative_dir
	negative_dir = args.positive_dir
	
	experiment_dir   = args.experiment_dir
	we_model_path	= args.model
	processes		= args.processes
	# DEFAULT PARAMETERS: 	sig_size = 50, min_df = 0.0, max_df = 1.0, distance_function = cosine_distance, \
	#						sigvec_strat = "tfidf", sigweight_strat = "tfidf", n_neighbors = 1, weight_normalization = "histogram"
	sigSize			= int(args.sig_size)
	minDf			= args.min_df
	maxDf			= args.max_df
	n_neighbors_list = args.n_neighbors_list
	euclidean		= args.euclideanWMD
	sigVecStrat		= args.sig_vec_strat
	sigWeightStrat	= args.sig_weight_strat

	log_path		= args.log_path
	
	# extract n_neighbors_list
	if n_neighbors_list == None:
		n_neighbors_list = [1,3,5,7,9]
	else:
		n_neighbors_list = [int(el) for el in n_neighbors_list.split(";")]
	
	
	# choose cosine distance as default distance metric for WMD
	if euclidean == "1":
		distance_function = euclidean_distance
	else:
		distance_function = cosine_distance
		
	# set default values for minDf and maxDf
	if minDf == None:
		minDf = 0.0
	else:	
		minDf = float(minDf)
	if maxDf == None:
		maxDf = 1.0
	else:	
		maxDf = float(maxDf)

	# choose 'histogram' weight normalization as default
	if weightNormalization == None:
		weightNormalization = "histogram"
	
	# LOAD FASTTEXT MODEL
	#####################
	we_model = load_we_model(we_model_path)
	
	
	# we are going to use the run_grid method from classifier.py with
	# for every folder in the experiment_dir
	# 	define knowledge_dir as the folder
	#	define label_dir as the union of the rest of the folders in experiment_dir
	#	run grid on this setup (with the parameter set given)

	# find all experiment folders, every experiment folder is going to act as a knowledge folder once!
	# FIND EXPERIMENT FOLDERS
	#########################
	experiment_folder_paths = []
	for root, dirs, files in os.walk(experiment_dir):
		for dir in dirs:
			experiment_folder_paths.append(os.path.join(root,dir))
			
	print(len(experiment_folder_paths))		
	# CROSS VALIDATE
	##########################

	# for every experiment folder
	for experiment_folder_path in experiment_folder_paths:
		# SPLIT knowledge_dir AND label_dir
		#####################################
		# allocate to_label_file_paths knowledge_dir
		knowledge_dir 			= experiment_folder_path
		knowledge_file_paths 	= get_file_paths(knowledge_dir, ext = ".txt")
		
		knowledge_signatures 	= load_knowledge(knowledge_file_paths, we_model, sigSize, minDf, maxDf, sigVecStrat, sigWeightStrat, weightNormalization)
		knowledge_labels 		= get_labels(knowledge_dir, positive_dir, negative_dir, as_list = True)
		
		
		# allocate to_label_file_paths
		to_label_file_paths = []
		rest_of_experiment_folder_paths = [el for el in experiment_folder_paths if el != experiment_folder_path]
		for rest_folder in rest_of_experiment_folder_paths:
			to_label_file_paths += get_file_paths(rest_folder, ext = ".txt")

		# run_grid for every setup
		print("Run evaluation on ", experiment_folder_path)
		
		# set label_dir
		label_dir = "all-but-" + knowledge_dir
		
		run_grid(to_label_file_paths, knowledge_signatures, knowledge_labels, knowledge_dir, label_dir, we_model_path, positive_dir, negative_dir, log_path, n_neighbors_list, we_model, \
				sigSizes = [50], minDfs = [0.0] , maxDfs=[1.0], distance_functions = [cosine_distance], sigVecStrats =["tfidf"], sigWeightStrats = ["tfidf"],\
				weightNormalizations = ["histogram"], verbose = False, processes = 4)
	
	
	return


if __name__ == "__main__":
	main()