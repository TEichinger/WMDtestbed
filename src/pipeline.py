#! /usr/lib/python3

# this is the pipeline file
# INPUT : experiment_folder with .txt files
# PIPELINE
#################
# For all .txt files in the experiment_folder
# check if a pickle file has been calculated previously
# if not
# calculate pairwise distances between the texts
# output a similarity (dissimilarity matrix)
# define a classifier by adding labels [this should probably be done earlier]
#########################

"""
BLOCK 0: Pre-Setup - Setup of paths needed for the Pipeline and necessary imports
"""
# import statements
import argparse
import os
import sys
import subprocess
from sklearn.feature_extraction.text import TfidfVectorizer

#from pathlib import PurePath
file_dir   = os.path.dirname(os.path.realpath(__file__))
src_dir	   = file_dir # ./src
root_dir   = os.path.split(file_dir)[0] # ./
sys.path.insert(0, src_dir)

# needed for the evaluation of the computed edgegraph, uses the 'knn'-directory
from evaluation.evaluate      import evaluate_via_simmat
# build pickle files for each document individually, pickle files are then used for similarity comparison
from preprocessing.preprocess import make_pickle_files
# helper functions e.g. for calculation of tfidf values
from util.utilities           import load_labels, check_supervised, log_LOO_accuracy, file_path_to_file_name, remove_twitter_tags
# function needed for the calculation of the distance between each document
from wmd.wmd		          import calcPairwiseDist
# needed for multidimensional scaling which is needed for a graphical output of the results
from MDS.MDS import load_edgegraph, run_MDS, make_colors

def main():
	"""
	BLOCK 1: General Setup - Setup of path-variables and parsing of the passed arguments
	"""

	# SET GLOBAL DIRECTORIES
	#########################
	pickle_dir     	= os.path.join(root_dir, 'data/picklefiles')
	similarity_dir 	= os.path.join(root_dir, 'data/similarityfiles')
	we_model_path  	= os.path.join(root_dir, 'data/we_models/reference_model.bin')
	image_dir  		= os.path.join(root_dir, 'data/plots/')
	positive_dir 	= os.path.join(root_dir, 'data/corpora/IMDB/train/pos')
	negative_dir 	= os.path.join(root_dir, 'data/corpora/IMDB/train/neg')

	# PARSE ARGS
	############
	parser = argparse.ArgumentParser()
	parser.add_argument("data"        				, help="path to the documents to be labeled" 															, action = "store")
	parser.add_argument("model"       				, help="path to the binary (.bin) word embedding model to use" 											, action = "store")
	parser.add_argument("processes"   				, help="number of parallel pool processes"																, action = "store")
	parser.add_argument("sig_size"    				, help="signature size for the WMD"	                													, action = "store")
	parser.add_argument("sig_vec_strat"    			, help="string specifying the signature vector strategy; first k by default"							, action = "store")
	parser.add_argument("sig_weight_strat" 			, help="string specifying the signature weight strategy; 1/k by default, where n the signature size"	, action = "store")
	parser.add_argument("log_path" 					, help="path to the log files"																			, action = "store")
	parser.add_argument("--unsupervised"			, help="'1': do not load a simmat in order to derive kNN accuracy; will not plot even if specified"		, action = "store")
	parser.add_argument("--label_file"				, help="if specified, use a custom-made label_file"														, action = "store")
	parser.add_argument("--n_neighbors_list" 		, help="semi-colon separated integers to use for evaluation the classification on, e.g. '1;3;5;7;9'"	, action = "store")
	parser.add_argument("--euclideanWMD" 			, help="1: Use WMD equipped with euclidean distance; default: cosine distance"							, action = "store")
	parser.add_argument("--min_df" 					, help="min_df only for tfidf, 0 by default"																	, action = "store")
	parser.add_argument("--max_df" 					, help="max_df only for tfidf, 1 by default"																	, action = "store")
	parser.add_argument("--sim"      				, help="use precalculated similarity file [edge_graph format]"											, action = "store")
	parser.add_argument("--vector_normalization"	, help="'1': normalize the word vectors before passing to the WMD, False by default"										, action = "store")
	parser.add_argument("--verbose"      			, help="'1': print signature word tokens and weights for all reviews"									, action = "store")
	parser.add_argument("--plot"					, help="'1': create an MDS plot of the input"															, action = "store")
	parser.add_argument("--clean_twitter_data"		, help="'1': removes hashtags, mentions and picture-links from the input-data. Recommended when using twitter-data as input", action = "store")
	parser.add_argument("--allow_3_grams"			, help="'1': allows the selected keyword extraction method to extract 2-grams and 3-grams in addition to just 1-grams", action = "store")
	parser.add_argument("--output_format"			, help="the output type of the similarity matrix; one of ['gephi', 'sim_dict', 'None']; default: 'gephi'", action = "store")
	parser.add_argument("--make_pickle"				, help="'1': pickle similarity matrix", action = "store")

	# PROVIDING OF VARIOUS VARIABLE NAMES TO ARGS WHICH IS EASIER FOR FURTHER USAGE
	###################################################
	args = parser.parse_args()
	input_path      = args.data
	input_folder	= os.path.split(input_path)[-1]
	we_model_path   = args.model
	processes = int(args.processes)
	sig_size = int(args.sig_size)
	sigvec_strat	= args.sig_vec_strat
	sigweight_strat	= args.sig_weight_strat
	log_path 		= args.log_path
	unsupervised	= args.unsupervised
	label_file_path	= args.label_file
	n_neighbors_list = args.n_neighbors_list
	euclideanWMD = args.euclideanWMD
	min_df = args.min_df
	max_df = args.max_df
	edge_graph_path = args.sim
	vector_normalization = args.vector_normalization
	verbose = args.verbose
	plot = args.plot
	clean_twitter_data = args.clean_twitter_data
	allow_3_grams = args.allow_3_grams
	output_format = args.output_format
	make_pickle = args.make_pickle

	# Setup of experiment name from the input path
	experiment_name = os.path.split(input_path)[-1]

	# setup whether the experiment should run unsupervised
	if unsupervised == "1":
		unsupervised = True

	# default n_neighbors_list for the knn-classifier
	if n_neighbors_list == None:
		n_neighbors_list = [1,3,5,7,9,11,13]
	else:
		n_neighbors_list = [int(el) for el in n_neighbors_list.split(";")]

	# specify distance_function_string (either "cosine" or "euclidean") that is used in WMD
	if euclideanWMD == "1":
		distance_function_string = "euclidean"
	else:
		distance_function_string = "cosine"

	# if min_df is specified, set it to this value, if not set it to 0
	if min_df != None:
		min_df = float(args.min_df)
	else:
		min_df = 0

	# if max_df is specified, set it to this value, if not set it to 1
	if max_df != None:
		max_df = float(args.max_df)
	else:
		max_df = 1

	# Boolean that defines whether the word vector should be normalized for WMD
	if vector_normalization == "1":
		vector_normalization = True
	else:
		vector_normalization = False

	# setup whether additional information about the ingoing experiment should be printed
	if verbose == "1":
		verbose = True
	else:
		verbose = False

	# boolean that defines whether a plot of the experiment should be rendered
	if plot == "1":
		plot = True
	else:
		plot = False

	# boolean that defines whether the input-files should be cleaned of hashtags etc.
	if clean_twitter_data == "1":
		clean_twitter_data = True
	else:
		clean_twitter_data = False

	# boolean that defines whether the selected keyword extraction method
	# is allowed to extract 2-grams and 3-grams in addition to just 1-grams
	if allow_3_grams == "1":
		allow_3_grams = True
	else:
		allow_3_grams = False

	# string that defines the output format of the similarity matrix
	if output_format is None:
		output_format = "gephi"

	# boolean that defines whether the similarity matrix is to be pickled or saved without pickling
	if make_pickle == "1":
		make_pickle = True
	else:
		make_pickle = False


	"""
	BLOCK 2: Pickle Setup - Sets up the path for the pickle files, in current version a new pickle folder is always created
	"""

	# CREATE PICKLE SUBDIRECTORY
	###################################################
	pickle_folder = os.path.join(pickle_dir, input_folder)
	# if directory doesnt exist already
	if not os.path.isdir(pickle_folder):
		# create the new directory
		os.mkdir(pickle_folder)

	# RUN ROUTINE FOR PICKLING REVIEWS
	###################################################
	file_names = []
	# collect all file paths of *.txt files to pickle from input_path
	to_pickle_file_paths = []
	# for all files in 'input_path'
	for root, dirs, files in os.walk(input_path):
		for name in files:
			# that are *.txt files
			isTxtFile = os.path.splitext(name)[1] == '.txt'
			if isTxtFile:
				# add them to the pickle path list
				to_pickle_file_paths.append(os.path.join(root, name))
				file_names.append(name)

	"""
	BLOCK 3: Pickle File Creation - calls the function 'make_pickle_files' inside the preprocess-package,
									which creates a pickle file for every input file.
									The 'make_pickle_files' expects different parameters based on the selected keyword extraction method
	"""

	# SPECIAL ROUTINE IF TFIDF IS CHOSEN. COLLECTS EVERY INPUT FILE AND CALCULATES THE IDF FOR EVERY TERM IN THE INPUT
	###################################################
	if sigvec_strat == "tfidf" and sigweight_strat == "tfidf":
		# define list where input files will be stored as strings
		docs = []
		# then walk through the input path and collect all files
		for root, dirs, files in os.walk(input_path):
			for name in files:
				# that are *.txt files
				isTxtFile = os.path.splitext(name)[1] == '.txt'
				if isTxtFile:
					# open the file and save it into the lines variable
					f = open(os.path.join(input_path, name))
					lines = f.read()
					# when the cleaning of twitter data is specified, append the cleaned version
					if clean_twitter_data:
						docs.append(remove_twitter_tags(lines))
					else:
						# otherwise append the uncleaned version
						docs.append(lines)

		# If 3_grams are allowed, the TfidfVectorizer is instanced with the optional ngram_range parameter set to (1,3)
		if allow_3_grams:
			# creation of a TfidfVectorizer instance as specified in the sklearn documentation
			vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, strip_accents='unicode', sublinear_tf=True, ngram_range=(1,3))
			vectorizer.fit(docs)

			# starts the pickling process of the input files. Note that the optional TfidfVectorizer instance is set as this is needed for the tfidf calculation later on
			pickle_paths = make_pickle_files(to_pickle_file_paths, pickle_dir, we_model_path, sig_size=sig_size,
											sigvec_strat=sigvec_strat, sigweight_strat=sigweight_strat,
											vector_normalization=vector_normalization, tfidf_vectorizer=vectorizer, clean_twitter_data=clean_twitter_data,
											allow_3_grams=allow_3_grams, verbose=verbose)

		# If 3_grams are NOT allowed, the TfidfVectorizer is instanced without the optional ngram_range parameter
		else:
			# creation of a TfidfVectorizer instance as specified in the sklearn documentation
			vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, strip_accents='unicode', sublinear_tf=True)
			vectorizer.fit(docs)

			# starts the pickling process of the input files. Note that the optional TfidfVectorizer instance is set as this is needed for the tfidf calculation later on
			pickle_paths = make_pickle_files(to_pickle_file_paths, pickle_dir, we_model_path, sig_size=sig_size,
											sigvec_strat=sigvec_strat, sigweight_strat=sigweight_strat,
											vector_normalization=vector_normalization, tfidf_vectorizer=vectorizer,
											clean_twitter_data=clean_twitter_data,
											allow_3_grams=allow_3_grams, verbose=verbose)

	# STANDARD ROUTINE IF YAKE OR RAKE ARE CHOSEN AS KWE METHOD
	###################################################
	else:
		# starts the pickling process of the input files. Note that the optional TfidfVectorizer instance is NOT set.
		pickle_paths = make_pickle_files(to_pickle_file_paths, pickle_dir, we_model_path, sig_size=sig_size,
										 sigvec_strat=sigvec_strat, sigweight_strat=sigweight_strat,
										 vector_normalization=vector_normalization, clean_twitter_data=clean_twitter_data, allow_3_grams=allow_3_grams ,verbose=verbose)

	"""
	BLOCK 4: Similarity Matrix - creates a similarity file as a similarity matrix or load specified edgegraph
	"""

	# if edge_graph_path has been given, do not recalculate pairwise distances
	if edge_graph_path:
		print("A precalculated edge graph containing pairwise similarities has been found at {}".format(edge_graph_path))
		print("Pairwise similarities are not recalculated.")

	# else calculate pairwise distances
	else:
		n = len(pickle_paths)
		print("No precalculated edge graph has been specified.")
		print('Calculate {} pairwise distances.'.format(((n-1)*(n))/2))
		edge_graph_path = calcPairwiseDist(pickle_paths, similarity_dir, distance_function_string = distance_function_string, from_dir = False, experiment_name = experiment_name+'_WMD'+str(sig_size), processes = processes,
										output_format = output_format, make_pickle = make_pickle)
		print('Save edge graph at {}.'.format(edge_graph_path))

	if not unsupervised:
		# load edge graph
		print("Load edge graph as similarity matrix")
		simmat = load_edgegraph(edge_graph_path, as_simmat=True)

	"""
	BLOCK 5: Calculate Accuracy - initially calculates the label for every input file and then computes the accuracy of the classifier
	"""

	# calculate labels in the input_path according to 'positive_dir' and 'negative_dir'
	# the label file name is by default 'label_file' and written into the 'input_path'
	if not label_file_path:
		subprocess.call(["python3", "preprocessing/setup_experiment.py", input_path, positive_dir, negative_dir])
		# load labels from that 'label' file, as pandas dataframe
		label_df = load_labels(input_path + '/label_file', as_list=False)
	else:
		# load labels from that 'label' file, as pandas dataframe
		label_df = load_labels(label_file_path, as_list=False)

	# CALCUALTE THE ACCURACY / EVALUATE
	###################################
	# 'picklefiles/pickle_file.pk' --> 'pickle_file' (as in the index of the label_df (pandas dataframe)
	pickle_names = [file_path_to_file_name(pickle_path) for pickle_path in pickle_paths]
	isSupervised = check_supervised(label_df)

	if isSupervised:
		print("The experiment is supervised. We can thus calculate a leave-one-out accuracy directly.")
		# NOTE that the label_df may contain items that are not in the pickle_files (and thus not considered for the experiment)
		# extract label_df's labels of items that are relevant to the experiment
		# knn_labels :
		knn_labels = label_df.loc[pickle_names,'label'].tolist()

		for n_neighbors in n_neighbors_list:
			accuracy   = evaluate_via_simmat(simmat, knn_labels, n_neighbors = n_neighbors)

			signature_vector_strategy = "top-"+str(sig_size)+"-"+ sigvec_strat
			signature_weight_strategy = sigweight_strat
			#                output_path, input_info, we_model_path, sigvec_strat, sigweight_strat,                     min_df, max_df, sig_size, dist_function, vector_normalization, kNN, accuracy
			log_LOO_accuracy(log_path, input_path, we_model_path, signature_vector_strategy, signature_weight_strategy, min_df, max_df, sig_size, distance_function_string, vector_normalization, n_neighbors, accuracy)
	else:
		print("The experiment is not supervised. We can thus not directly calculate an accuracy.")

	# PLOT, if specified
	####################
	if plot:
		# run ainDS on the pairwise distances (embedding into two dimensions) and draw the plot, the plot is saved in 'image_path'
		#similarity_matrix = load_edgegraph(edge_graph_path)
		image_path = os.path.basename(edge_graph_path)
		image_path = image_dir + os.path.splitext(image_path)[0] + '.png'


		# for every pickle_name in pickle_names that is specified in label_df (index of the pandas df); create a legend of label strings (e.g. ['positive','negative']
		# and a list of color strings to represent each label
		legend, colors = make_colors(label_df, pickle_names, color_palette = ["black", "red", "blue", "green", "yellow", "orange", "magenta", \
									"lightblue","grey", "lightred"])

		# run MDS and write a plot file
		run_MDS(simmat, from_file = False, file_name = image_path, colors = colors, legend = legend)#, names = file_names)

	print('PIPELINE FINISHED!')

if __name__ == "__main__":
	main()
