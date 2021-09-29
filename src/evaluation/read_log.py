import pandas as pd
import re
import argparse

############################################################################
# NOT FUNCTIONAL AS OF NOW
############################################################################

# Extract information from ../../data/logs/results.log (result logs) in an aggregated fashion
# The logs have been standardized to fit the following format:
# format:
# Timestamp          ## log type   inputPath [either one path or two];             WEmodel;                               signature vector strategy;                   signature weight strategy; minDf; maxDf; signature size; distance function (WMD); weight normalization; kNN; acc;
# 2019_01_17_19_01_43##LOO:inputPath=../data/experiments/IMDB_supervised_small_241;WEmodel=../data/we_models/test_we.bin;sigVecStrat=top-50-tfidfmin_df=0.0max_df=1.0;sigWeightStrat=tfidf;minDf=0.0;maxDf=1.0;sigSize=50;distFunctionWMD=cosine;weightNormalization=histogram;kNN=5;acc=0.64;
# The following functions should provide enough functionality to read out any log of the above format
# per experiment folder: mean accuracy
# the results are formed into a pandas DataFrame
#
#

	
# input a pandas df of experiment records (without zeros?)
# output is a pandas series!
def average_experiments(df):
	df_avg = df.groupby(["experiment"])["accuracy"].mean()
	return df_avg
	


# assess the inference_accuracy by reading classifier logs for all possible classifiers!
def assess_inference_accuracy(log_path, basecaption, no_of_experiments, range_of_k, parameter_string):
	# create a template as in the above
	df_overall = pd.DataFrame(0, index = range(number_of_experiments), columns = ["experiment", "params", "accuracy"])
	
	# for every experiment number (corresponding to an experiment folder, and thereby to a classifier)
	for i in range(no_of_experiments):
		# index shift
		j = i+1
		# read_classifier_log and get the performance dataframes
		df = read_classifier_log(log_path,  basecaption, j, no_of_experiments, range_of_k , parameter_string)
		df = drop_empty_experiments(df)
		# get the average accuracy and log them for every experiment folder and parameter setting
		mean_accuracy = df["accuracy"].mean()

		# fill row with index == j
		df_overall.loc[j,:] = [basecaption+'_'+str(j), parameter_string, mean_accuracy]
		
	return df_overall
	
# input "featureName=foo;", "featureName"
# output "foo"
def extract_feature(line, feature_name):
	pattern = re.compile(feature_name + "=.*?;") # assume that the feature_name does not contain "="
	pattern_string = pattern.findall(line)[0]
	# featureName=foo;" --> "foo;"
	pattern_string = pattern_string.split("=")[1]
	# "foo;" --> "foo"
	result_string = pattern_string.split(";")[0] # assume that the feature_name foes not contain "="
	return result_string
	

# extract all parameters in the line, they are separated by ';'
# log_type, inputPath, WEmodel, sigVecStrat, sigWeightStrat,\
# minDf, maxDf, sigSize, distFunctionWMD, weightNormalization, kNN, acc 
# 2019_01_17_22_53_21##LOO:inputPath=../data/experiments/IMDB_supervised_small_247;WEmodel=../data/we_models/test_we.bin;sigVecStrat=top-50-tfidfmin_df=0.0max_df=1.0;sigWeightStrat=tfidf;minDf=0.0;maxDf=1.0;sigSize50;distFunctionWMD=cosine;weightNormalization=euclidean;kNN=7;acc=0.59;
def extract_line_info(line):
	# extract log_type
	log_type_p = re.compile("##.*?:")
	log_type_string = log_type_p.findall(line)[0]
	log_type = log_type_string[2:-1]
	# extract inputPath
	inputPath 				= extract_feature(line, "inputPath")
	# extract WEmodel
	WEmodel					= extract_feature(line, "WEmodel")
	# extract sigVecStrat
	sigVecStrat				= extract_feature(line, "sigVecStrat")
	# extract sigWeightStrat
	sigWeightStrat			= extract_feature(line, "sigWeightStrat")
	# extract minDf
	minDf					= extract_feature(line, "minDf")
	minDf					= float(minDf)
	# extract maxDf
	maxDf					= extract_feature(line, "maxDf")
	maxDf					= float(maxDf)
	# extract sigSize
	try:
		sigSize				= extract_feature(line, "sigSize")
	except:
		sigSize = 0
	sigSize					= int(sigSize)
	# extract distFunctionWMD
	distFunctionWMD			= extract_feature(line, "distFunctionWMD")
	# extract weightNormalization
	weightNormalization		= extract_feature(line, "weightNormalization")
	# extract kNN
	kNN						= extract_feature(line, "kNN")
	kNN						= int(kNN)
	# extract acc
	acc						= extract_feature(line, "acc")
	acc						= float(acc)
	

	return log_type, inputPath, WEmodel, sigVecStrat, sigWeightStrat, minDf, maxDf, sigSize, distFunctionWMD, weightNormalization, kNN, acc

	
# enter a string 'line_element' and a substring 'element'
# output True if it contains the substring
def check_match(line_element, element, exact = False):
	if element != None:
		# if exact == True, check if the line_element and element match exactly
		if exact:
			return str(line_element) == str(element)
		element = str(element)
		line_element = str(line_element)
		return line_element.find(element) != -1
	else:
		# default True (no query specification)
		return True
	
#
# inputPath;WEmodel;sigVecStrat;sigWeightStrat;minDf;maxDf;sigSize;distFunctionWMD;weightNormalization;kNN;acc;
# [log_path] : "../../data/logs/results.log"
# [log_type] : e.g. "LOO"
# [inputPath] : CHANGE "IMDB_supervised_small"
# [WEmodel] : path to the word embedding model
# [sigVecStrat] : signature vector strategy (e.g. "tfidf")
# [sigWeightStrat] : signature weight strategy (e.g. "tfidf")
# [minDf] : <float> e.g. 0.0
# [maxDf] : <float> e.g. 1.0
# [sigSize] : <int>e.g. 50
# [distFunctionWMD] : e.g. "cosine"
# [weightNormalization] : e.g. "cosine"
# [kNN] : <int> e.g. 3
# apart from the log_path, all arguments are kwargs, that is you can decide which properties you want to query.
# most cases you will want to query for a
#
def read_log(log_path, inputPath,  log_type = None, WEmodel = None, sigVecStrat = None, sigWeightStrat = None, minDf = None, \
						maxDf = None, sigSize = None, distFunctionWMD = None, weightNormalization = None, kNN = None, acc = None, verbose = True):

	if verbose:
		print("Read logs in {} for experiment paths matching {}.".format(log_path, inputPath))
	
	df = pd.DataFrame(columns = ["type", "experimentInput",  "we_model", "sigVecStrat", "sigWeightStrat", "minDf", "maxDf",	"sigSize", "distFunctionWMD", "weightNormalization", "k", "accuracy"])
	
	# open 'log_path'
	with open(log_path, mode = 'r') as f:
		# for line in log
		for line in f:
			# extract line information, these elements are of <str> type!
			log_type_line, experimentInputs_line, WEmodel_line, sigVecStrat_line, sigWeightStrat_line,\
			minDf_line, maxDf_line, sigSize_line, distFunctionWMD_line, weightNormalization_line, kNN_line, acc_line                 = extract_line_info(line)
			
			# make a distinction between LOO logs and classifier logs
			##########################################################
			# LOO only have one input directory, whereas classifier logs have 2 (knowledgeDir and labelDir)
			if log_type_line == "LOO":
				experimentInput_line = experimentInputs_line
			elif log_type_line == "classifier":
				experimentInput_line = "classfier"
				
				# ADD a split into knowledgeDir and labelDir
				# experimentInput = knowledgeDir
			else:
				print("ERROR")
				return
				
			# QUERY MATCHES
			################
			# if inputPath matches experimentInput
			matches_inputPath = check_match(experimentInput_line, inputPath, exact = True)
			# if log_type matches
			matches_log_type = check_match(log_type_line, log_type)
			# if WEmodel matches
			matches_WEmodel = check_match(WEmodel_line, WEmodel)
			# if sigVecStrat matches
			matches_sigVecStrat = check_match(sigVecStrat_line, sigVecStrat)
			# if sigWeightStrat matches
			matches_sigWeightStrat = check_match(sigWeightStrat_line, sigWeightStrat)
			# if minDf matches
			matches_minDf = check_match(minDf_line, minDf, exact = True)
			# if maxDf matches
			matches_maxDf = check_match(maxDf_line, maxDf)
			# if sigSize matches
			matches_sigSize = check_match(sigSize_line, sigSize)
			# if distFunctionWMD matches
			matches_distFunctionWMD = check_match(distFunctionWMD_line, distFunctionWMD)
			# if weightNormalization matches
			matches_weightNormalization = check_match(weightNormalization_line, weightNormalization)
			# if kNN matches
			matches_kNN = check_match(kNN_line, kNN, exact = True)
			# if acc matches
			matches_acc = check_match(acc_line, acc)
			
			
			# check if all conditions are met
			matches_all = matches_inputPath & matches_log_type & matches_WEmodel & matches_sigVecStrat & matches_sigWeightStrat & matches_minDf \
							& matches_maxDf & matches_sigSize & matches_distFunctionWMD & matches_weightNormalization & matches_kNN & matches_acc

			if matches_all:
				# fill row with index == experiment_string
				df_row = pd.DataFrame([[log_type_line, experimentInput_line, WEmodel_line, sigVecStrat_line, sigWeightStrat_line,\
										minDf_line, maxDf_line, sigSize_line, distFunctionWMD_line, weightNormalization_line, kNN_line, acc_line]], \
											columns = ["type", "experimentInput",  "we_model", "sigVecStrat", "sigWeightStrat", "minDf", "maxDf", \
											"sigSize", "distFunctionWMD", "weightNormalization", "k", "accuracy"])
				df = pd.concat([df,df_row], ignore_index = True)
			
	return df
	
	
	
	
def main():
	# PARSE ARGS
	############
	parser = argparse.ArgumentParser()
	
	parser.add_argument("log_path"  , help="path to the results.log" , action = "store")
	parser.add_argument("inputPath" , help="string to be found in 'inputPath' of the log line; e.g. 'IMDB_small' matches all experiment folders that have 'IMDB_small' in it" , action = "store")
	# these parameters are optional and allow to specify the query
	# e.g. --log_type LOO will only retrieve logs that include LOO results (generated by pipeline) etc.
	parser.add_argument("--log_type" 		, help="logtype: either 'LOO' or 'classifier' " , action = "store")
	parser.add_argument("--WEmodel" 		, help="path to the word embedding" , action = "store")
	parser.add_argument("--sigVecStrat" 	, help="signature vector strategy string" , action = "store")
	parser.add_argument("--sigWeightStrat" 	, help="signature weight strategy string" , action = "store")
	parser.add_argument("--minDf" 			, help="min df, drop all with df lower than min_df" , action = "store")
	parser.add_argument("--maxDf" 			, help="max df, drop all with df higher than max_df" , action = "store")
	parser.add_argument("--sigSize" 		, help="signature size" , action = "store")
	parser.add_argument("--distFunctionWMD" , help="distance function to use for WMD, either 'euclidean' or 'cosine'" , action = "store")
	parser.add_argument("--weightNormalization" , help="weight normalization for signature weights, either 'euclidean' or 'histogram'" , action = "store")
	parser.add_argument("--kNN" 			, help="k nearest neighbors for inference" , action = "store")
	parser.add_argument("--acc" 			, help="accuracy" , action = "store")
	
	args = parser.parse_args()
	
	log_path				= args.log_path
	inputPath				= args.inputPath
	
	log_type				= args.log_type
	WEmodel					= args.WEmodel
	sigVecStrat				= args.sigVecStrat
	sigWeightStrat			= args.sigWeightStrat
	minDf					= args.minDf
	maxDf					= args.maxDf
	sigSize					= args.sigSize
	distFunctionWMD			= args.distFunctionWMD
	weightNormalization		= args.weightNormalization
	kNN						= args.kNN
	acc						= args.acc
	
	
	# change data type, if necessary
	if minDf != None:
		minDf = float(minDf)
	if maxDf != None:
		maxDf = float(maxDf)
	if sigSize != None:
		sigSize = int(sigSize)
	if kNN != None:
		kNN = int(kNN)
	if acc != None:
		acc = float(acc)
	
	# read log and write results into a pandas dataframe
	df = read_log(log_path,  inputPath, log_type = log_type, WEmodel = WEmodel, sigVecStrat = sigVecStrat, sigWeightStrat = sigWeightStrat, minDf = minDf, \
						maxDf = maxDf, sigSize = sigSize, distFunctionWMD = distFunctionWMD, weightNormalization = weightNormalization, kNN = kNN, acc = acc)
	
	# show query results
	print("Consider {} results".format(df.shape[0]))
	print("The mean accuracy for the query is {}.".format(df.loc[:,"accuracy"].mean()))
	print("The max. accuracy for the query is {}.".format(df.loc[:,"accuracy"].max()))
	print("The min. accuracy for the query is {}.".format(df.loc[:,"accuracy"].min()))
	
	max_indices = df[df["accuracy"] == 0.81].index
	for max_index in max_indices:
		print(df.iloc[max_index,1])
	
	# EXAMPLE
	###################
	
	example = False
	
	if example:
		log_path   = "../../data/logs/results.log"
		inputPath = "../data/experiments/IMDB_supervised_small_247"
		#
		df = read_log(log_path,  inputPath, log_type = None, WEmodel = None, sigVecStrat = None, sigWeightStrat = None, minDf = None, \
						maxDf = None, sigSize = None, distFunctionWMD = None, weightNormalization = None, kNN = None, acc = None)
		print(df.shape)

						
	return
	
	
if __name__ == "__main__":
	main()

