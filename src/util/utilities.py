# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime
import os,sys
from math import log
import pickle
import subprocess
import fasttext
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def file_path_to_file_name(file_path):
	""" Take a file path ('this/is/a/file/path/test_file.ext') and get the file name --> 'test_file'."""
	# get the basename (file name including extension)
	filename_plus_extension = os.path.basename(file_path)
	filename = os.path.splitext(filename_plus_extension)[0]

	return filename



# recursive function!
# for a list of parameter lists, build all possible combinations
# e.g. [[a,b,c,] ,[d,e]] --> [a,d], [a,e], [b,d], [b,e], [c,d], [c,e]
def make_grid(parameter_list_of_lists):
	first_list = parameter_list_of_lists[0]
	if len(parameter_list_of_lists) != 1:
		second_list = parameter_list_of_lists[1]
	# if the list contains only one element either the recursion has finished (or the input only contains one list in a list)
	else: # stop recursion
		return first_list

	rest = parameter_list_of_lists[2:]

	# do a recursion on make_grid, if the second list exists and is not empty
	if second_list == []:
		return first_list
	else:
		return make_grid([combine_lists(first_list, second_list)] + rest)


# take two lists of elements and build pairs as a list
# not outer:
# L = [1,2,3]; K = [4,5]
# result : [[1,4], [1,5], [2,4], [2,5], [3,4], [3,5]]
# outer :
# result : [[[1],[4]], [[1],[5]], [[2],[4]], [[2],[5]], [[3],[4]], [[3],[5]]]
def combine_lists(L, K, outer = False):
	result = []
	for l in L:
		for k in K:
			if type(l) != list:
				l = [l]
			if type(k) != list:
				k = [k]
			if not outer:
				result.append(l + k)
			else:
				result.append([l,k])
	return result




# calculate the accuracy of a list with true labels and a list with predicted labels as a float
# [true_labels] : <list> of true labels
# [predicted_labels]: <list> of predicted labels
def get_accuracy(true_labels, predicted_labels):
	counter = 0
	for true_label, predicted_label in zip(true_labels, predicted_labels):
		if true_label == predicted_label:
			counter += 1
	if len(true_labels) != len(predicted_labels):
		print("note that the size of the true and predicted labels do not coincide")
	if len(true_labels) == 0:
		print("!!!true labels is empty!!!")
	if len(predicted_labels) == 0:
		print("!!!predicted labels is empty!!!")
	accuracy = safe_division(counter,min(len(true_labels), len(predicted_labels)))

	return accuracy

# try to get true labels for files in 'to_label_file_paths' according to their location in either 'positive_folder'
# or 'negative_folder'
# output both accuracy and true_labels
def evaluate_prediction(predicted_labels, to_label_file_paths, positive_folder, negative_folder):
	true_labels = []
	for file_path in to_label_file_paths:
		# get filename
		filename = os.path.split(file_path)[1]
		# build path to the positive folder
		pos_file_path = os.path.join(positive_folder,filename)
		is_pos_file = os.path.isfile(pos_file_path)
		# build path to the negative folder
		neg_file_path = os.path.join(negative_folder, filename)
		is_neg_file = os.path.isfile(neg_file_path)

		if is_pos_file:
			true_labels.append(1)
		elif is_neg_file:
			true_labels.append(0)
		else:
			true_labels.append(-1)

	accuracy = get_accuracy(true_labels, predicted_labels)

	return accuracy, true_labels



# write an event into the result logarithm
# requires: [message] string that should be logged
def write_log(output_path, message):
	time_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S##")
	# create a log file, in case non with <output_path> exists
	if not os.path.isfile(output_path):
		with open( output_path, mode = "w") as f:
			pass

	with open(output_path, mode= 'a') as f:
		f.write(time_string + message + '\n')

	return

# log a leave-one-out accuracy [e.g. in pipeline.py]
def log_LOO_accuracy(output_path, input_info, we_model_path, sigvec_strat, sigweight_strat, min_df, max_df, sig_size, dist_function, vector_normalization, kNN, accuracy):
	# make standardized log message
	log_message = make_log_message("LOO:", input_info, we_model_path, sigvec_strat, sigweight_strat, min_df, max_df, sig_size, dist_function, vector_normalization, kNN, accuracy)
	if type(vector_normalization) == type(None):
		vector_normalization = "none"
	write_log(output_path, log_message)
	print('Leave-one-out Accuracy logged')
	return

# log a semisupervision accuracy [e.g. in classifier.py]
def log_classifier_accuracy(output_path, knowledge_path, label_path, we_model_path,  sigvec_strat, sigweight_strat, min_df, max_df, sig_size, dist_function, vector_normalization, kNN, accuracy):
	# create input_info
	input_info = "[[knowledgePath=" + knowledge_path + "labelPath=" + label_path + "]]"

	# make standardized log message
	log_message = make_log_message("classifier:", input_info, we_model_path, sigvec_strat, sigweight_strat, min_df, max_df, sig_size, dist_function, vector_normalization, kNN, accuracy)

	write_log(output_path, log_message)
	print('Classifier Accuracy logged')
	return



# for a list of parameters, output a standardized log message line
def make_log_message(result_type, input_info, we_model_path, sigvec_strat, sigweight_strat, min_df, max_df, sig_size, dist_function, vector_normalization, kNN, accuracy):
	log_message  = result_type # e.g. "LOO:" or "classifier:"
	log_message += "inputPath=" + input_info +";"
	log_message += "WEmodel="+ we_model_path +";"
	log_message += "sigVecStrat="+sigvec_strat +";"
	log_message += "sigWeightStrat=" + sigweight_strat +";"
	log_message += "minDf=" + str(min_df) +";"
	log_message += "maxDf=" + str(max_df) +";"
	log_message += "sigSize=" + str(sig_size) +";"
	log_message += "distFunctionWMD=" + str(dist_function) +";"
	log_message += "vectorNormalization=" + str(vector_normalization) +";"
	log_message += "kNN=" + str(kNN) +";"
	log_message += "acc=" + str(accuracy) +";"
	return log_message


# load a pickle file from a pickle_path
def load_pickle(pickle_path):
    # load both users' pickle files
    with open(pickle_path, 'rb') as f:
        pickle_load = pickle.load(f)
    return pickle_load

# calculate the norm of a vector unless it is non-zero.
# return 1 else
def safe_norm(vector):
	norm = np.linalg.norm(vector)
	if norm == 0:
		return 1.0
	return norm



# normalize a list (vector) according to euclidean norm,
# return the zero vector when the vector norm is zero
def safe_normalize_vec(vector):
	norm = euclidean_norm(vector)
	if norm == 0:
		return [0 for _ in vector]
	else:
		return [el/norm for el in vector]

# normalize a list (vector) of positives such that they sum to 1
# return the zero vector when the vector norm is zero
def safe_normalize_histo(vector):
	denominator = sum(vector)
	if denominator == 0:
		return [0 for _ in vector]
	else:
		return [el/denominator for el in vector]




# Generate a <list> of <str> tokens for a given document <str>
# requires :
#	[document]     : <str>, document to tokenize
#	[make_unicode] : <bool>, cast the word tokens to unicode (via utf-8; by replacing bytes that cannot be decoded)
def custom_tokenize(document, make_unicode = True, unique = False):
	# make the work tokens utf-8 encoded, if specified
	############################
	#if make_unicode == True:
	#		document = document.decode('utf8',"replace")

	# drop (some) punctuation
	############################
	document = remove_html_tags(document)

	# remove some puctuation
	############################
	document = document.replace(",", " ")
	document = document.replace(".", " ")
	document = document.replace("(", " ")
	document = document.replace(")", " ")

	# tokenize
	############################
	word_tokens = word_tokenize(document)#document.split()

	# lowercase everything?
	############################
	word_tokens = [token.lower() for token in word_tokens]

	# drop stop words
	############################
	word_tokens, stop_words_used = drop_stop_words(word_tokens)

	# drop duplicates if unique == True
	############################
	if unique:
		word_tokens = list(set(word_tokens))

	return word_tokens, stop_words_used



# load a fasttext model
def load_we_model(we_model_path):
	print("Load WE model...")
	we_model_path 	= we_model_path
	we_model    	= fasttext.load_model(we_model_path)
	print("Done!")
	return we_model

# remove html tags such as <...>
def remove_html_tags(input_string):
	regex 		= re.compile(r"<.*?>", re.IGNORECASE)
	output_string   = re.sub(regex, "",input_string)
	return output_string

# removes Hashtags, Twitter mentions and twitter picture links and then returns the input string.
# not the prettiest/fastest implementation. Chosen to be readeable and fast enough for the operations needed in the pipeline.
def remove_twitter_tags(input_string):
	re_hashtag = re.compile('#\w+')
	re_mention = re.compile('@\w+')
	re_twitter_link = re.compile('\S*pic\.twitter\.com/\S*')

	# re.sub replaces the word with empty space if it matches the regex
	input_string = re.sub(re_hashtag, '', input_string)
	input_string = re.sub(re_mention, '', input_string)
	input_string = re.sub(re_twitter_link, '', input_string)
	return input_string


# drop stop_words; tokenize first, if it is a text
def drop_stop_words(text):
	# set stop words
	stop_words = set(stopwords.words('english'))
	if type(text) != list:
		print('text is not a list')
		text,_= custom_tokenize(text)
	# drop stop words
	non_stop_word_tokens = [token for token in text if not token in stop_words]
	# stop_words
	stop_words_used      = [token for token in text if token in stop_words]
	return non_stop_word_tokens, stop_words_used

# Function that is used in order to convert a list of keyphrases to a list of keywords with sig_size as its maximum length
# For that, the first k words from the keyphrase list are put into the new list, excluding stopwords that may occur in the keyphrase list
def convert_3_grams_into_single_keywords(sig_size, top_k_keywords, normalized_word_weights):

	# temporary lists that are returned as new top_k_keywords and normalized_word_weights in the end
	temp_top_k_keywords = []
	temp_normalized_word_weights = []

	# for every keyphrase in the initial list
	for i, str in enumerate(top_k_keywords):
		# break the loop if there are enough keywords in the output list
		if len(temp_top_k_keywords) >= sig_size:
			break
		# for every word in the keyphrase
		for word in str.split():
			# if the word is not a stopword and if it is not in the temp_top_k_keywords list already
			if word not in stopwords.words() and word not in temp_top_k_keywords:
				# append the word to the temp_top_k_keywords
				temp_top_k_keywords.append(word)
				temp_normalized_word_weights.append(normalized_word_weights[i])
			# break the loop if there are enough keywords in the output list
			if len(temp_top_k_keywords) == sig_size:
				break

	return temp_top_k_keywords, temp_normalized_word_weights

# return the current time as a timestamp in the format "%Y_%m_%d_%H_%M_%S"
def get_current_timestamp():
	return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# for files in an input_dir, that are duplicates of files in positive_dir or negative dir
# return labels (1: positive, 0: negative, -1: unknown)
# as either a pandas DataFrame ('as_list' == False), or a list (else)
def get_labels(input_dir, positive_dir, negative_dir, as_list = False):
	# create 'label' file in 'input_dir'
	subprocess.call(["python3", "../preprocessing/setup_experiment.py", input_dir, positive_dir, negative_dir])
	# load labels from that 'label' file
	label_df = load_labels(input_dir + '/label_file', as_list = as_list)
	return label_df

# load labels [format: 'file_name_without_extension' 'label'], where 'label' in {0,1,-1}
# the output is a pandas dataframe
def load_labels(label_path, as_list = False):
	labels = pd.read_csv(label_path, names = ['label'], sep = ' ', index_col = [0])
	if as_list:
		labels = labels.loc[:,'label'].tolist()
	# CAVEAT does not work in python2
	# cast index to str, if ever it is to by converted to int by pandas
	labels.index = labels.index.map(str)
	return labels

# load an edge graph [format:
# dtypes: object, object, float64
# source, target, weight
# ID11, ID12, 0.79
# ID21, ID22, 0.60
# ...]
# as a pandas df
def load_edge_graph(edge_graph_path):
	return pd.read_csv(edge_graph_path)



def check_supervised(label_df):
	""" Check if the label_df (<pd.DataFrame>) contains only labels 'positive' and 'negative'
	If yes, return False, else True
	"""
	if list(label_df[label_df['label'] == "unknown"]['label']):
		return False
	else:
		return True

# calculate the euclidean norm of 'vector' (<list>)
def euclidean_norm(vector):
	from math import sqrt
	return sqrt(sum([el**2 for el in vector]))




# define safe division, that is in case of division by zero, the result is zero
def safe_division(numerator, denominator):
	if denominator == 0:
		return 0
	else:
		return numerator/denominator

# apply the mathematical logarithm if the 'input' is not zero, return zero else
def safe_log(input):
	if input == 0:
		return 0
	else:
		return log(input)

# safely calculate the cosine distance between two vectors of type <list>
# cos_distance(x1,x2) = 1 - cos(x1,x2)    (in [0,1])
# identical non-zero vectors --> 0
# orthogonal vectors         --> 1
# opposing vectors			 --> 2
def cosine_distance(x1,x2):
	#nonzerofound1 = False
	#nonzerofound2 = False
	#for i in range(0,len(x1)):
	#	if(x1[i] != 0):
	#		nonzerofound1 = True
	#	if(x2[i] != 0):
	#		nonzerofound2 = True
	#if(not (nonzerofound1 & nonzerofound2)):
	#	return 0
	norm_x1 = np.linalg.norm(x1)
	norm_x2 = np.linalg.norm(x2)
	if not (norm_x1 == 0 and norm_x2 == 0):
		dot_product = np.dot(x1,x2)
		cosine_distance = 1.0 - (dot_product / ((norm_x1 * norm_x2)))
	else:
		cosine_distance = 0.0
	return (cosine_distance)

# safely calculate the cosine similarity between two vectors of type <list>
# cos_distance(x1,x2) = 1 - cos(x1,x2)    (in [0,1])
# identical non-zero vectors --> 1
# orthogonal vectors         --> 0
# opposing vectors			 --> -1
def cosine_similarity(x1,x2):
	norm_x1 = np.linalg.norm(x1)
	norm_x2 = np.linalg.norm(x2)
	if not (norm_x1 == 0 and norm_x2 == 0):
		dot_product = np.dot(x1,x2)
		cosine_distance = dot_product / (norm_x1 * norm_x2)
	else:
		cosine_distance = None
	return cosine_distance


# calculate the euclidean distance between two vectors of type <list>
def euclidean_distance(x1,x2):
	return np.sqrt( np.sum((np.array(x1) - np.array(x2))**2) )

# create a pickle file of the classifier
def save_model(model, model_name):
	with open(model_name , 'wb') as f:
		pickle.dump(model, f)
	return

# create a pickle file of the classifier
def load_model(model_name):
	with open(model_name , 'rb') as f:
		model = pickle.load(f)
	return model

# read a file and return it
def read_file(file_path):
	with open(file_path) as f:
		document = f.read()
	return document

# load a review-file, and output it as a <str>; expects 'utf-8' encoding
# requires:
#	[review_path] : path to the review file
def load_review(review_path):
	with open(review_path, mode = 'r') as f:
		review = f.read()

	#process review
	return review


# take a list of word tokens ('term_list') and create a dictionary
# with unique terms and term count values: e.g. ['a', 'b', 'a'] --> {'a': 2, 'b':1}
# requires:
#	[term_list] : <list> of <str>
def create_BoW_dict(term_list):
	BoW_dict = {}
	unq_terms = set(term_list)
	for unq_term in unq_terms:
		BoW_dict[unq_term] = term_list.count(unq_term)
	return BoW_dict


def extract_unique_txtfiles_from_dirs(review_dirs):
	"""For a list of review directories (review_dirs), return a list of unique *.txt files
		review_dirs: <list> of <str> paths of directories to query for .txt files. Does not consider subdirectories."""
	files_to_register = []
	# for all review_dirs
	for review_dir in review_dirs:
		# pick all *.txt files in review_dir
		files_to_register.extend([review_dir+'/'+name for name in list(os.walk(review_dir))[0][2] if os.path.splitext(name)[1] == '.txt'])
	# rid all duplicates
	files_to_register = list(set(files_to_register))

	return files_to_register

# collect file paths from a given directory
# if ext is given, then only matches a given file file, if it has the extensionextention
# e.g. ext = ".txt"
def get_file_paths(refDir, ext = "", verbose = False):
	file_paths = []
	if verbose:
		print("Find file paths in {}...".format(refDir))

	for root, dirs, files in os.walk(refDir):
		for name in files:
			# that are *.txt files
			isTxtFile = os.path.splitext(name)[1] == ext
			if isTxtFile:
				file_path = os.path.join(root,name)
				file_paths.append(file_path)

	return file_paths

# collect folder paths from a given directory
# such that the folder name matches a basecaption
# basecaption = "IMDB_supervised_small" --> matches IMDB_supervised_small_1
def get_folder_paths(refDir, basecaption = ""):
	return




if __name__ == "__main__":
	#print(get_accuracy([1,0,0,0],[1,0,0]))
	#print(combine_lists([1,2,3],[5,6,7]))
	#print(euclidean_distance([0,1],[1,0]))
	print(cosine_distance([-1,0],[1,0]))
