import sys, os
from kwe_methods.kwe_methods import YakeKeywordextractor, RakeKeywordextractor, \
	TfidfKeywordextractor
from util.utilities import load_we_model, get_current_timestamp, file_path_to_file_name, \
	safe_normalize_vec, remove_twitter_tags
from wmd.get_pickle_files import apply_model, create_pickle

sys.path.append('../tfidf/')
sys.path.append("../util")
sys.path.append("../wmd")

def make_pickle_file(file_path, pickle_dir, we_model, sig_size = 10, sigvec_strat = "tfidf", sigweight_strat = "tfidf", vector_normalization=False, clean_twitter_data=False, allow_3_grams=False, tfidf_vectorizer ="none", verbose = False):
	"""Pickle a single file.
		[file_path]				: <str> path to the (*.txt) file to pickle a (WMD) signature of
		[pickle_dir]			: <str> path to the directory of picklefiles; store pickle file there
		[we_model]				: a loaded we model (via <load_we_model> in utilities.py)
		[sig_size]				: <int> signature size
		[sigvec_strat]			: <str> strategy string to define the vector selection strategy (e.g. 'tfidf' for top k tfidf valued vectors)
		[sigweight_strat]		: <str> strategy string to define the weight selection strategy (e.g. 'tfidf' for tfidf weights)
		[vector_normlization]	: <bool> True, if word vectors are to be normalized to length 1 (euclidean distance)
		[clean_twitter_data]	: <bool> If True removes hashtags, mentions and picture-links from the input-data. Recommended when using twitter-data as input
		[tfidf_vectorizer]		: a tfidf_vectorizer-instance created in pipeline.py and needed if kwe-method is tfidf
		[verbose]				: <bool> True: more information on the CLI
	"""
	# 'file/path/ref_file.txt' --> 'ref_file'
	file_name   = file_path_to_file_name(file_path)
	# if 'to_label_path' points at a *.txt file
	isTxtFile = os.path.splitext(file_path)[1] == '.txt'
				
	if isTxtFile:
		# use canonical pickle_path
		pickle_path = os.path.join(pickle_dir, os.path.splitext(file_name)[0]) + '.pk'

		# Create a pickle file
		#######################
		# load the text
		with open(file_path) as f:
			document = f.read()
			# clean twitter data if specified
			if clean_twitter_data:
				document = remove_twitter_tags(document)

			# When 3-grams are allowed the respective KWE method class is called accordingly
			if allow_3_grams:
				# KWE is done via the provided KWE method, functionality can be found in the 'kwe_methods' directory
				if sigvec_strat == "yake" and sigweight_strat =="yake":
					kwe = YakeKeywordextractor(document, sig_size=sig_size, n_gram_size=3)
					word_tokens, word_weights = kwe.run_extraction()

				elif sigvec_strat == "rake" and sigweight_strat == "rake":
					kwe = RakeKeywordextractor(document, sig_size=sig_size, n_gram_size=3)
					word_tokens, word_weights = kwe.run_extraction()

				elif sigvec_strat == "tfidf" and sigweight_strat =="tfidf":
					kwe = TfidfKeywordextractor(document, sig_size=sig_size, tfidf_vectorizer=tfidf_vectorizer, n_gram_size=3)
					word_tokens, word_weights = kwe.run_extraction()
				else:
					sys.exit("Please enter a valid KWE-method as a starting parameter. You entered: sigvec_strat -> +" + sigvec_strat + " sigweight_strat -> " + sigweight_strat)

			# when 3-grams are NOT allowed the respective KWE method is called with 'n_gram_size=1'
			else:
				# KWE is done via the provided KWE method, functionality can be found in the 'kwe_methods' directory
				if sigvec_strat == "yake" and sigweight_strat == "yake":
					kwe = YakeKeywordextractor(document, sig_size=sig_size, n_gram_size=1)
					word_tokens, word_weights = kwe.run_extraction()

				elif sigvec_strat == "rake" and sigweight_strat == "rake":
					kwe = RakeKeywordextractor(document, sig_size=sig_size, n_gram_size=1)
					word_tokens, word_weights = kwe.run_extraction()

				elif sigvec_strat == "tfidf" and sigweight_strat == "tfidf":
					kwe = TfidfKeywordextractor(document, sig_size=sig_size, tfidf_vectorizer=tfidf_vectorizer, n_gram_size=1)
					word_tokens, word_weights = kwe.run_extraction()
					# if no valid kwe-method is given, quit the pipeline
				else:
					sys.exit("Please enter a valid KWE-method as a starting parameter. You entered: sigvec_strat -> +" + sigvec_strat + " sigweight_strat -> " + sigweight_strat)

		# apply word embedding
		word_vectors = apply_model(word_tokens, we_model)

		# normalize word_vectors
		if vector_normalization == "euclidean":
			word_vectors = [safe_normalize_vec(el) for el in word_vectors]

		create_pickle(word_vectors, word_weights, pickle_path)

		# PRINT SIGNATURE WORDS AND WEIGHTS, IF NEEDED
		###############################################
		if verbose:
			print("word tokens:")
			print(word_tokens)
			print("word weights")
			print(word_weights)

		if "Schumer" in file_name:
			print("Filename: ", file_name)
			print("Keywords: ", word_tokens)
					
		return pickle_path
	else:
		return

	
def make_pickle_files(file_paths, pickle_dir, we_model_path, sig_size = 10, sigvec_strat = "tfidf" , sigweight_strat = "tfidf", vector_normalization=False,  clean_twitter_data=False, allow_3_grams=False , tfidf_vectorizer = "none",verbose = False, ):
	"""Pickles all input files by using the 'make-pickle-file' function.
		[file_path]				: <str> path to the (*.txt) file to pickle a (WMD) signature of
		[pickle_dir]			: <str> path to the directory of picklefiles; store pickle file there
		[we_model]				: a loaded we model (via <load_we_model> in utilities.py)
		[sig_size]				: <int> signature size
		[sigvec_strat]			: <str> strategy string to define the vector selection strategy (e.g. 'tfidf' for top k tfidf valued vectors)
		[sigweight_strat]		: <str> strategy string to define the weight selection strategy (e.g. 'tfidf' for tfidf weights)
		[vector_normlization]	: <bool> True, if word vectors are to be normalized to length 1 (euclidean distance)
		[clean_twitter_data]	: <bool> If True removes hashtags, mentions and picture-links from the input-data. Recommended when using twitter-data as input
		[tfidf_vectorizer]		: a tfidf_vectorizer-instance created in pipeline.py and needed if kwe-method is tfidf
		[verbose]				: <bool> True: more information on the CLI
	"""

	output_pickle_paths = []

	# load fattext model
	we_model = load_we_model(we_model_path)

	# CREATE PICKLE SUBDIRECTORY
	###################################################
	# get the timestring of now
	time_string = get_current_timestamp() +"##"
	# define a directory to put pickle files into (these are temporary files for later analysis)
	pickle_folder = os.path.join(pickle_dir, time_string)[:-2]
	# make the directory, if it does not yet exist
	if not os.path.isdir(pickle_folder) :
		os.mkdir(pickle_folder)

	# RUN ROUTINE FOR PICKLING REVIEWS
	###########################
	# for all 'to_label_file_paths'
	for i, file_path in enumerate(file_paths, 1):
		print("{}/{}({}%) files pickled.".format(i, len(file_paths), round(100*(i/len(file_paths)),2)), end = "\r")
		# pickle the file at file_path, save it to pickle_folder, and return the corresponding pickle_path

		# if tfidf is chosen as kwe method, call 'make-pickle-file' with the optional tfidf-vectorizer instance
		if sigvec_strat == "tfidf" and sigweight_strat =="tfidf":
			pickle_path = make_pickle_file(file_path, pickle_folder, we_model, sig_size=sig_size,
										sigvec_strat=sigvec_strat, sigweight_strat=sigweight_strat, vector_normalization=vector_normalization,
										tfidf_vectorizer=tfidf_vectorizer, clean_twitter_data=clean_twitter_data,allow_3_grams=allow_3_grams, verbose=verbose)

		# if tfidf ist NOT chosen, call the 'make-pickle-file' without the tfidf-vectorizer instance
		else:
			pickle_path = make_pickle_file(file_path, pickle_folder, we_model, sig_size = sig_size, sigvec_strat = sigvec_strat, sigweight_strat = sigweight_strat, vector_normalization = vector_normalization, clean_twitter_data=clean_twitter_data, allow_3_grams=allow_3_grams, verbose = verbose)

		# remember the pickle path; either the precalculated one, or the canonical one
		output_pickle_paths.append(pickle_path)

	print("Collected all pickle paths")
	return output_pickle_paths

def main():
	pass

if __name__ == "__main__":
	main()





