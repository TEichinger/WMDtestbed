#! /usr/bin/python3
import sys, os
import subprocess
import argparse


# reconstruct the experiment folders according to the input to <create_data_folder>, and delete the respective folders
# this method can be simplified [TODO]
def delete_data_folders(basecaption = "test", positive_dir = "../../data/corpora/IMDB/train/pos", negative_dir = "../../data/corpora/IMDB/train/neg",\
	experiment_dir = "../../data/experiments", batch_size = 50):
	# count the number of files in positive_dir and negative_dir
	num_pos, pos_file_paths = count_files(positive_dir, ".txt")
	num_neg, neg_file_paths = count_files(negative_dir, ".txt")
	batch_size = min(num_pos, num_neg, batch_size)
	# create file path batches [of size 'size'] (pos/neg)
	pos_path_batches = make_batches(pos_file_paths, batch_size)
	neg_path_batches = make_batches(neg_file_paths, batch_size)
	# mix batches (pos/neg) in (50/50) ratio
	path_batches = [pos_batch + neg_batch for pos_batch, neg_batch in zip(pos_path_batches, neg_path_batches)]
	print("Found {} negative files, {} positive files. Will fill {} experiment batches of size {} now.".format(str(num_neg), str(num_pos), str(len(path_batches)) , str(batch_size)))
	# for every batch:
	for i,batch in enumerate(path_batches, 1):
		# create a folder_path
		folder_path = os.path.join(experiment_dir, basecaption + '_' + str(i))
		# create/empty a folder in experiment_dir with the basecaption and a number
		delete_folder(folder_path)
	return
		
# delete (recursively and forcedly) the elements in 'target_dir'
def delete_folder(target_dir):
	subprocess.call(["rm" , "-rf", target_dir])
	print("Deleted {}.".format(target_dir))
	return

# make experiment folders from positive and negative labeled files from a corpus
# the experiment folders will have the basecaption_i, where i as an enumeration integer (e.g. test_1, test_2 ,..)
# [basecaption]    : <str> caption for the experiment folder (e.g. verbalising the batch size
# [positive_dir]   : 
# [negative_dir]   : 
# [experiment_dir] : 
# [batch_size]	   : <int> number of positive and negative files respectively (--> number of files per folder = 2 * batch_size)
def create_data_folders(basecaption = "test", positive_dir = "../../data/corpora/IMDB/train/pos", negative_dir = "../../data/corpora/IMDB/train/neg",\
	experiment_dir = "../../data/experiments", batch_size = 50):
	# count the number of files in positive_dir and negative_dir
	num_pos, pos_file_paths = count_files(positive_dir, ".txt")
	num_neg, neg_file_paths = count_files(negative_dir, ".txt")
	batch_size = min(num_pos, num_neg, batch_size)
	# create file path batches [of size 'size'] (pos/neg)
	pos_path_batches = make_batches(pos_file_paths, batch_size)
	neg_path_batches = make_batches(neg_file_paths, batch_size)
	# mix batches (pos/neg) in (50/50) ratio
	path_batches = [pos_batch + neg_batch for pos_batch, neg_batch in zip(pos_path_batches, neg_path_batches)]
	print("Found {} negative files, {} positive files. Will fill {} experiment batches of size {} now.".format(str(num_neg), str(num_pos), str(len(path_batches)) , str(batch_size)))
	# for every batch:
	for i,batch in enumerate(path_batches, 1):
		# create a folder_path
		folder_path = os.path.join(experiment_dir, basecaption + '_' + str(i))
		# create/empty a folder in experiment_dir with the basecaption and a number
		create_folder(folder_path)
		# copy the files into the folder
		copy_batch(batch,folder_path)
		
	print("Filled all data folders")
	return
	
	
# copy a list of files to a target directory
def copy_batch(file_paths, target_dir):
	for file_path in file_paths:
		subprocess.call(["cp", file_path, target_dir])
	return
	
	
# check if a given target directory exists, if yes, empty it, else create it
# [target_dir]: <str> string of the target directory path
def create_folder(target_dir):
	if os.path.isdir(target_dir):
		print("experiment folder {} already exists, remove its entries".format(target_dir))
		subprocess.call(["rm" , "-rf", target_dir])# os.path.join(target_dir, "*")])
		subprocess.call(["mkdir", "-p", target_dir])
	else:
		print("experiment folder {} does not exist, make directory".format(target_dir))
		subprocess.call(["mkdir", "-p", target_dir])		
	return

# for a list, and a modulo, return a list of lists with modulo elements per sublist
# [object_list]
# [modulo]
def make_batches(object_list, modulo):
	if modulo == 0:
		return object_list
	else:
		if len(object_list) % modulo == 0:
			result = [object_list[i*modulo:(i+1)*modulo] for i in range((len(object_list)//modulo))]
		else:
			result = [object_list[i*modulo:(i+1)*modulo] for i in range((len(object_list)//modulo)+1)]
		return result

	
# count the number of files ( with a certain extension) in a reference directory
def count_files(ref_dir, fileext):
	c = 0
	file_paths = []
	for root, dirs, files in os.walk(ref_dir):
		for name in files:
			# that are *.fileext files
			hasFileExt = os.path.splitext(name)[1] == fileext
			if hasFileExt:
				c = c + 1
				file_paths.append(os.path.join(root,name))
	return c, file_paths


	
def main():	
	parser = argparse.ArgumentParser()
	# PARSE ARGUMENTS
	####################
	parser.add_argument("experiment_dir" , help="directory to put the experiment folders into"	, action = "store")
	parser.add_argument("positive_dir"    , help="directory holding the positive files"											, action = "store")
	parser.add_argument("negative_dir"     , help="directory holding the negative files"					, action = "store")
	parser.add_argument("basecaption"      , help="base caption for the experiment folders, numberated via <basecaption>_i", action = "store")
	parser.add_argument("batch_size"      , help="number of positive and negative files per experiment folder respectively", action = "store")
	parser.add_argument("--delete"      , help="'1': delete any folders that have been created without this flag", action = "store")
	args = parser.parse_args()
		
	experiment_dir	= args.experiment_dir
	positive_dir	= args.positive_dir
	negative_dir	= args.negative_dir
	basecaption		= args.basecaption
	batch_size		= int(args.batch_size)
	delete			= args.delete

	
	# create experiment folders
	if not delete == "1":
		create_data_folders(basecaption = basecaption, positive_dir = positive_dir, negative_dir = negative_dir,\
		experiment_dir = experiment_dir, batch_size = batch_size)
	else:
		delete_data_folders(basecaption = basecaption, positive_dir = positive_dir, negative_dir = negative_dir,\
		experiment_dir = experiment_dir, batch_size = batch_size)
	
	
if __name__ == "__main__":
	main()