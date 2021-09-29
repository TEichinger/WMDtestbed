#! /usr/bin/python3
import sys
from glob import glob
import os
import subprocess

# create a 'label_file' for an 'experiment_folder', where the labels are given by being either in a 
# folder of 'positive' [named: 'pos'] or 'negative' [named: 'pos'] review files, else the file will be flagged as 'unknown'
def create_label_file(experiment_folder, positive_folder, negative_folder,label_file):
	# delete any existing label file (to be removed perhaps)
	with open(label_file, mode = 'w') as f:
		pass	
	# for every (review) file in 'experiment_folder', see if it is contained in either positive_folder, or negative_folder, then write a line
	# in a label file in the format: where positive, and negative, unknown
	# (review)_file_name {1,0,-1}
	# 
	# check if it is in the positive folder
	for filename in os.listdir(experiment_folder):
		if filename.endswith(".txt"):
			# build path to the positive folder
			pos_file_path = os.path.join(positive_folder,filename)
			is_pos_file = os.path.isfile(pos_file_path)
			# build path to the negative folder
			neg_file_path = os.path.join(negative_folder, filename)
			is_neg_file = os.path.isfile(neg_file_path)
		
			#print(pos_file_path)
			#print(neg_file_path)
			
			if is_pos_file:
				add_row(label_file, filename, 'positive')
			elif is_neg_file:
				add_row(label_file, filename, 'negative')
			else:
				add_row(label_file, filename, 'unknown')
	print("Wrote label file in {}.".format(label_file))
			
			
# add a row in the 'label_file'
def add_row(label_file, file_name, label):
	with open(label_file, mode = 'a') as f:
		# write 'file_name' without file extention
		f.write(os.path.splitext(file_name)[0] + ' ' + label + '\n')

		


	
	
	
def main():

	# PARSE ARGS
	#############
	# folder containing the experiment reviews
	experiment_folder = sys.argv[1]
	# folder containing the positive reviews
	positive_folder   = sys.argv[2]
	# folder containing the negative reviews
	negative_folder   = sys.argv[3]
	# name label_file (path)
	label_file = experiment_folder+'/label_file'
	
	print(label_file)
	
	# create a label file, overwrite, if another exists already (!)
	create_label_file(experiment_folder, positive_folder, negative_folder, label_file)

	
if __name__ == "__main__":
	main()
