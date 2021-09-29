import sys, os
import subprocess


# this file is supposed to glue together all bits and pieces that are in pipeline
# to conduct whole 'experiment-lines' in a single GO
# IDEA: insert a set of parameters --> create an experiment [including result]

# Structure
############
# 1. Split a corpus of positive and negative reviews into folders with equally many positive and negative reviews [say exp_folder_i, for i = 1,..n]
# 2. Create a script of single jobs to perform on the experiment folders (defined in 1.)
# 3. Run Script from 1. --> Log results to a separate log file
# 4. read out the results log
# 5. assemble experiment statistics


from pipeline.preprocessing.draw_experimens import create_data_folders

# 1. Create data folder

create_data_folders(basecaption = basecaption, positive_dir = positive_dir, negative_dir = negative_dir,\
		experiment_dir = experiment_dir, batch_size = batch_size)
