import subprocess


# function that writes a script with repetitive elements:
# e.g. python foo.py expt1 &
#      python foo.py expt2 &
#      wait
#	   ...
# input: list of strings to be entered separated by ' ', where there may be block_separators
# that appear after block_size lines
# enter {iterator} when you want to have an iterator
def make_script(base_line, number_of_lines, script_path, block_size, block_separator = "wait", end= "echo DONE!", append = False):
	
	# if a new (!) script should be created
	if not append:
		# clear previous script
		with open(script_path, mode = 'w') as f:
			pass
		# make the script runnable under linux	
		subprocess.call(["chmod", "+x", script_path])
	
	# for 'number_of_lines' times
	for i in range(number_of_lines):
		# create a line for the script
		line = base_line.format(iterator = str(i+1))
		# append the line to the script
		write_line(script_path, line + '\n')
		# every 'block_size' lines, add a block separator line
		if (i+1) % block_size == 0:
			write_line(script_path, block_separator + '\n')
	
	# add a final line
	write_line(script_path, end+'\n')
	
	return

# write a line in a script [actually appending]
def write_line(script_path, line):
	with open(script_path, mode = 'a') as f:
		f.write(line)


def main():
	# note that when you activate the --reset switch, some experiments might crash when they want to load a pickled file 
	# yet another (parallel procedure) has deleted the pickle directory, perhaps it makes sense here to use the experiment folder in order to store the pickle files there
	# and also use the --reset switch to delete only files in there !
	#
	"""
	############################################
	# small series (100 knowledge instances) LOO [default 1]
	############################################
	# euclidean vs. cosine WMD
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/test_we.bin  50  50  tfidf  tfidf  ../data/logs/results_eucVScosineDistFunc_small.log --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --weight_normalization histogram &", number_of_lines = 250, script_path = "../eucVScosineDistFunc_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!")	
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/test_we.bin  50  50  tfidf  tfidf  ../data/logs/results_eucVScosineDistFunc_small.log --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --weight_normalization histogram --euclideanWMD 1&", number_of_lines = 250, script_path = "../eucVScosineDistFunc_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)	
	# minDf
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/test_we.bin  50  50  tfidf  tfidf  ../data/logs/results_minDf_small.log --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --weight_normalization histogram &", number_of_lines = 250, script_path = "../minDf_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!")	
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/test_we.bin  50  50  tfidf  tfidf  ../data/logs/results_minDf_small.log --pickle 1  --reset 1 --min_df 0.05 --max_df 1.0 --weight_normalization histogram &", number_of_lines = 250, script_path = "../minDf_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)	
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/test_we.bin  50  50  tfidf  tfidf  ../data/logs/results_minDf_small.log --pickle 1  --reset 1 --min_df 0.10 --max_df 1.0 --weight_normalization histogram &", number_of_lines = 250, script_path = "../minDf_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)	
	# maxDf
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/test_we.bin  50  50  tfidf  tfidf  ../data/logs/results_maxDf_small.log --pickle 1  --reset 1 --min_df 0.0 --max_df 1.0 --weight_normalization histogram &", number_of_lines = 250, script_path = "../maxDf_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!")
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/test_we.bin  50  50  tfidf  tfidf  ../data/logs/results_maxDf_small.log --pickle 1  --reset 1 --min_df 0.0 --max_df 0.95 --weight_normalization histogram &", number_of_lines = 250, script_path = "../maxDf_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/test_we.bin  50  50  tfidf  tfidf  ../data/logs/results_maxDf_small.log --pickle 1  --reset 1 --min_df 0.0 --max_df 0.90 --weight_normalization histogram &", number_of_lines = 250, script_path = "../maxDf_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)
	# self-trained vs pretrained
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  tfidf  tfidf  ../data/logs/results_selfVSpretrained_small.log --pickle 1  --reset 1 --min_df 0.0 --max_df 1.0 --weight_normalization histogram &", number_of_lines = 250, script_path = "../selfVSpretrained.sh", block_size = 5, block_separator = "wait", end= "echo DONE!")
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/test_we.bin  50  50  tfidf  tfidf  ../data/logs/results_selfVSpretrained_small.log --pickle 1  --reset 1 --min_df 0.0 --max_df 1.0 --weight_normalization histogram &", number_of_lines = 250, script_path = "../selfVSpretrained.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)
	# weight normalization
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/test_we.bin  50  50  tfidf  tfidf  ../data/logs/results_eucVShistoWeightNorm_small.log --pickle 1  --reset 1 --min_df 0.0 --max_df 1.0 --weight_normalization histogram &", number_of_lines = 250, script_path = "../eucVShistoWeightNorm_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!")
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/test_we.bin  50  50  tfidf  tfidf  ../data/logs/results_eucVShistoWeightNorm_small.log --pickle 1  --reset 1 --min_df 0.0 --max_df 1.0 --weight_normalization euclidean &", number_of_lines = 250, script_path = "../eucVShistoWeightNorm_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)
	# sigsize
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/test_we.bin  50  25  tfidf  tfidf  ../data/logs/results_sigSizes_small.log --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --weight_normalization histogram &", number_of_lines = 250, script_path = "../sigSizes_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!")	
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/test_we.bin  50  50  tfidf  tfidf  ../data/logs/results_sigSizes_small.log --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --weight_normalization histogram &", number_of_lines = 250, script_path = "../sigSizes_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)	
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/test_we.bin  50  75  tfidf  tfidf  ../data/logs/results_sigSizes_small.log --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --weight_normalization histogram &", number_of_lines = 250, script_path = "../sigSizes_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)	
	# sigvecstrat
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/test_we.bin  50  50  tfidf  tfidf  ../data/logs/results_sigVecStrats_small.log --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --weight_normalization histogram &", number_of_lines = 250, script_path = "../sigVecStrats_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!")	
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/test_we.bin  50  50  unif   tfidf  ../data/logs/results_sigVecStrats_small.log --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --weight_normalization histogram &", number_of_lines = 250, script_path = "../sigVecStrats_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/test_we.bin  50  50  unif   unif  ../data/logs/results_sigVecStrats_small.log --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --weight_normalization histogram &", number_of_lines = 250, script_path = "../sigVecStrats_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)
	# sigweightstrat
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/test_we.bin  50  50  tfidf  tfidf  ../data/logs/results_sigWeightStrats_small.log --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --weight_normalization histogram &", number_of_lines = 250, script_path = "../sigWeightStrats_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!")	
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/test_we.bin  50  50  tfidf  unif   ../data/logs/results_sigWeightStrats_small.log --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --weight_normalization histogram &", number_of_lines = 250, script_path = "../sigWeightStrats_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/test_we.bin  50  50  unif  unif   ../data/logs/results_sigWeightStrats_small.log --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --weight_normalization histogram &", number_of_lines = 250, script_path = "../sigWeightStrats_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)	

	############################################
	# small series (100 knowledge instances) LOO [default 2] = default 1 with WE model replacement wiki.
	############################################
	# euclidean vs. cosine WMD
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  tfidf  tfidf  ../data/logs/results_eucVScosineDistFunc2_small.log --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --weight_normalization histogram &", number_of_lines = 250, script_path = "../eucVScosineDistFunc2_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!")	
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  tfidf  tfidf  ../data/logs/results_eucVScosineDistFunc2_small.log --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --weight_normalization histogram --euclideanWMD 1&", number_of_lines = 250, script_path = "../eucVScosineDistFunc2_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)	
	# minDf
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  tfidf  tfidf  ../data/logs/results_minDf2_small.log --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --weight_normalization histogram &", number_of_lines = 250, script_path = "../minDf2_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!")	
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  tfidf  tfidf  ../data/logs/results_minDf2_small.log --pickle 1  --reset 1 --min_df 0.05 --max_df 1.0 --weight_normalization histogram &", number_of_lines = 250, script_path = "../minDf2_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)	
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  tfidf  tfidf  ../data/logs/results_minDf2_small.log --pickle 1  --reset 1 --min_df 0.10 --max_df 1.0 --weight_normalization histogram &", number_of_lines = 250, script_path = "../minDf2_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)	
	# maxDf
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  tfidf  tfidf  ../data/logs/results_maxDf2_small.log --pickle 1  --reset 1 --min_df 0.0 --max_df 1.0 --weight_normalization histogram &", number_of_lines = 250, script_path = "../maxDf2_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!")
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  tfidf  tfidf  ../data/logs/results_maxDf2_small.log --pickle 1  --reset 1 --min_df 0.0 --max_df 0.95 --weight_normalization histogram &", number_of_lines = 250, script_path = "../maxDf2_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  tfidf  tfidf  ../data/logs/results_maxDf2_small.log --pickle 1  --reset 1 --min_df 0.0 --max_df 0.90 --weight_normalization histogram &", number_of_lines = 250, script_path = "../maxDf2_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)
	# weight normalization
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  tfidf  tfidf  ../data/logs/results_eucVShistoWeightNorm2_small.log --pickle 1  --reset 1 --min_df 0.0 --max_df 1.0 --weight_normalization histogram &", number_of_lines = 250, script_path = "../eucVShistoWeightNorm2_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!")
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  tfidf  tfidf  ../data/logs/results_eucVShistoWeightNorm2_small.log --pickle 1  --reset 1 --min_df 0.0 --max_df 1.0 --weight_normalization euclidean &", number_of_lines = 250, script_path = "../eucVShistoWeightNorm2_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)
	# sigsize
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  25  tfidf  tfidf  ../data/logs/results_sigSizes2_small.log --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --weight_normalization histogram &", number_of_lines = 250, script_path = "../sigSizes2_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!")	
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  tfidf  tfidf  ../data/logs/results_sigSizes2_small.log --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --weight_normalization histogram &", number_of_lines = 250, script_path = "../sigSizes2_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)	
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  75  tfidf  tfidf  ../data/logs/results_sigSizes2_small.log --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --weight_normalization histogram &", number_of_lines = 250, script_path = "../sigSizes2_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)	
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  100 tfidf  tfidf  ../data/logs/results_sigSizes2_small.log --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --weight_normalization histogram &", number_of_lines = 250, script_path = "../sigSizes2_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)	
	# sigvecstrat
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  tfidf  tfidf  ../data/logs/results_sigVecStrats2_small.log --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --weight_normalization histogram &", number_of_lines = 250, script_path = "../sigVecStrats2_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!")	
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  unif   tfidf  ../data/logs/results_sigVecStrats2_small.log --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --weight_normalization histogram &", number_of_lines = 250, script_path = "../sigVecStrats2_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  unif   unif  ../data/logs/results_sigVecStrats2_small.log --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --weight_normalization histogram &", number_of_lines = 250, script_path = "../sigVecStrats2_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)
	# sigweightstrat
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  tfidf  tfidf  ../data/logs/results_sigWeightStrats2_small.log --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --weight_normalization histogram &", number_of_lines = 250, script_path = "../sigWeightStrats2_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!")	
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  tfidf  unif   ../data/logs/results_sigWeightStrats2_small.log --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --weight_normalization histogram &", number_of_lines = 250, script_path = "../sigWeightStrats2_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  unif  unif   ../data/logs/results_sigWeightStrats2_small.log --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --weight_normalization histogram &", number_of_lines = 250, script_path = "../sigWeightStrats2_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)	

	
	############################################
	# small series (100 knowledge instances) LOO [default 3] = default 2 with WMDdistance replacement euclidean.
	############################################
	# minDf
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  tfidf  tfidf  ../data/logs/results_minDf3_small.log --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 250, script_path = "../minDf3_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!")	
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  tfidf  tfidf  ../data/logs/results_minDf3_small.log --pickle 1  --reset 1 --min_df 0.05 --max_df 1.0 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 250, script_path = "../minDf3_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)	
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  tfidf  tfidf  ../data/logs/results_minDf3_small.log --pickle 1  --reset 1 --min_df 0.10 --max_df 1.0 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 250, script_path = "../minDf3_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)	
	# maxDf
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  tfidf  tfidf  ../data/logs/results_maxDf3_small.log --pickle 1  --reset 1 --min_df 0.0 --max_df 1.0 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 250, script_path = "../maxDf3_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!")
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  tfidf  tfidf  ../data/logs/results_maxDf3_small.log --pickle 1  --reset 1 --min_df 0.0 --max_df 0.95 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 250, script_path = "../maxDf3_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  tfidf  tfidf  ../data/logs/results_maxDf3_small.log --pickle 1  --reset 1 --min_df 0.0 --max_df 0.90 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 250, script_path = "../maxDf3_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)
	# weight normalization
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  tfidf  tfidf  ../data/logs/results_eucVShistoWeightNorm3_small.log --pickle 1  --reset 1 --min_df 0.0 --max_df 1.0 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 250, script_path = "../eucVShistoWeightNorm3_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!")
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  tfidf  tfidf  ../data/logs/results_eucVShistoWeightNorm3_small.log --pickle 1  --reset 1 --min_df 0.0 --max_df 1.0 --weight_normalization euclidean --euclideanWMD 1 &", number_of_lines = 250, script_path = "../eucVShistoWeightNorm3_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)
	# sigsize
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  25  tfidf  tfidf  ../data/logs/results_sigSizes3_small.log --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 250, script_path = "../sigSizes3_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!")	
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  tfidf  tfidf  ../data/logs/results_sigSizes3_small.log --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 250, script_path = "../sigSizes3_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)	
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  75  tfidf  tfidf  ../data/logs/results_sigSizes3_small.log --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 250, script_path = "../sigSizes3_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)	
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  100 tfidf  tfidf  ../data/logs/results_sigSizes3_small.log --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 250, script_path = "../sigSizes3_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)	
	# sigvecstrat
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  tfidf  tfidf  ../data/logs/results_sigVecStrats3_small.log --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 250, script_path = "../sigVecStrats3_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!")	
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  unif   tfidf  ../data/logs/results_sigVecStrats3_small.log --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 250, script_path = "../sigVecStrats3_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  unif   unif  ../data/logs/results_sigVecStrats3_small.log --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 250, script_path = "../sigVecStrats3small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)
	# sigweightstrat
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  tfidf  tfidf  ../data/logs/results_sigWeightStrats3_small.log --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 250, script_path = "../sigWeightStrats3_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!")	
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  tfidf  unif   ../data/logs/results_sigWeightStrats3_small.log --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 250, script_path = "../sigWeightStrats3_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  unif  unif   ../data/logs/results_sigWeightStrats3_small.log --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 250, script_path = "../sigWeightStrats3_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)	
	
	############################################
	# small series (100 knowledge instances) LOO [default 4] = default 3 with vector normlalization before WMD calculation
	############################################
	# minDf
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  tfidf  tfidf  ../data/logs/results_minDf4_small.log --tfidf_triple_name '' --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --vector_normalization 1 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 250, script_path = "../minDf4_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!")	
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  tfidf  tfidf  ../data/logs/results_minDf4_small.log --tfidf_triple_name '' --pickle 1  --reset 1 --min_df 0.05 --max_df 1.0 --vector_normalization 1 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 250, script_path = "../minDf4_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)	
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  tfidf  tfidf  ../data/logs/results_minDf4_small.log --tfidf_triple_name '' --pickle 1  --reset 1 --min_df 0.10 --max_df 1.0 --vector_normalization 1 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 250, script_path = "../minDf4_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)	
	# sigsize
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  tfidf  tfidf  ../data/logs/results_sigSizes4_small.log --tfidf_triple_name '' --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --vector_normalization 1 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 250, script_path = "../sigSizes4_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!")	
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  75  tfidf  tfidf  ../data/logs/results_sigSizes4_small.log --tfidf_triple_name '' --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --vector_normalization 1 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 250, script_path = "../sigSizes4_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)	
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  100  tfidf  tfidf  ../data/logs/results_sigSizes4_small.log --tfidf_triple_name '' --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --vector_normalization 1 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 250, script_path = "../sigSizes4_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)	
	# sigvecstrat
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  tfidf  tfidf  ../data/logs/results_sigVecStrats4_small.log --tfidf_triple_name '' --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --vector_normalization 1 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 250, script_path = "../sigVecStrats4_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!")	
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  unif   tfidf  ../data/logs/results_sigVecStrats4_small.log --tfidf_triple_name '' --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --vector_normalization 1 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 250, script_path = "../sigVecStrats4_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)
	# sigweightstrat
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  tfidf  tfidf  ../data/logs/results_sigWeightStrats4_small.log --tfidf_triple_name '' --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --vector_normalization 1 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 250, script_path = "../sigWeightStrats4_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!")	
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  tfidf  unif   ../data/logs/results_sigWeightStrats4_small.log --tfidf_triple_name '' --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --vector_normalization 1 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 250, script_path = "../sigWeightStrats4_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  unif  unif   ../data/logs/results_sigWeightStrats4_small.log --tfidf_triple_name '' --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --vector_normalization 1 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 250, script_path = "../sigWeightStrats4_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)	
	
	############################################
	# small series (100 knowledge instances) LOO [default 5] = default 4 with sigWeightStrat bow
	############################################
	# minDf
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  tfidf  bow  ../data/logs/results_minDf5_small.log --tfidf_triple_name '' --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --vector_normalization 1 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 250, script_path = "../minDf5_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!")	
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  tfidf  bow  ../data/logs/results_minDf5_small.log --tfidf_triple_name '' --pickle 1  --reset 1 --min_df 0.05 --max_df 1.0 --vector_normalization 1 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 250, script_path = "../minDf5_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)	
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  tfidf  bow  ../data/logs/results_minDf5_small.log --tfidf_triple_name '' --pickle 1  --reset 1 --min_df 0.10 --max_df 1.0 --vector_normalization 1 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 250, script_path = "../minDf5_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)	
	# sigsize
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  tfidf  bow  ../data/logs/results_sigSizes5_small.log --tfidf_triple_name '' --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --vector_normalization 1 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 250, script_path = "../sigSizes5_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!")	
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  75  tfidf  bow  ../data/logs/results_sigSizes5_small.log --tfidf_triple_name '' --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --vector_normalization 1 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 250, script_path = "../sigSizes5_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)	
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  100  tfidf  bow  ../data/logs/results_sigSizes5_small.log --tfidf_triple_name '' --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --vector_normalization 1 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 250, script_path = "../sigSizes5_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)	
	# sigvecstrat
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  tfidf  bow  ../data/logs/results_sigVecStrats5_small.log --tfidf_triple_name '' --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --vector_normalization 1 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 250, script_path = "../sigVecStrats5_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!")	
	make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} ../data/we_models/wiki.en.bin  50  50  unif   bow  ../data/logs/results_sigVecStrats5_small.log --tfidf_triple_name '' --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --vector_normalization 1 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 250, script_path = "../sigVecStrats5_small.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)
	"""
	
	############################################
	# GOLD STANDARD (for pipeline):
	#
	# small series (100 knowledge instances) LOO [default 5] = default 4 with sigWeightStrat bow
	#
	#	sigSize					100 (max)
	#	sigVecStrat				tfidf
	#	sigWeightStrat			tfidf
	#	distance_function		euclidean
	#	weight_normalization	histogram
	#	vector_normalization	histogram
	#	min_df					0.0
	#	max_df					1.0
	#	WEmodel					pretrained
	#
	############################################
	
	#make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_verysmall/IMDB_supervised_verysmall_{iterator}	../data/we_models/wiki.en.bin  50  100  tfidf  tfidf  ../data/logs/results_batch_sizes_gold_standard.log --tfidf_triple_name '' --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --vector_normalization 1 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 2500, script_path = "../batch_sizes_gold_standard.sh", block_size = 5, block_separator = "wait", end= "echo DONE!")	
	#make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_smaller/IMDB_supervised_smaller_{iterator} 		../data/we_models/wiki.en.bin  50  100  tfidf  tfidf  ../data/logs/results_batch_sizes_gold_standard.log --tfidf_triple_name '' --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --vector_normalization 1 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 500, script_path = "../batch_sizes_gold_standard.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)	
	#make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_small/IMDB_supervised_small_{iterator} 			../data/we_models/wiki.en.bin  50  100  tfidf  tfidf  ../data/logs/results_batch_sizes_gold_standard.log --tfidf_triple_name '' --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --vector_normalization 1 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 250, script_path = "../batch_sizes_gold_standard.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)	
	#make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_smallish/IMDB_supervised_smallish_{iterator} 		../data/we_models/wiki.en.bin  50  100  tfidf  tfidf  ../data/logs/results_batch_sizes_gold_standard.log --tfidf_triple_name '' --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --vector_normalization 1 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 125, script_path = "../batch_sizes_gold_standard.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)	
	#make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_mediumish/IMDB_supervised_mediumish_{iterator} 	../data/we_models/wiki.en.bin  50  100  tfidf  tfidf  ../data/logs/results_batch_sizes_gold_standard.log --tfidf_triple_name '' --pickle 1  --reset 1 --min_df 0.00 --max_df 1.0 --vector_normalization 1 --weight_normalization histogram --euclideanWMD 1 &", number_of_lines = 100, script_path = "../batch_sizes_gold_standard.sh", block_size = 5, block_separator = "wait", end= "echo DONE!", append = True)	
	

	
	##############################################
	# GOLD STANDARD (for classifier):
	#
	# smallish series (200 knowledge instances) LOO [default 5] = default 4 with sigWeightStrat bow
	#
	#	sigSize					100 (max)
	#	sigVecStrat				tfidf
	#	sigWeightStrat			tfidf
	#	distance_function		euclidean
	#	vector_normalization	histogram
	#	min_df					0.0
	#	max_df					1.0
	#	WEmodel					pretrained
	#
	##############################################
	
	make_script("python ./classifier.py ../../data/experiments/IMDB_supervised_smallish/IMDB_supervised_smallish_{iterator}	'' ../../data/we_models/wiki.en.bin  	50	100	tfidf	tfidf	../../data/logs/benchmark_classifier_logs/gold_standard_smallish_{iterator}_classifier.log	../../data/corpora/IMDB/train/pos	../../data/corpora/IMDB/train/neg	--tfidf_triple_name ''	--n_neighbors_list '1;3;5;7;9'	--euclideanWMD 1	--vector_normalization	1	--weight_normalization	histogram	--rest 1 --precalculated_dir ../../data/picklefiles/IMDB_supervised_gold_standard &", number_of_lines = 125, script_path = "../kNN/run_classifier_gold_standard.sh", block_size = 1, block_separator = "wait", end= "echo DONE!")	

	
	
	
	# large 
	#make_script("python ./pipeline.py ../data/experiments/IMDB_supervised_large_{iterator} ../data/we_models/test_we.bin  50  50  tfidf  tfidf  ../data/logs/results.log --pickle 1  --reset 1 --min_df 0.0 --max_df 1.0 --weight_normalization histogram &", number_of_lines = 3, script_path = "large.sh", block_size = 3, block_separator = "wait", end= "echo DONE!")
	
	return

if __name__ == "__main__":
	main()