import argparse, os
import sys

sys.path.append('..')

from util.utilities import safe_division

# evaluate classifier log [currently only for binary classfication]
def evaluate_classifier_log(classifier_log, n_neighbors, positive_dir, negative_dir):
	line_counter = 0
	true_counter = 0
	
	
	# for every n_neighbors in n_neighbors_list
	# for every line in the classifier_log
	with open(classifier_log, mode = "r") as f:
		# read the line, and check if the majority of the votes is in favor of the true label
		for line in f:

			line_counter += 1
			
			line_elements = line.split()
			file_name = line_elements[0]
			sorted_predictions = line_elements[1:n_neighbors+1]
			
			majority_count = 0
			# for all distinct labels
			for label_type in set(sorted_predictions):
				# count the occurrences of the label
				label_count = sorted_predictions.count(label_type)
				# if it is greater than the previous majority
				if  label_count > majority_count:
					# switch label_count
					majority_count = label_count
					# and label_type of the majority
					majority_label = label_type
			
			# retrieve the true label
			if os.path.isfile(os.path.join(positive_dir,file_name + ".txt")):
				true_label = "1"
			elif os.path.isfile(os.path.join(negative_dir,file_name + ".txt")):
				true_label = "0"
			else:
				true_label = "-1"
				print("ERROR, unknown label!")
			
			# if prediction is correct, increase the true_counter by 1
			if majority_label == true_label:
				true_counter += 1
							
	accuracy = safe_division(true_counter,line_counter)
	
	return accuracy
			




def main():
	# PARSE ARGS
	#############
	
	parser = argparse.ArgumentParser()
	parser.add_argument("classifier_log", help="path to the classifier_log generated by classifier.py") 	
	parser.add_argument("n_neighbors_list", help="semi-colon separated n_neighbors to use for evaluation, n_neighbors should not be higher than the recorded labels in the classifier_log") 	
	parser.add_argument("positive_dir", help="path to the directory holding the positive documents") 	
	parser.add_argument("negative_dir", help="path to the directory holding the negative documents") 	
	
	args = parser.parse_args()
	

	classifier_log		= args.classifier_log
	n_neighbors_list	= args.n_neighbors_list
	positive_dir		= args.negative_dir
	negative_dir		= args.positive_dir
	
	# extract n_neighbors_list
	if n_neighbors_list == None:
		n_neighbors_list = [1,3,5,7,9]
	else:
		n_neighbors_list = [int(el) for el in n_neighbors_list.split(";")]
	
	
	
	# evaluate the accuracy
	for n_neighbors in n_neighbors_list:
		accuracy 		= evaluate_classifier_log(classifier_log, n_neighbors, positive_dir, negative_dir)
		print("The accuracy for n_neighbors = {} is {}".format(n_neighbors, accuracy))
	
if __name__ == "__main__":
	main()













