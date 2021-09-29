# this python file is meant to provide transduction functionality
# 1. Load labels (generated via 'setup_experiment.sh')
# 2. Load similarity file [edge graph format] (generated via '')
# 3. Split the similarity file into [labeled_sim] and [unlabeled_sim]
# 4. For every unknown label:
# 		- search the nearest neighbors in [labeled_sim] --> majority vote
# 		- label it
# 5. Save the assigned labels
# 6. Evaluate the labels


import sys, os
sys.path.insert(0, "../")
from util.utilities import load_labels, load_edge_graph, get_current_timestamp
from evaluation.evaluate import pick_not_N, find_kNN, find_label, filter_out_labels, find_majority

def main():
	label_path      = sys.argv[1]
	edge_graph_path = sys.argv[2]
	k               = sys.argv[3]

	# e.g. label_path = ../../data/experiments/IMDB_semisupervised_small/label_file
	# --> experiment_name = IMDB_semisupervised_small
	experiment_name = label_path.split('/')[-2]

	majorities = []
	
	# 1. Load labels (generated via 'setup_experiment.sh')
	labels = load_labels(label_path)
	
	# 2. Load similarity file [edge graph format] (generated via 'wmd/calcPairwiseDist')
	edge_graph = load_edge_graph(edge_graph_path)
	# split labels with unknown (-1) label [to be transducted]
	unknown_labels = labels[labels['label'] == -1]
	# extract a list of known_labels (in {0,1})
	known_labels = list(labels[labels['label'] != -1].index)
	
	# 3. For every node N in 'label_file' with an unknown label ('-1'):
	for N in unknown_labels.index:
		# - pick edges with either source OR target == N as a pandas subdataframe of 'edge_graph' --> [edge_graph_N]
		edge_graph_N = edge_graph[(edge_graph['source'] == N) | (edge_graph['target'] == N)]
		# filter out edges that contain nodes with unknown label (-1)
		filtered_edge_graph_N = filter_out_labels(edge_graph_N, unknown_labels)
		# - find the k nearest neighbors (highest similarity) in 'N_subframe' --> [kNN] as a <list> of node names
		kNN = find_kNN(filtered_edge_graph_N, N, 3)
		# - replace every node name in 'kNN' with its label
		kNN_labels = [find_label(node, labels) for node in kNN]
		# - pick the majority label
		majority = find_majority(kNN_labels)
		# - append to result data frame
		majorities.append(majority)
	# 5. Save the assigned labels (result data frame)
	output_filename = get_current_timestamp() + '_transducted_label_file_' + experiment_name
	with open(output_filename, mode = 'w') as f:
		for unknown_label, majority_vote in zip(unknown_labels.index, majorities):
			f.write(unknown_label + ' ' + str(majority_vote) + '\n')
	
	# show the transducted reviews and their assigned 'polarity' (pos/neg)
	print(list(zip(unknown_labels.index,majorities)))
	
	
	

if __name__ == "__main__":
	main()

