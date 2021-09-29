import sys, os
sys.path.insert(0, "../")
from util.utilities import load_edge_graph

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def simfile2hist(edge_graph_path, output_dir):
	""" Create the information of similarities condensed in an edge-graph format similarity file into a histogram distribution of
	similarities over the entire population.
	requires:
		[edge_graph_path]: <str> path to the edge graph
	"""
	# load edge graph as pandas dataframe
	edge_graph 	= load_edge_graph(edge_graph_path)
	
	
	plt.title("Similarity-Histogram of {}".format(edge_graph_path))
	plt.xlim([-2,1])
	# convert similarities to histogram
	similarities = edge_graph['weight'].hist(bins = 100)
	
	
	# this/is/a/path/edge_graph.csv --> edge_graph.csv
	file_name   = os.path.splitext(os.path.split(edge_graph_path)[1])[0] + '.svg'
	output_path = os.path.join(output_dir, file_name) 
	# save histogram plot
	plt.savefig(output_path, bbox_inches = 'tight')
	
	return

def get_correlation_of_histogram_and_sim_file(histogram_file_path, similarity_file_path):
	"""A histogram file contains the similarities of a single reference user to a list of other users. A similarity file contains
	arbitrary similarities between source and target user. The goal is now to calculate for the reference user specified in the histogram file,
	to calculate the correlation to the similarities in the similarity file, in the hope that every similarity has an analogue in the similarity file.
	requires:
		[histogram_file_path]: <str> path to the histogram file. The histogram file should contain indices of IDs and a column containing the reference ID
		[similarity_file_path]:	<str> path to the similarity file
	"""
	# read histogram file
	histogram_file	= pd.read_csv(histogram_file_path, index_col = 0)
	#extract reference user ID : "spec_reviewerID" --> reviewerID
	ref_user_ID		= histogram_file.columns[0].split("_")[-1]
	# read similarity file
	similarity_file = pd.read_csv(similarity_file_path, index_col = "target")#load_edgegraph(similarity_file_path, as_simmat = False)
	# filter edges that do not contain ref_user_ID in neither column
	# TO BE coded! For the simple case of similarity files with source = ref_user_ID for all rows, this works
	#similarity_file.index = similarity_file['target']

	# concatenate both column of "similarity" and "weight" along the source index
	# shape (n,)
	sims1 = similarity_file.loc[:,"weight"]
	sims2 = histogram_file.loc[:,histogram_file.columns[0]]
	aligned_df = pd.concat([sims1,sims2], axis = 1, sort = True)
	C = aligned_df.corr(method="pearson")
	correlation_coefficient = C.iloc[0,1]
	#print(sims1.columns)
	# check now whether the top cosine (rating) similar users are also similar in the WMD case
	sorted_sims1 = sims1.sort_values(axis=0, ascending = False)
	sorted_sims2 = sims2.sort_values(axis=0, ascending = False)
	sorted_indices1 = list(sorted_sims1.index)
	sorted_indices2 = list(sorted_sims2.index)
	# 
	k = 25
	top_histo_indices = sorted_indices1[:k]
	top_sim_indices = [sorted_indices2.index(el) for el in top_histo_indices]
	print(top_sim_indices)
	
	
	return correlation_coefficient


# find the closest n neighbors of a certain node in a similarity matrix
# input :
# - n_neighbors : number of neighbors
# - simmat      : similarity matrix (symmetric) with similarities between nodes i,j in simmat[i,j]
# - node_index  : the index in the list of row_indices
# e.g. node_index = 2; simmat.index = [1,5,7] --> the node to be considered is named 7
# output:	[nearest_neighbors]   : <list> of <int> [n_neighbors] node indices
#			[nearest_similarities]: <list> of <floats> the [n_neighbros] highest similarities of the node with the respective [node_index]
def findNN(n_neighbors, node_index, simmat):
	minus_infty = -100000.0
	# pick the similarities
	node_row = list(simmat.loc[simmat.index[node_index], :])
	# remove self-similarity
	node_row[node_index] = minus_infty
	nearest_similarities = []
	nearest_neighbors = []
	for i in range(n_neighbors):
		max_sim 				= max(node_row)
		nearest_neighbor_index  = node_row.index(max_sim)
		nearest_similarities.append(max_sim)
		nearest_neighbors.append(nearest_neighbor_index)
		node_row[nearest_neighbor_index] = minus_infty
	return nearest_neighbors , nearest_similarities   

# transform similarities, cf. method 'to_similarity' for the analogue for dissimilarities
# this is only of interest for plotting! since the relative distances are not affected and
# consequently the kNN are the same for every node
def transform_weights(x):
	spread_exponent = 2
	result = x**spread_exponent# math.sqrt(1-x)# for negative exponents (e.g. -0.5)
	#result = int(round(result * 100000,0))
	return result

# submit a similarity matrix 'simmat', 'knn_labels', and 'n_neighbors' as the number of neighbors to vote on
# currently, only two distinct labels are recognized
# [simmat]		: pandas dataframe, matrix with file_index and column and their respective similarity
# [knn_labels]	: <list> of labels infered from the label_df, the knn_labels have to be in the same order as simmat index! (corresponding files)
# [n_neighbors] : <int> number of neighbors to consider for inference
# [verbose]		: <bool>, show more details if True
def evaluate_via_simmat(simmat, knn_labels, n_neighbors = 3, verbose = False):
	#verbose = True
	# count all correctly predicted classes
	correct_predictions_counter = 0
	# for all nodes given in the simmat
	for i in range(len(simmat.index)):
		# find the node's n_neighbors
		nearest_neighbors, nearest_sims = findNN(n_neighbors, i, simmat)
		# get their corresponding class labels
		class_labels = []
		for neighbor in nearest_neighbors:
			class_labels.append(knn_labels[neighbor])

		if verbose:
			print("True Label: {}".format(knn_labels[i]))
			print("Nearest Class labels: {}".format(class_labels))
		# find the node's majority class_label
		######################
		majority_class_label = 0
		num_labels           = 0
		# for all class_labels
		for label in set(class_labels):
			# switch the majority class label and the number of labels if a label has more votes
			if class_labels.count(label)> num_labels:
				num_labels           = class_labels.count(label)
				majority_class_label = label
		
		# if the prediction is correct, increase the correct_predictions_counter
		if knn_labels[i] == majority_class_label:
			correct_predictions_counter += 1
	accuracy = correct_predictions_counter/len(knn_labels)*1.0
	print('n_neighbors: {}'.format(n_neighbors))
	print('accuracy: {}'.format(accuracy))
	return accuracy  

	
# evaluate the accuracy (leave-one-out-approach) via an edge_graph table instead of a similarity matrix
# requires:	- edge_graph [source, target, weight]    [<pd.DataFrame>]
#			- label_file                             [<pd.DataFrame>]
#			- n_neighbors (number of nearest neighbors to consider
def evaluate_via_edgegraph(edge_graph, label_df, n_neighbors = 3):
	majority_votes = []
	n_neighbors = int(n_neighbors)
	# 1. extract all nodes from the label_file
	unq_nodes = label_df['label'].index
	labels = list(label_df['label'])
	# 2. for all nodes:
	for N in unq_nodes:
		# 2a. filter out edges that do not contain N as a node
		edge_graph_N = edge_graph[(edge_graph['source'] == N) | (edge_graph['target'] == N)]
		# 2b. Find k nearest neighbors <list> of <str>s
		kNN = find_kNN(edge_graph_N, N, n_neighbors)

		# 2c. Find the k corresponding labels
		kNN_labels = [find_label(node, label_df) for node in kNN]

		# 2d. Pick the majority label
		majority_vote = find_majority(kNN_labels)

		# 2e. Append majority to list
		majority_votes.append(majority_vote)
	
	correct_votes = 0
	for true, majority_vote in zip(labels, majority_votes):
		if true == majority_vote:
			correct_votes += 1
	
	leave_one_out_accuracy = correct_votes/len(labels)	
	return leave_one_out_accuracy

# extract all unique nodes from an edge_graph (pd.DataFrame) as a <list>
def extract_all_nodes_edge_graph(edge_graph):
	sources = edge_graph['source'].drop_duplicates()
	targets = edge_graph['target'].drop_duplicates()
	unq_nodes = list(pd.concat([sources,targets]).drop_duplicates())
	return unq_nodes
	

	
# given an edge of an edge graph (pandas.Series)
# pick the node that is not N
def pick_not_N(edge, N):
	if (edge['source'] == N) & (edge['target'] != N):
		return edge['target']
	elif (edge['source'] != N) & (edge['target'] == N):
		return edge['source']
	else:
		return "ERROR"


# extract the nearest neighbors from an edgegraph
# that contains similarities of 'N' to every other node
# [edge_graph] : pd.DataFrame; edge graph
# [N] : <str> node name to find k nearest neighbors for
# [k] : number of nearest neighbors
def find_kNN(edge_graph, N, k):
	nearest_neighbors = []
	# sort the edges by (descending) similarities
	edge_graph = edge_graph.sort_values('weight', ascending = False)
	# find the k_th index for slicing
	k_th_index = edge_graph.index[k-1]
	# pick the edges containing N's nearest neighbors (highest similarity)
	top_k = edge_graph.loc[:k_th_index,:]
	# for every nearest neighbor edge
	for i in top_k.index:
		# pick the nearest neighbor node
		nearest_neighbor = pick_not_N(top_k.loc[i,:], N)
		# append to the result list
		nearest_neighbors.append(nearest_neighbor)
	return nearest_neighbors

# lookup the label of 'node' in the 'label_df' pandas.DataFrame
def find_label(node, label_df):
	return label_df.loc[node,'label']

# input an edge graph, label file, and filter out all edges that include the 'filter_labels'
def filter_out_labels(edge_graph, filter_labels):
	# from 'filter_labels' df, find the nodes that correspond to those labels
	filter_nodes = filter_labels.index
	# find the indices that contain at least one node in 'filter_nodes'
	filtered_indices = [index for index in edge_graph.index if (edge_graph.loc[index,'source'] not in filter_nodes) | (edge_graph.loc[index,'target'] not in filter_nodes)]
	# select only those edges (rows)
	filtered_edge_graph = edge_graph.loc[filtered_indices,:]
	return filtered_edge_graph
	
# return the most frequent label in 'label_list'
def find_majority(label_list):
	unq_labels = set(label_list)
	majority_count = 0
	majority_label = None
	for unq_label in unq_labels:
		unq_label_count = label_list.count(unq_label)
		if  unq_label_count > majority_count:
			majority_count = unq_label_count 
			majority_label = unq_label
	return majority_label
		
	

	



	


def main():
	#edge_graph_path = sys.argv[1]
	#label_path 		= sys.argv[2]
	#n_neighbors 	= sys.argv[3]
	
	output_dir = "../../data/plots/histograms"
	# this was with cosine in fact..
	#edge_graph_path = "../../data/similarityfiles/2019_07_02_14_23_51_A2SUAM1J3GNN3B_[yake-1n]WMD_of_nonzero_cosine_sims.csv"
	# now with euclidean
	#edge_graph_path = "../../data/similarityfiles/2019_07_02_15_48_46_A2SUAM1J3GNN3B_[yake-1n]WMD_of_nonzero_cosine_sims.csv"
	#edge_graph_path = "../../data/similarityfiles/2019_07_02_16_07_59_A2SUAM1J3GNN3B_[yake-1n]WMD_of_1000sample_sims.csv"
	#simfile2hist(edge_graph_path, output_dir)


	histogram_file_path  = "../../data/corpora/amazon_kcore/histogram_data/cosine_A2SUAM1J3GNN3B.csv"
	similarity_file_path = "../../data/similarityfiles/2019_07_02_15_48_46_A2SUAM1J3GNN3B_[yake-1n]WMD_of_nonzero_cosine_sims.csv"
	get_correlation_of_histogram_and_sim_file(histogram_file_path, similarity_file_path)
	
	
	"""
	# Load edge_graph
	print('Load edge graph {}.'.format(edge_graph_path))
	edge_graph 	= load_edge_graph(edge_graph_path)
	# Load labels
	label_df 	= load_labels(label_path)
	print('Load labels {}.'.format(label_path))
	# Evaluate via edge_graph
	print('Calculate accuracy via edge graph.')
	accuracy  	= evaluate_via_edgegraph(edge_graph, label_df, n_neighbors = n_neighbors)
	print('The accuracy is {}.'.format(str(accuracy)))
	return accuracy
	"""
if __name__ == "__main__":
	main()



