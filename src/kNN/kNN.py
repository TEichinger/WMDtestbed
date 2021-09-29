# insert x and y values of the embedding (2-dim projection), plus the number of neighbors to consider (default = 3)
# ignore the value to be predicted
# leave-one-out accuary!
def evaluate_embedding(X, Y, knn_labels, n_neighbors=3):
    from sklearn.neighbors import KNeighborsClassifier

    # evaluate the separability by taking the overall accuracy over all predictions
    wrong_counter = 0
    correct_counter = 0
    # only pick Republican/Democrat senators by leaving out SenSanders, FLOTUS and realDonaldTrump
    base_set = list(zip(X, Y))

    for i in range(len(base_set)):
        neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
        train_set = base_set[:i] + base_set[i + 1:]
        train_labels = knn_labels[:i] + knn_labels[i + 1:]
        neigh.fit(train_set, train_labels)

        # get predictions
        pred = neigh.predict([base_set[i]])
        label = knn_labels[i]

        if pred == label:
            correct_counter += 1
        else:
            wrong_counter += 1

    print('Correctly classified:', correct_counter)
    print('Incorrectly classified:', wrong_counter)
    accuracy = correct_counter / (wrong_counter + correct_counter)
    print('Accuracy:', accuracy)
    return accuracy


# throws out the elements at the indices of the prep_list
# D./M.Trump, and both independent senators --> (0,1,2,99)
def prep_sample(prep_list, indices):
    result = list(prep_list)
    for i in sorted(indices, reverse=True):
        result.pop(i)
    return result


# infer label from distances_sorted_withlabels
# for a certain signature, and knowledge points, input a list of similarities (anti-distances)
# to nearest neighbors and their associated labels
# output the majority vote for a given 'n_neighbors' neighbors to consider for inference
# [distances_sorted_withlabels]	: <list> of <list>s, e.g. [['file1', 1], ['file2', 1], ...]
# [n_neighbors]					: <int>, specify the number of nearest neighbors to consider for inference
def infer_label_from_distances(distances_sorted_withlabels, n_neighbors):
    top_n_neighbors = distances_sorted_withlabels[:n_neighbors]
    top_labels = [el[1] for el in top_n_neighbors]
    # cast a majority vote
    majority_vote, majority_support = find_majority(top_labels)
    return majority_vote


# find that majority (most frequent element) out of a <list> of elements
# pick the first, if tied
def find_majority(votes):
    unq_votes = tuple(set(votes))
    majority_vote = unq_votes[0]
    majority_vote_support = votes.count(majority_vote)
    for unq_vote in unq_votes:
        if votes.count(unq_vote) > majority_vote_support:
            majority_vote = unq_vote
            majority_vote_support = votes.count(unq_vote)
    return majority_vote, majority_vote_support


# find the closest n neighbors of a certain node
# input :
# - similarity matrix (symmetric) with similarities between nodes i,j in simmat[i,j]
# - n : number of neighbors
# - node_index: the index in the list of row_indices
# e.g. node_index = 2; simmat.index = [1,5,7] --> the node to be considered is named 7
# output: node indices
def findNN(n_neighbors, node_index, simmat):
    # pick the similarities
    node_row = list(simmat.loc[simmat.index[node_index], :])
    # remove sel-similarity
    node_row[node_index] = -1.0
    nearest_similarities = []
    nearest_neighbors = []
    for i in range(n_neighbors):
        max_sim = max(node_row)
        nearest_neighbor_index = node_row.index(max_sim)
        nearest_similarities.append(max_sim)
        nearest_neighbors.append(nearest_neighbor_index)
        node_row[nearest_neighbor_index] = -1.0

    return nearest_neighbors, nearest_similarities


# transform similarities, cf. method 'to_similarity' for the analogue for dissimilarities
def transform_weights(x):
    spread_exponent = 2
    result = x ** spread_exponent  # math.sqrt(1-x)# for negative exponents (e.g. -0.5)
    # result = int(round(result * 100000,0))
    return result


# submit a similarity matrix, knn_labels, and n_neighbors as the number of neighbors to vote on
# currently, only two distinct labels are recognized
def evaluate_embedding_raw(simmat, knn_labels, n_neighbors=3):
    # count all correctly predicted classes
    correct_predictions_counter = 0
    # for all nodes give in the simmat
    for i in range(len(simmat.index)):
        # find the node's n_neighbors
        nearest_neighbors, _ = findNN(n_neighbors, i, simmat)
        # get their corresponding class labels
        class_labels = []
        for neighbor in nearest_neighbors:
            class_labels.append(knn_labels[neighbor])

        # find the node's majority class_label
        ######################
        majority_class_label = 0
        num_labels = 0
        # for all class_labels
        for label in set(class_labels):
            # switch the majority class label and the number of labels if a label has more votes
            if class_labels.count(label) > num_labels:
                num_labels = class_labels.count(label)
                majority_class_label = label

        # if the prediction is correct, increase the correct_predictions_counter
        if knn_labels[i] == majority_class_label:
            correct_predictions_counter += 1
    accuracy = correct_predictions_counter / len(knn_labels)
    print('accuracy:', accuracy)
    return accuracy
