# This Readme covers the Meta-Pipeline

INSERT: A Data Set of n distinct documents (short documents preferably e.g. sentences)

# 1. Label the n documents [emoticon --> positive/negative mood for example] : only if the data is not labeled/semi-labeled
# 2. Preprocessing:
#     2a. Drop stop words
#     2b. Transform to n BoWs
# 3. Apply a word embedding in order to extract n Bag-of-vectors
# 4. Find/Appy weights (default: uniform)
# 5. Apply the WMD to the (n-1)/2*(n-2) Bag-of-Vector-pairs --> extract a (nxn) similarity matrix
# 6. Define a kNN classifier that labels unlabeled (e.g. unlabeled hold out set) according to the majority of either negative/positive emotions
# 7. Calculate the accuracy




