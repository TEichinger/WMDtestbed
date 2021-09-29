import math
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_edgegraph(filepath, as_simmat = True, output_type = "gephi"):
	""" Load an edgegraph at <filepath>. The edgegraph is a .csv file of the format:

		source, target, weight
		s1, t1, w1,
		s2, t2, w2,
		...

		The column names 'source', 'target', and 'weight' are not fixed.
		The function takes the first 3 columns and ignores all further columns.

	"""
	# load csv as a pandas dataframe
	df = pd.read_csv(filepath)

	if as_simmat:
		source_col, target_col, weight_col = df.columns[:3]
		# extract all nodes
		nodes = pd.concat([df[source_col], df[target_col]], axis = 0).drop_duplicates() #['source'], df['target']], axis = 0).drop_duplicates()
		# initialize a similarit matrix
		sim_mat = pd.DataFrame(0, index = nodes, columns = nodes)
		# fill the similarity matrix
		for i in df.index:
			source, target, weight = df.loc[i, [source_col, target_col, weight_col]].tolist()#list(df.loc[i, ['source', 'target', 'weight']])
			# set upper triangle
			sim_mat.loc[source, target] = weight
			# set lower triangle
			sim_mat.loc[target, source] = weight

		# set diagonal
		for i in nodes:
			sim_mat.loc[i,i] = 1.000

		return sim_mat
	else:
		return df


# transform a dissimilarity in [0,1] to one in [0,1]
def to_similarity(x):
	spread_exponent = 2

	result = (1-x)**spread_exponent # math.sqrt(1-x) # for negative exponents (e.g. -0.5)
	result = int(round(result * 10000, 0))
	return result

# input a similarity file (as an edge graph)
# format:
####
#source, target, weight
#0, 1, 0.984
# 0,2, 0.514
# ...

# if from_file == False: sim_file is a similarity matrix
def run_MDS(sim_file, colors = None, names = None, markers = None, marker_sizes = None,\
	 file_name = 'test', random_state = 1, legend = None, from_file = True):
	# load the sklearn MDS library
	from sklearn.manifold import MDS
	print('Run MDS on pairwise distances..')

	# load the edge graph
	print('Load edge graph..')
	if from_file:
		sim_mat = load_edgegraph(sim_file)
	else:
		sim_mat = sim_file
	#prepare for MDS input
	print('Convert to dissimilarities..')
	MDS_input = sim_mat.applymap(to_similarity)

	# initialize an MDS object and set dissimilarity == 'precomputed' to specify that the input is a dissimilarity matrix
	print('Run MDS..')
	embedding = MDS(n_components=2, dissimilarity='precomputed', random_state = 5071)

	result = embedding.fit_transform(MDS_input)
	x = [el[0] for el in result]
	y = [el[1] for el in result]
	print('Draw Plot..')
	draw_scatter(x,y,colors,names,markers,marker_sizes, file_name = file_name, legend = legend)

	return x,y


# pass a list of labels and file paths in order to return a list of corresponding labels
# also add a corresponding ['legend']
# 'label_df' format:
# file_name1 1
# file_name2 0
# ...
# --> filename1 is a positive review
# --> filename2 is a negative review
# color_palette : <dict> of label keys and their corresponding color values
def make_colors(label_df, pickle_names, color_palette = None):
	""" For a label dataframe, and a corresponding list of pickle_names, define a palette of colors (for each label one)
	and return a list of colors representing each and every file's label in label_df. If a color palette has been specified
	take the latter, else randomly sample RGB colors.
		[label_df]		: pandas dataframes with file_names (w/o file extensions) as index and column label with string type labels
		[pickle_names]	: <list> of <str>s containing the file_names/pickle_names
		[color_palette]	: <list> of <str> RGB color codes to use for color coding the labels, if the color_palette is not long enough, randomly sample colors
	"""
	colors = []

	# extract unique labels from the label file
	unq_labels = list(label_df['label'].drop_duplicates())

	# map them to colors; if the color_palette is not long enough, there will not be a designated error message thrown!
	color_map = dict()

	# if not a color_palette is specified, or the latter is too short to map the unq_labels injectively
	if (not color_palette) | (not (len(color_palette) >= len(unq_labels))):
		import random
		r = lambda: random.randint(0,255)
		# random color_palette
		color_palette = ['#%02X%02X%02X' % (r(),r(),r()) for _ in range(len(unq_labels))]

	# for all unique labels
	for i,unq_label in enumerate(unq_labels,0):
		# map the unique label to the i-th element of the color_palette
		color_map[unq_label] = color_palette[i]

	for pickle_name in pickle_names:
		file_label = label_df.loc[pickle_name,'label']
		file_color = color_map[file_label]
		colors.append(file_color)

	# used colors
	used_colors = set(colors)
	# used_labels
	used_unq_labels = [unq_label for unq_label in unq_labels if color_map[unq_label] in used_colors]

	return used_unq_labels,colors

# draw a scatter plot with x,y values
def draw_scatter(x,y,colors=None,names=None,markers=None,marker_sizes=None, xlim=None, ylim=None,\
                 file_name = 'test', legend = None):

    import matplotlib.patches as mpatches                        # submodule for legends in plots

    # start drawing
    fig, ax = plt.subplots()

    # check if parameters are contributed
    if not colors:
        colors = ['blue' for _ in x]
    if not names:
        names = ['' for _ in x]
    if not markers:
        markers = ['o' for _ in x]
    if not marker_sizes:
        marker_sizes = [10 for _ in x]

    # sort by color
    L = sorted(list(zip(colors, names, markers,marker_sizes,x,y)))
    colors       = [el[0] for el in L]
    names        = [el[1] for el in L]
    markers      = [el[2] for el in L]
    marker_sizes = [el[3] for el in L]
    x = [el[4] for el in L]
    y = [el[5] for el in L]

    # find indices between colors
    color_indices = [i for i in range(len(colors)-1) if colors[i]!= colors[i+1]]
    color_indices.insert(0,-1)
    color_indices.append(len(x)-1)
    #
    if not legend:
        legend = [colors[i] for i in color_indices]
    # draw dots
    for i in range(len(color_indices)-1):
        li = color_indices[i]+1
        ui = color_indices[i+1]+1
        ax.scatter(x[li:ui], y[li:ui], color = colors[li:ui], marker=markers[li], s = marker_sizes[li:ui], label = legend[i])

    # annotate dots
    for i, txt in enumerate(names):
            ax.annotate(txt, (x[i], y[i]))

    if xlim:
        plt.xlim([xlim[0],xlim[1]])
    if ylim:
        plt.ylim([ylim[0], ylim[1]])


    # create legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    # resize axes
    plt.gca().set_aspect('equal', 'datalim')

    # save the output
    plt.savefig(file_name, bbox_inches = 'tight')
    print('Saved the plot.')

    return


# load labels [format: 'file_name_without_extension' 'label'], where 'label' in {0,1,-1}
# the output is a pandas dataframe
def load_labels(label_path, as_list = False):
	labels = pd.read_csv(label_path, names = ['label'], sep = ' ', index_col = [0])
	if as_list:
		labels = labels.loc[:,'label'].tolist()
	return labels




def main():
	import pandas as pd

	label_path = '../../data/corpora/affinity/ref_histories/age_label_file'
	pickle_folder = '../../data/picklefiles/2019_06_07_16_46_19'

	pickle_names = []

	for root, dirs, files in os.walk(pickle_folder):
		for name in files:
			if os.path.splitext(name)[1] == '.pk':
				pickle_names.append(os.path.splitext(name)[0])

	print("pickle_paths", pickle_names)
	label_df = load_labels(label_path, as_list=False)

	colors =  make_colors(label_df, pickle_names, color_palette = ["black", "red", "blue", "green", "yellow", "orange", "magenta", "lightblue","grey", "lightred"])


if __name__ == "__main__":
	main()
