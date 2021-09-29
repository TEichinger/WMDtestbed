Welcome to the **WMDtestbed**. WMDtestbed is a Python3 library and a testbed for experimenting with classification and analysis of
pairwise distances between text documents with the aid of Word Mover's Distance (WMD). The WMD has been introduced byas per [Kusner et al. (2015)](https://github.com/mkusner/wmd).
 
WMDtestbed has been successfully tested on 

* Python 3.5 on Ubuntu 18.04
* Python 3.8 on Ubuntu 20.04

# Deployment Instructions
The deployment of the library requires the following (minimal steps).

### Clone the Repository
```
git clone https://github.com/moxplayer/WMDtestbed.git
cd WMDtestbed
```

### Install `pip` and setup a virtual environment with `virtualenv`
```
sudo apt install python3-pip
sudo python3 -m pip install virtualenv
sudo python3 -m virtualenv WMDtestbed_venv
source WMDtestbed_venv/bin/activate
```

### You can now install the required Python packages into the environment
```
python3 -m pip install numpy
python3 -m pip install nltk
python3 -m pip install gensim
python3 -m pip install pandas
python3 -m pip install sklearn
python3 -m pip install fasttext
python3 -m pip install yake
python3 -m pip install rake-nltk
python3 -m pip install matplotlib
python3 -m pip install pyemd
python3 -m pip install wmd
python3 -m pip install regex
```

### Add File Structure
WMDtestbed foresees a certain file structure. First of all, we separate code from data. `/src` contains scripts and Python modules.
`/data` contains data such as text files/corpora, test batches, picklefiles (persisted WMD signatures of text documents), plots,
similarityfiles, word embedding models, and last but not least (evaluation) logs.
```
mkdir data/corpora -p
mkdir data/experiments/test_experiment -p
mkdir data/picklefiles -p
mkdir data/plots -p
mkdir data/similarityfiles -p
mkdir data/we_models -p
mkdir data/logs -p
```

Your directory structure should now look like this:
![Alt text](/README_screenshots/data_directory_init.png?raw=true "data directory")

### Download Test Data

A pretrained *fasttext* word embedding file is necessary. You can download an English word embedding *binary* (!) file following these [instructions](https://fasttext.cc/docs/en/crawl-vectors.html).
Move the word embedding file into the 'data/we_models' folder. Alternatively to a pre-trained model, you may also choose to train a custom model.
Alternatively, an English benchmark word embedding binary model can be downloaded via Python as follows:

```
python
>>> import fasttext.util
>>> fasttext.util.download_model('en', if_exists='ignore')
>>> ft = fasttext.load_model('cc.en.300.bin')
>>> quit()
mv cc.en.300.bin data/we_models
```

### Setup a Test Experiment

The library defines experiments via directories, where an experiment can be text classification, or the clustering of text documents via pairwise similarity comparisons.

### Example Experiment

An example experiment consisting of a binary classification task using twitter data from republican and democratic senators can be downloaded via the following link: https://tubcloud.tu-berlin.de/s/SxDGfyH5W7cQDki (pw: senator_test)
Extract the files inside the compressed folder into the just created 'data/experiments/test_experiment' folder so that this folder now contains all the .txt files and the label file (which we will specify as a starting parameter later) for this experiment.
The second experiment based on the 20 Newsgroups data can be downloaded via the following link: https://tubcloud.tu-berlin.de/s/4dtR6JJQCb92aTs (pw: 20NG_test)

After downloading the test experiment your directory should look like this:
![Alt text](/README_screenshots/test_experiment_directory.png?raw=true "test_experiment directory")

### Label Directories

A **label directory** is a directory that contains text (.txt) files with a distinct name of a **single class/label**. That name is usually the author in case the
 text has a (unique) author. The directory will be used later for classification/labeling. If the text files do not have a label, then we may also skip the definition of label directories.

### Experiment Directories

An **experiment directory** is a directory that contains text (.txt) files with a distinct name, yet the files do not need to be of a single class/label. 
With the files in the experiment directory, one can now perform:
	* classification (
	* text clustering (

In order to use the files inside the experiment directory for a classification task, the usage of a **label file** is possible. A label file is a text file that maps every input file to its class/label. The labe file has to be given as a starting parameter to the program. The above mentioned example experiment includes a label file (conveniently named "senator_twitter_account_names_115th_congress_20180926").

### Run a Test Experiment

You can now test your setup by running the test experiment. Therefore, activate the *run_pipeline.sh* script and run from the *./src directory*.

``` 
cd src
chmod +x run_pipeline.sh
./run_pipeline.sh
```

This should prompt you the following results:

```
Load WE model...
Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.
senator_twitter_account_names_115th_congress_20180926Done!
Collected all pickle pathsd.
No precalculated edge graph has been specified.
Calculate 1770.0 pairwise distances.
Calculated 1770/1770(100)% of all similarities
Wrote a file with pairwise similarities
Save edge graph at ../data/similarityfiles/2020_08_11_12_32_35_test_experiment_sims.csv.
Load edge graph as similarity matrix
The experiment is supervised. We can thus calculate a leave-one-out accuracy directly.
n_neighbors: 1
accuracy: 0.7833333333333333
Leave-one-out Accuracy logged
n_neighbors: 3
accuracy: 0.8
Leave-one-out Accuracy logged
n_neighbors: 5
accuracy: 0.8
Leave-one-out Accuracy logged
n_neighbors: 7
accuracy: 0.8166666666666667
Leave-one-out Accuracy logged
n_neighbors: 9
accuracy: 0.8666666666666667
Leave-one-out Accuracy logged
n_neighbors: 11
accuracy: 0.8666666666666667
Leave-one-out Accuracy logged
n_neighbors: 13
accuracy: 0.8833333333333333
Leave-one-out Accuracy logged
PIPELINE FINISHED!
```

### Get the Documentation of the Program Parameters

If you want to change the program parameters but don't know yet what they are supposed to do, you can run the help script. To do so, activate *run_help.sh* script and run from the *./src directory*.

``` 
chmod +x run_help.sh
./run_help.sh
```
	
### Feedback & Contact

Let me know if you have any questions at tobias.eichinger AT tu-berlin DOT de. If you use the code, please cite using the following BibTeX entry: 

    @inproceedings{eichinger2019,
    author={T. {Eichinger} and F. {Beierle} and S. U. {Khan} and R. {Middelanis}},
    booktitle={ICC 2019 - 2019 IEEE International Conference on Communications (ICC)},
    title={Affinity: A System for Latent User Similarity Comparison on Texting Data},
    year={2019},
    doi={10.1109/ICC.2019.8761051},
    ISSN={1550-3607},
    }

