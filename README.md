WMDtestbed is a Python3 testbed for experimenting with the Word Mover's Distance (WMD) by [Kusner et al. (2015)](https://github.com/mkusner/wmd).

You can ...
1. ... calculate pairwise distances between a set of text documents,
1. ... run keyword extraction to speed up similarity calculation,
1. ... perform and evaluate k-nearest-neighbor classification on the basis of WMD.

The WMDtestbed has been successfully tested on

* Python 3.5 on Ubuntu 18.04
* Python 3.8 on Ubuntu 20.04
* Python 3.9 on MacOS 11.6

# 1. Environment Setup

### 1.1 Clone the Repository

```
git clone https://github.com/TEichinger/WMDtestbed.git
cd WMDtestbed
```

### 1.2 Install `pip` and setup a virtual environment with `virtualenv`

On Linux/Ubuntu you can use the following commands.

```
sudo apt install python3-pip
sudo python3 -m pip install --upgrade pip
sudo python3 -m pip install virtualenv
sudo python3 -m virtualenv WMDtestbed_venv
source WMDtestbed_venv/bin/activate
```

### 1.3 Install Python packages into the virtual environment

The installation is done line by line from the `./requirements.txt` file.
```
cat requirements.txt | xargs -n 1 sudo -H python3 -m pip install
```

# 2. Demo

You can run a demo by running the following commands:
```
chmod +x ./demo.sh
./demo.sh
```

# 3. Experiment Setup

An experiment is a full run of the `./src/pipeline.py` Python3 script.

### 3.1 Experiment Requirements

An experiment requires:

1. an experiment folder containing text (.txt) files such as `./data/experiments/test_experiment`
1. a word embedding model such as `./data/we_models/cc.en.300.bin` downloaded by the demo script (see Section 2).

and takes a label file as optional input such as `./data/experiments/test_experiment/test_experiment_labels`.


### 3.2 Experiment Procedure

1. Drop stop words in every text file
1. Run keyword extraction on every text file
1. Apply a word embedding model to all words in a text file to derive a signature (list of weighted word vectors).
1. Calculate pairwise WMD distances
1. Transform distances into similarities

### 3.3 Pickled signatures and similarity files

The pipeline generally uses serialized (pickled[https://pypi.org/project/pickle5/]) version of signatures (`./data/picklefiles`) and similarity files (`./data/similarityfiles`). You can inspect these subresults by running the following commands on a test pickle file `test.pk`):

```
python3
import pickle
with open("./data/picklefiles/2021_09_29_18_34_04/atheism_1.pk", "rb") as f:
  d = pickle.read(f)
d
quit()
```


### 3.4 Get the Documentation of the Program Parameters

Run the following command to get further help on correctly setting the parameters.

```
python3 ./src/pipeline.py	--help
```



### Feedback & Contact

Let me know if you have any questions at tobias.eichinger AT tu-berlin DOT de. If you use the code, please cite one of the following papers:

    @inproceedings{eichinger2021,
      author = {Eichinger, Tobias},
      title = {Reviews Are Gold!? On the Link between Item Reviews and Item Preferences},
      year = {2021},
      booktitle = {Proceedings of the KaRS & ComplexRec 2021 Joint Workshop at ACM RecSys 2021},
      publisher = {CEUR-WS},
      note = {Accepted for publication},
    }

    @inproceedings{eichinger2019,
    author={T. {Eichinger} and F. {Beierle} and S. U. {Khan} and R. {Middelanis}},
    booktitle={ICC 2019 - 2019 IEEE International Conference on Communications (ICC)},
    title={Affinity: A System for Latent User Similarity Comparison on Texting Data},
    year={2019},
    doi={10.1109/ICC.2019.8761051},
    ISSN={1550-3607},
    }
