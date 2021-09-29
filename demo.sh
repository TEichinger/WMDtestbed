#! /bin/sh

export ROOT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"


# 1. Add folder structure for data
mkdir -p "${ROOT_DIR}/data/picklefiles"
mkdir -p "${ROOT_DIR}/data/plots"
mkdir -p "${ROOT_DIR}/data/similarityfiles"
mkdir -p "${ROOT_DIR}/data/we_models"
mkdir -p "${ROOT_DIR}/data/logs"


# 2. Download binary fasttext word embedding (we) model
echo "Download cc.en.300.bin fasttext word embedding model..."
WE_PATH="$ROOT_DIR/data/we_models"

if [ ! -f "${WE_PATH}/cc.en.300.bin" ]
then
  curl https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz --output "${WE_PATH}/cc.en.300.bin.gz"
  gunzip "${WE_PATH}/cc.en.300.bin.gz"
else
  echo "... skipping download (cc.en.30.bin word embedding model found at ${WE_PATH})."
fi


# 3. Run the test experiment
#		 	       pipeline script                     experiment folder			                   word embedding file	             no. of parallel processes         	signature size	           keyword extraction for word vectors         keyword extraction method for word weights	                   log file		              	                              label file (optional)                                               draw plot (optional, only if label file given)
##############################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
python3 "$ROOT_DIR/src/pipeline.py"	"$ROOT_DIR/data/experiments/test_experiment" 	"$ROOT_DIR/data/we_models/cc.en.300.bin" 	      10			                           20 		                            yake 		                                           	yake 		                       	"$ROOT_DIR/data/logs/test_log"	    	--label_file "$ROOT_DIR/data/experiments/test_experiment/test_experiment_labels"                           --plot 1
