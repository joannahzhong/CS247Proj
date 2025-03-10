## CS247 Project Gloss Definition

# Directory structure

Root dir: codewoe

Sub-dir: 

*   code: contains the all the codes for model training, predicting, scoring, and etc.
*   models: contains all the files generated during the model training, predicating, and scoring.
*   util: contains utility codes for predication cleanup and generating the visualization


Folders under the 'models':
  
*   defmod-baseline: contains the generated models, predictions and scores using the baseline codes
*   concat: contains the generated models, predictions and scores using the modified codes for the embeddings concatenation approach

    ** electra_sgns: for the electra and sgns embeddings concatenation approach (including the tuned model

    ** electra_sgns_char: for all 3 embeddings concatenation approach (including the tuned model)
*   weighted_sum: contains the generated models, predictions and scores using the modified codes for the weighted sum embeddings approach


Note: under the 'code' directory
*   _defmod_ori.py_: the original unmodified defmod codes
*   _defmod_concat.py_: the modified defmod codes for the embeddings concatenation approach
*   _defmod_ws.py_: the modified defmod codes for the embeddings weighted summation approach
*   _models_ori.py_: the original unmodified model construction codes
*   _models_concat.py_: the modified model construction codes for the embeddings concatenation approach
*   _models_ws.py_: the modified model construction codes for the embeddings weighted summation approach

Typical files inside a specific model folder

  _.pt files_: model related files saved during the training

  _best_scores.txt_: the best loss 

  _hparams.json_: the best hyperparameters

  _defmod_predictions_<name>.json_: the generated predication file before any cleanup

  _defmod_predictions_<name>_clean.json_: the generated predication file after the cleanup (i.e. removing unicode characters)

  _scores.txt_: the evaluation scores (BLEU scores and MoverScore)



# Instructions on how to run the codes.

## Environment Setup:
Use the provided Dockerfile.gpu to build a docker image and then create a docker container using the built image. Note this docker file is tested on a windows computer which has cuda enabled. 

## Dataset:
* Go to https://codwoe.atilf.fr/ 
* Download "Train and development datasets". Exact "en.train.json" and "en.dev.json" from the download zip. 
* Download "Test datasets". Exact "en.test.defmod.json" from the download zip.
* Download "Reference data for scoring program". Exact "en.test.defmod.complete.json" from the download zip.
* Place all the exacted json files into the data directory.

## Running the codes:
### 0. set codwoe as your root/current directory in your terminal.

### 1. To train a model:
  Select a model, i.e. the model which uses the all 3 embedding concatenation approach.
  * In your IDE, n
  * Navigate to the code directory
  * Rename models_concat.py to models.py
  * Rename defmod_concat.py to defmod.py
  * In the terminal, run the following command:
    python3 code/defmod.py --do_train \
    --train_file data/en.train.json \
    --dev_file data/en.dev.json \
    --device cuda \
    --source_arch electra sgns char\
    --summary_logdir models \
    --save_dir models \
    --spm_model_path models
   
  Note:
  * a new folder named "electra_sgns_char" will be automatically created under models
  * the generated files during training will be saved in baseline_archs/models/electra_sgns_char
  * for help with the defmod arguments, run:
   python3 code/defmod.py --help
   
### 2. To make predication:
  * In the terminal, run the following command:
   python3 code/defmod.py --do_pred \
    --test_file data/en.test.defmod.json \
    --device cuda \
    --source_arch electra sgns char\
    --save_dir models/electra_sgns_char \
    --pred_file models/electra_sgns_char/your_pred_file_name.json

  Note: the predication json file will contain unicode characters and/or seq token

### 3. To clean up the prediction file before scoring:
   * In the terminal, run the following command:
     util/clean_defmod_predictions.py models/electra_sgns_char/your_pred_file_name.json
  
  Note: the cleaned predication file will be saved in the same directly as the original file, i.e. your_pred_file_name_clean.json
  
### 4. To score the prediction file:
   * In the terminal, run the following command:
     python3 code/score.py \
      models/electra_sgns_char/your_pred_file_name_clean.json \
      --reference_files_dir data \
      --output_file models/electra_sgns_char/scores.txt 

### 5. To tune the model:
   * In the terminal, run the following command:
     python3 baseline_archs/code/defmod.py --do_htune \
	--train_file data/en.train.json \
	--dev_file data/en.dev.json \
	--device cuda \
    	--source_arch electra sgns char\
    	--summary_logdir models/electra_sgns_char/tune \
    	--save_dir models/electra_sgns_char/tune \
    	--spm_model_path models/electra_sgns_char/tune

### 6. To generate the word cloud visualization:
  e.g. Generate Word Clouds for Ground Truth vs. Predicted Glosses 
   * In the terminal, run the following command:
     python3 util/vis_word_cloud_vs_gt.py \
	    --ground_truth_file data/en.test.defmod.complete.json \
      --pred_file models/electra_sgns_char/your_pred_file_name_clean.json \
	    --save_dir models/electra_sgns_char

     
