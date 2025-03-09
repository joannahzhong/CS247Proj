# CS247Proj
CS247 Project Gloss Definition

## Directory structure

Root: codewoe
Sub-dir: 

*   code: contains the all the codes for model training, predicting, scoring, and etc.
*   models: contains all the files generated during the model training, predicating, and scoring.
*   util: contains utility codes for predication cleanup and generating the visualization

Note: under the 'models' directory
*   _defmod_ori.py_: the original unmodified defmod codes
*   _defmod_concat.py_: the modified defmod codes for the embeddings concatenation approach
*   _models_ori.py_: the original unmodified model construction codes
*   _models_concat.py_: the modified model construction codes for the embeddings concatenation approach

Under the 'models' directory
  
*   defmod-baseline: contains the generated models, predictions and scores using the baseline codes
*   concat: contains the generated models, predictions and scores using the modified codes for the embeddings concatenation approach
  
  -electra_sgns: for the electra and sgns embeddings concatenation approach (including the tuned model
    
  -electra_sgns_char: for all 3 embeddings concatenation approach (including the tuned model)

*   weighted_sum: contains the generated models, predictions and scores using the modified codes for the weighted sum embeddings approach


Typical files inside a specific model folder

  _.pt files_: model related files saved during the training

  _best_scores.txt_: the best loss 

  _hparams.json_: the best hyperparameters

  _defmod_predictions_<name>.json_: the generated predication file before any cleanup

  _defmod_predictions_<name>_clean.json_: the generated predication file after the cleanup (i.e. removing unicode characters)

  _scores.txt_: the evaluation scores (BLEU scores and MoverScore)


------------------------------------------------------------------------------------------------------------------------------------------

## Instructions on how to run the codes.

Environment Setup:
Use the provided Dockerfile.gpu to build a docker image and then create a docker container using the built image. Note this docker file is tested on a windows computer which has cuda enabled. 

Dataset:
-Go to https://codwoe.atilf.fr/ 
-Download "Train and development datasets". Exact "en.train.json" and "en.dev.json" from the download zip. 
-Download "Test datasets". Exact "en.test.defmod.json" from the download zip.
-Download "Reference data for scoring program". Exact "en.test.defmod.complete.json" from the download zip.
-Place all the exacted json files into the data directory.

Running the codes:
1. To train a model:
  Select a model, i.e. the model which uses the all 3 embedding concatenation approach.
  -Navigate to codwoe\code
  -Rename models_concat.py to models.py
  -Rename defmod_concat.py to defmod.py
  -In the terminal, run the following command:
    python3 baseline_archs/code/defmod.py --do_train \
    --train_file data/en.train.json \
    --dev_file data/en.dev.json \
    --device cuda \
    --source_arch electra sgns char\
    --summary_logdir baseline_archs/models \
    --save_dir baseline_archs/models \
    --spm_model_path baseline_archs/models
   
  Note:
  -a new folder named "electra_sgns_char" will be created under models
  -the generated files during training will be saved in baseline_archs/models/electra_sgns_char
  -for help with the defmod arguments, run:
   python3 baseline_archs/code/defmod.py --help
   
2. To make predication:
  -In the terminal, run the following command:
   python3 baseline_archs/code/defmod.py --do_pred \
    --test_file data/en.test.defmod.json \
    --device cuda \
    --source_arch electra sgns char\
    --save_dir baseline_archs/models/electra_sgns_char \
    --pred_file baseline_archs/models/electra_sgns_char/your_pred_file_name.json
   Note:
   -the predication json file will contain unicode characters and/or seq token

3. To clean up the prediction file before scoring:
   -In the terminal, run the following command:

