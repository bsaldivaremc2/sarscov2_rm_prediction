# SARS-CoV-2 Prediction of recurrent mutations  
This repository contains supplementary material to the publication XXXXX.  
Contents:  
<li> Models for prediction of recurrent mutations and positions

## Models usage for prediction/inference
<ul>
<li> Dowload the models from:  <a href="https://rovira-my.sharepoint.com/:u:/g/personal/39706766-a_epp_urv_cat/EbpFCfw0L3hMunIpUs7wYK0BHyJjDkv_UFzBiun5xnSHow?e=IDvMqR">7zfile</a>  </li>
[7zfile](https://rovira-my.sharepoint.com/:u:/g/personal/39706766-a_epp_urv_cat/EbpFCfw0L3hMunIpUs7wYK0BHyJjDkv_UFzBiun5xnSHow?e=IDvMqR "Main models for inference")    

<li> Unzip the <strong>pos_mut_models.7z</strong> file with </li>  
</ul>

```bash  
7z e pos_mut_models.7z
```

You should be able to see the following directories after extraction:  
```
tree -d pos_mut_models/
pos_mut_models/
├── mut_pred_5_10_15
│   ├── th10
│   ├── th15
│   └── th5
└── pos_pred_1_5_10_15
    ├── th1
    ├── th10
    ├── th15
    └── th5
```

<li> Create a new conda environment with the requirements of the file <strong>spec-file.txt</strong> . You can replace the <i>new_env_name</i> with your own new environment name</li>
  
```
conda create --name new_env_name --file spec-file.txt
```
Further information to manage conda environments can be seen in [link](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#cloning-an-environment)   

The input file for prediction is expected to be a csv file with the variable columns present in the files:
<li>input_position_example.csv, for RM position prediction, 611 columns </li>
<li>input_mutation_example.csv, for RM mutation prediction 803 columns </li>

Activate the conda environment with:

```
conda activate new_env_name
```
Run the prediction script, example mutation prediction
```
python3 pos_mut_predict.py -data_file="input_mutation_example.csv" \
-models_dir="./pos_mut_models/" \
-output_dir="./output_dir/" \
-batch_size=32 \
-th="15" \
-prediction_type="mutation" \
-file_separator="," \
-first_column_is_index=1 
```
position prediction example
```
python3 pos_mut_predict.py -data_file="input_position_example.csv" \
-models_dir="./pos_mut_models/" \
-output_dir="./output_dir/" \
-batch_size=32 \
-th="5" \
-prediction_type="position" \
-file_separator="," \
-first_column_is_index=0 
```

* data_file: input file with the features used to predict.
* models_dir: directory where the models are stored.
* output_dir: directory where to save the predictions.
* batch_size: Amount of rows from the data_file to process at the same time. 
* th: Recurrent Mutation threshold. 1, 5, 10 and 15 available for position and 5, 10, 15 for mutation.
* prediction_type: Type of prediction, options available are **position** and **mutation**.
* file separator: character that separates the columns in the **data_file**.
* first_columns_is_index: set as 0 if the first columns in **data_file** is a feature and not an index.


The description of each feature column is in the file **variable_names_and_description.csv**.

## Predict mutation using only nucleotide sequence
In addition to work presented, 3 additional models were introduced using only the nucleotide sequence with the format: <i>TACAAA**C**CTTTTC>G</i>. The C in the middle is the position that mutates to **>G**.  
Since the nucleotide sequence is used only, there is a loss of performance of 3%-5% as is present in **seq_only_mut_pred_scores.csv**. 
These modeles can be downloaded from [link](https://rovira-my.sharepoint.com/:u:/g/personal/y-6848578-c_epp_urv_cat/EUO6H3_WKtdOrQNojm65KNMBEt53_vqxo0BOH8HvNJyisQ?e=OV4GRb) 
```
python3 rm_predict_from_seq.py -sequence="TACAAACCTTTTC>G" \
-models_dir="./sequence_predict_models/" \
-output_dir="./output_dir/" \
-batch_size=32 \
-th="15" \
-prediction_type="mutation"
```
Or by using a file with a sequence per file like **seq_file_txt**. Each line with the format: <i>TACAAA**C**CTTTTC>G</i>
```
python3 rm_predict_from_seq.py -sequence_file="seq_file.txt" \
-models_dir="./sequence_predict_models/" \
-output_dir="./output_dir/" \
-batch_size=32 \
-th="15" \
-prediction_type="mutation"
```

If you are using this material please cite the work as:

