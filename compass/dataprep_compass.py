#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
import sys,os
def preprocessing(tmp_df, mean_std_dict, vocab_dict):
    features = {}
    # Normalize numberiacal columns
    for col_name in mean_std_dict.keys():
        _mean, _std = mean_std_dict[col_name]
        features[col_name] = ((tmp_df[col_name]-_mean)/_std)
    # Encode categorical columns as indices
    print (vocab_dict.keys())

    for col_name in vocab_dict.keys():
        features[col_name] = tmp_df[col_name].map(
            {
                j:i for i,j in enumerate(vocab_dict[col_name])
            }
        )
        print (features[col_name])

    # One-hot encoding categorical features
    #for col_name in ["c_charge_degree", "c_charge_desc", "age_cat"]:
    for col_name in ["c_charge_degree", "c_charge_desc", "age_cat"]:#, 'r_charge_degree','vr_charge_degree']:
        features[col_name] = pd.get_dummies(features[col_name], prefix=col_name)
    return pd.concat(features.values(), axis=1)

from fairlib import networks, BaseOptions, dataloaders
import torch

class CustomizedDataset(dataloaders.utils.BaseDataset):

    def load_data(self):

        self.data_dir = os.path.join(self.args.data_dir, "{}.pkl".format(self.split))

        data = pd.read_pickle(self.data_dir)

        self.X = data.drop(['sex', 'race', 'is_recid'], axis=1).to_numpy().astype(np.float32)
        self.y = list(data["is_recid"])
        self.protected_label = list(data["race"])

def compass_analysis():
  Shared_options = {
    # The name of the dataset, correponding dataloader will be used,
    "dataset":  "COMPAS",

    # Specifiy the path to the input data
    "data_dir": "./compass_data",

    # Device for computing, -1 is the cpu
    "device_id": -1,

    # The default path for saving experimental results
    "results_dir":  r"results",

    # The same as the dataset
    "project_dir":  r"dev",

    # We will focusing on TPR GAP, implying the Equalized Odds for binay classification.
    "GAP_metric_name":  "TPR_GAP",

    # The overall performance will be measured as accuracy
    "Performance_metric_name":  "accuracy",

    # Model selections are based on DTO
    "selection_criterion":  "DTO",

    # Default dirs for saving checkpoints
    "checkpoint_dir":   "models",
    "checkpoint_name":  "checkpoint_epoch",


    "n_jobs":   1,
  }
  args = {
    "dataset":Shared_options["dataset"], 
    "data_dir":Shared_options["data_dir"],
    "device_id":Shared_options["device_id"],

    # Give a name to the exp, which will be used in the path
    "exp_id":"vanilla",

    "emb_size": 450-3,
    "lr": 0.001,
    "batch_size": 128,
    "hidden_size": 32,
    "n_hidden": 1,
    "activation_function": "ReLu",

    "num_classes": 2,
    "num_groups": 3, # Balck; White; and Other
  }
# Init the argument
  options = BaseOptions()
  state = options.get_state(args=args, silence=True)
  customized_train_data = CustomizedDataset(args=state, split="train")
  customized_dev_data = CustomizedDataset(args=state, split="dev")
  customized_test_data = CustomizedDataset(args=state, split="test")

# DataLoader Parameters
  tran_dataloader_params = {
        'batch_size': state.batch_size,
        'shuffle': True,
        'num_workers': state.num_workers}

  eval_dataloader_params = {
        'batch_size': state.test_batch_size,
        'shuffle': False,
        'num_workers': state.num_workers}

# init dataloader
  customized_training_generator = torch.utils.data.DataLoader(customized_train_data, **tran_dataloader_params)
  customized_validation_generator = torch.utils.data.DataLoader(customized_dev_data, **eval_dataloader_params)
  customized_test_generator = torch.utils.data.DataLoader(customized_test_data, **eval_dataloader_params)
# Init the argument

  model = networks.classifier.MLP(state)
  model.train_self(
    train_generator = customized_training_generator,
    dev_generator = customized_validation_generator,
    test_generator = customized_test_generator,
  )

def prepare_compass_data():
        pd.options.display.float_format = '{:,.2f}'.format
        dataset_base_dir = "compass_data/"
        dataset_file_name = 'compas-scores-two-years.csv'
        file_path = os.path.join(dataset_base_dir,dataset_file_name)
        temp_df = pd.read_csv(file_path)
        #Notes : preprocessing of data
        temp_df2 = temp_df[( 
               (temp_df['days_b_screening_arrest'] <= 30 ) &  (temp_df['days_b_screening_arrest'] >= -30)  &
               (temp_df['days_b_screening_arrest'].notna() ) &
               (temp_df['c_charge_degree']  != "O") &
               ( temp_df['score_text'] != 'N/A' ) )
               ]


        #print((temp_df['days_b_screening_arrest'] ))
        print (temp_df.columns)
        # Columns of interest
        columns = ['juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count',
                        'age', 
                        'c_charge_degree', 
                        'c_charge_desc',
                        'age_cat',
                        'sex', 'race',  'is_recid', 'days_b_screening_arrest']
        target_variable = 'is_recid'
        target_value = 'Yes'
        columns = [
        'juv_fel_count',
'juv_misd_count',
'juv_other_count',
'priors_count',
'days_b_screening_arrest',
'c_charge_degree',
'c_charge_desc',
'is_recid',
#'r_charge_degree',
#'is_violent_recid',
#'vr_charge_degree',
'v_decile_score',
'sex',
'age_cat',
'race',
#'two_year_recid',
'age'
]
        # Drop duplicates
        temp_df2 = temp_df2[['id']+columns].drop_duplicates()


        df = temp_df2[columns].copy()
       
        print((df.shape[0]))
        # Convert columns of type ``object`` to ``category`` 
        df = pd.concat([
                df.select_dtypes(include=[], exclude=['object']),
                df.select_dtypes(['object']).apply(pd.Series.astype, dtype='category')
                ], axis=1).reindex(df.columns, axis=1)

        # Binarize target_variable
        df['is_recid'] = df.apply(lambda x: 1 if x['is_recid']==1 else 0, axis=1).astype(int)
        
        # Process protected-column values
        race_dict = {'African-American':'Black','Caucasian':'White'}
        df['race'] = df.apply(lambda x: race_dict[x['race']] if x['race'] in race_dict.keys() else 'Other', axis=1).astype('category')
        train_df, test_df = train_test_split(df, test_size=0.30, random_state=42)
        train_df, dev_df = train_test_split(train_df, test_size=0.1, random_state=42)
        cat_cols = train_df.select_dtypes(include='category').columns
        vocab_dict = {}
        for col in cat_cols:
          vocab_dict[col] = list(set(train_df[col].cat.categories))
        

        print(cat_cols)
        temp_dict = train_df.describe().to_dict()
        mean_std_dict = {}
        for key, value in temp_dict.items():
          mean_std_dict[key] = [value['mean'],value['std']]
        print(mean_std_dict)
        train_df = preprocessing(train_df, mean_std_dict, vocab_dict)
        dev_df =  preprocessing(dev_df, mean_std_dict, vocab_dict)
        test_df = preprocessing(test_df, mean_std_dict, vocab_dict)
        train_df.to_pickle(os.path.join(dataset_base_dir, "train.pkl"))
        dev_df.to_pickle(os.path.join(dataset_base_dir, "dev.pkl"))
        test_df.to_pickle(os.path.join(dataset_base_dir, "test.pkl"))

if __name__ == '__main__':
   prepare_compass_data()
#   compass_analysis()
