#!/usr/bin/env python3
#python /home/user/miniconda3/envs/pristine/lib/python3.8/site-packages/fairlib/__main__.py  --dataset Bios_gender --num_classes 28  --emb_size 768 --batch_size 512  --data_dir $HOME/miniconda3/data/bios --lr 0.003 --dropout 0.5 --test_batch_size 512 --hidden_size 300 --n_hidden 2 --epochs_since_improvement 10 --selection_criterion DTO --Performance_metric_name accuracy #h
import numpy as np
import pandas as pd
import os
import fairlib
from fairlib.src.base_options import State
import torch
#import tensorflow_datasets as tfds

do_train=True
bs=64
exp_id="BTEO"
#exp_id="adv"
#exp_id="dadv"
#exp_id="test"
#exp_id="vanilla"

#exp_id_sub="2109.08253_lrs"
exp_id_sub=""

#exp_id="mnist"
dataset="Bios_gender"
#dataset="bffhq"
#dataset="MNIST"
#dataset="COMPAS_race"
import yaml
from dotmap import DotMap
from fairlib import analysis
import altair as alt
from vega_datasets import data
from pathlib import Path
checkpoint_dir="./results"

base_data_dir = "/home/user/miniconda3/"

import fairlib.src.dataloaders.loaders as fldl
fldl.default_dataset_roots['Bios_gender']='/bios'
data_dir=base_data_dir + fldl.data_dir +fldl.default_dataset_roots[dataset]

import yaml
from pathlib import Path



"""
Shared_options = {

    "batch_size": bs,
    "num_classes": 28,
    "emb_size": 768,
    "lr": 0.003,
    "dropout": 0.5,
    "test_batch_size": bs,
    "hidden_size": 300,
    "n_hidden": 2,
    "epochs_since_improvement": 10,


    # The name of the dataset, corresponding dataloader will be used,
    "dataset":  "Bios_gender",

    # Specifiy the path to the input data
    "data_dir": data_dir,

    # Device for computing, -1 is the cpu
    "device_id":    0,

    # The default path for saving experimental results
    "results_dir":  r"results",

    "save_path" : save_path_new,

    # Will be used for saving experimental results
    "project_dir":  r"dev",

    # We will focusing on TPR GAP, implying the Equalized Odds for binary classification.
    "GAP_metric_name":  "TPR_GAP",

    # The overall performance will be measured as accuracy
    "Performance_metric_name":  "accuracy",

    # Model selections are based on DTO
    "selection_criterion":  "DTO",

    # Default dirs for saving checkpoints
    "checkpoint_dir":   "results",
    "checkpoint_name":  "checkpoint_epoch",

    # Loading experimental results
    "n_jobs":   1,
}
"""
exp=exp_id
opts_file=Path('/home/user/fairlibcode/example_conf_file/'+dataset+"_"+exp+'.yaml')
Shared_options = yaml.safe_load(opts_file.read_text())



args = {
    "dataset":Shared_options["dataset"],
    "data_dir":Shared_options["data_dir"],
    "device_id":Shared_options["device_id"],

    # Give a name to the exp, which will be used in the path
    #"exp_id":dataset+"_"+exp,
}
Shared_options['exp_id']=exp
args=Shared_options

# Init the argument

# Init Model
# customer with
# https://hanxudong.github.io/fairlib/backup/tutorial_notebooks/tutorial_colored_MNIST.html?highlight=mnist



# Init the argument

def configure_args(args,hyperparameters):
    hypkeys = hyperparameters.keys()
    expid_post=""
    args=DotMap(args)
    key='dropout'
    seen_keys=[key]
    if key in hypkeys:
        value=hyperparameters[key]
        args.adv_dropout = value
        args.dropout = value
        expid_post+=f"_{key}{value}"

    key='lr'
    seen_keys.append(key)
    if key in hypkeys:
        value = hyperparameters[key]
        args.lr= value
        args.adv_lr= value
        expid_post+=f"_{key}{value}"
    key='batch_size'
    seen_keys.append(key)
    if key in hypkeys:
        value = hyperparameters[key]
        args.batch_size= value
        args.adv_batch_size= value
        args.test_batch_size= value
        args.adv_test_batch_size= value
        expid_post+=f"_{key}{value}"

    for key, value in hyperparameters.items():
        #skip those with special handling
        if key in seen_keys: continue
        args[key]=value
        seen_keys.append(key)
        expid_post+=f"_{key}{value}"

    args.exp_id =  args.exp_id + expid_post
    #args.model_dir =  expid_post
    print ("configure_args",  args.model_dir)
    print (args.toDict())
    return args.toDict()


hyperparameters_main={'all' :
        {
        'dropout': [0.0,0.2,0.3,0.41,0.39,0.5],
        'lr': [0.000001,0.000003,0.00001,0.00003,0.0001,0.0003,0.001,0.003, 0.01,0.03,0.1],
        'batch_size':[ 2048,1024,512,256,128]
        },
        "COMPAS_race":
        { 'vanilla': {
        'dropout': [0.4,0.3, 0.5,0.7,0.9],
        'lr': [0.1,0.2,0.05,0.5,1],
        'batch_size':[ 4096, 2048, 1024],
        'weight_decay':[0.5, 0.7,1,2,10,100]
            },

            'BTEO':
            { 'BT' : ['Reweighting', 'Resampling'],
              'BTObj': ['EO']
            }
         },
        "Bios_gender":
         {
             'vanilla':
             {
              #'dropout': [0.1,0.2,0.3,0.5, 0.4],
              'dropout': [0.3,0.5, 0.4],

              #main search
              #'lr': [0.00003, 0.0001,0.0003,0.001,0.003, 0.01],
              #derived search
              'lr': [0.0003,0.001,0.003,],
              'batch_size':[512,256],
              'weight_decay':[0.3,0.4,0.5]
             },
            'BTEO':
            { 'BT' : ['Reweighting', 'Resampling'],
              'BTObj': ['EO'],
#              'dropout': [0.3,0.5, 0.4],
              'dropout': [0.3,0.2,0.1],
              'lr': [0.0001,0.0003,0.001],
              'batch_size':[512,256],
              'weight_decay':[0.0,0.1]
            },
            'dadv':
            { 'adv_corr_loss' : [False],
#              'adv_diverse_lambda': [1,10,100,1000],
              'adv_diverse_lambda': [10,100,1000],
#              'adv_lambda': [0.1,0.3,0.8,1.0,3.0],
              'adv_lambda': [0.8,1.0,3.0],
              'adv_num_subDiscriminator' : [1, 3,5],
              'dropout': [0.2,0.3,0.4],
              'lr': [0.0003,0.001,0.003,],
              'batch_size':[512,256],
            },
         },
         'bffhq':
         {
             'vanilla':
             {
                'batch_size':[512],
                'dropout': [0.1,0.2,0.3,0.4,0.5],
                'lr': [ 0.001, 0.003, 0.01,0.03],
                'weight_decay':[0.0,0.2,0.4,0.6,0.8],
             }
        }
}
try:
        
    hyperparameters = hyperparameters_main[dataset][exp]
except:
   hyperparameters = hyperparameters_main["all"]
   raise
print (dataset, exp, hyperparameters_main[dataset][exp])
count=0
from sklearn.model_selection._search import  ParameterGrid
training=0
if do_train:
  for hyper_now in ParameterGrid(hyperparameters):
  
    options = fairlib.BaseOptions()
    fairlib.utils.seed_everything(2022)
    print(hyper_now)
    _args=configure_args(args, hyper_now)
    _md=State(_args).get_model_directory()
    print(_md)
    if os.path.exists (os.path.join(_md, "BEST_checkpoint.pth.tar")): continue
    print(_args['exp_id'], _args['dropout'])

    state = options.get_state(args=_args, silence=True)

    training=1

    print ( state.exp_id)
    print(_args)
    #_exp= state.exp_id

    fairlib.utils.seed_everything(2022)

# Init Model
# customer with
# https://hanxudong.github.io/fairlib/backup/tutorial_notebooks/tutorial_colored_MNIST.html?highlight=mnist


    model = fairlib.networks.get_main_model(state)
    print (state)
    print (model)

    model.train_self()

    print('Finished training!')
    ds=state.test_generator.dataset
    #oupt=model(ipts)
    enumerate(state.test_generator)
    predicted_labels=[]
    for  batch in  state.test_generator:

    #   x, y = batch
       print(batch[0].shape)
       with torch.no_grad():
          output = model(batch[0].to('cuda'))
          _, predicted_class = torch.max(output, 1)

          predicted_labels+=predicted_class.cpu().tolist()
     #  print (batch[1])
    del model
    del state
    torch.cuda.empty_cache()

if not training:
     print("didnt actually do anything!")
     sys.exit(1)
#print(f"Predicted class index: {predicted_labels}")
#dfdf
#pred, output= model(ds.X)
project_dir="dev"
results_dir="./results"
checkpoint_name="checkpoint_epoch"
#epoch="11.00"

print (opts_file)
Shared_options={}
GAP_metric_name=  "TPR_GAP"
Performance_metric_name = "accuracy"
selection_criterion="DTO"
#if os.path.exists (opts_file) :
#    Shared_options = yaml.safe_load(opts_file.read_text())
#    print(Shared_options)
#else:
#    print ("Cant find shared opts file")

if "project_dir" in Shared_options.keys():
    project_dir=Shared_options['project_dir']
else:
    Shared_options['project_dir'] = project_dir

if "results_dir" in Shared_options.keys():
    results_dir=Shared_options['results_dir']
else:
    Shared_options['results_dir'] = results_dir

if "checkpoint_dir" in  Shared_options.keys():
    checkpoint_dir=Shared_options['checkpoint_dir']
else:
    Shared_options['checkpoint_dir'] = checkpoint_dir

if "checkpoint_name" in  Shared_options.keys():
    checkpoint_name=Shared_options['checkpoint_name']
else:
    Shared_options['checkpoint_name'] = checkpoint_name

if "GAP_metric_name" in  Shared_options.keys():
    GAP_metric_name=Shared_options['GAP_metric_name']
else:
    Shared_options['GAP_metric_name'] = GAP_metric_name

if "Performance_metric_name" in  Shared_options.keys():
    Performance_metric_name=Shared_options['Performance_metric_name']
else:
    Shared_options['Performance_metric_name'] = Performance_metric_name

if "selection_criterion" in  Shared_options.keys():
    selection_criterion=Shared_options['selection_criterion']
else:
    Shared_options['selection_criterion'] = selection_criterion

fname =  "_".join([dataset, exp, "df"]) +".pkl"
print (results_dir,project_dir,"analysis",dataset, exp, fname)
save_path_dir = os.path.join(results_dir,project_dir,"analysis")
save_path= os.path.join(save_path_dir, fname)

try: os.mkdir(save_path_dir)
except: pass

save_path_new= os.path.join(results_dir,project_dir,dataset,str(exp)+"_results.pkl")
print (save_path_new)
print (save_path)
print ( Shared_options["Performance_metric_name"], Shared_options["selection_criterion"])
index_column_names = ["BT", "BTObj", "adv_debiasing","adv_num_subDiscriminator",  "adv_diverse_lambda", "adv_lambda"]
index_column_names+=list(hyperparameters.keys())

print(index_column_names)

#Shared_options["Performance_metric_name"]='Performance'
results = analysis.model_selection(
    # exp_id started with model_id will be treated as the same method, e.g, vanilla, and adv
    model_id= exp,

    # the tuned hyperparameters of a methods, which will be used to group multiple runs together.
    # This option is generally used for differentiating models with the same debiasing method but 
    # with different method-specific hyperparameters, such as the strength of adversarial loss for Adv
    # Random seeds should not be included here, such that, random runs with same hyperparameters can
    # be aggregated to present the statistics of the results. 
    index_column_names = list(set(index_column_names)),

    # to convenient the further analysis, we will store the resulting DataFrame to the specified path
    save_path = save_path_new,

    # Follwoing options are predefined
    results_dir= results_dir,
    project_dir= project_dir+"/"+dataset,
    GAP_metric_name = Shared_options["GAP_metric_name"],
    Performance_metric_name = Shared_options["Performance_metric_name"],
    # We use DTO for epoch selection
    selection_criterion = Shared_options["selection_criterion"],
    checkpoint_dir= checkpoint_dir,
    checkpoint_name= checkpoint_name,
    # If retrive results in parallel
    #n_jobs=Shared_options["n_jobs"],
)
print ("Shared_options")
print (Shared_options)

print(results.to_string())
os.rename(save_path_new, save_path)
print (f"Moved {save_path_new} to {save_path}")

#note line 267 fairlib/src/analysis/utils.py dev alteady used to select best epoch from a hyper run
print ("Best DTO")
print ("dev:", results.iloc[(results['dev_DTO'].argmin())]['dev_DTO'])
print ("test:",results.iloc[(results['dev_DTO'].argmin())]['test_DTO'])
print ("hyperparameters")
for i, x  in  zip(
                  results.iloc[(results['dev_DTO'].argmin())].name,
                  results.index.names ):
    print (x , i)
