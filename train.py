#!/usr/bin/env python3
#python /home/user/miniconda3/envs/pristine/lib/python3.8/site-packages/fairlib/__main__.py  --dataset Bios_gender --num_classes 28  --emb_size 768 --batch_size 512  --data_dir $HOME/miniconda3/data/bios --lr 0.003 --dropout 0.5 --test_batch_size 512 --hidden_size 300 --n_hidden 2 --epochs_since_improvement 10 --selection_criterion DTO --Performance_metric_name accuracy #h
import numpy as np
import pandas as pd
import os
import fairlib
import torch
#import tensorflow_datasets as tfds

bs=64

exp_id="adv"
#exp_id="test"
exp_id="vanilla"
exp_id="BTEO"

exp_id_sub="2109.08253_lrs"
exp_id_sub=""

#exp_id="mnist"
#exp_id="BTEO"
dataset="Bios_gender"
#dataset="bffhq"
#dataset="MNIST"
dataset="COMPAS_race"
dataset='bffhq'
import yaml

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
if len(exp_id_sub):
    exp=exp+'_'+exp_id_sub
opts_file=Path('/home/user/fairlibcode/example_conf_file/'+dataset+"_"+exp+'.yaml')
Shared_options = yaml.safe_load(opts_file.read_text())


args = {
    "dataset":Shared_options["dataset"],
    "data_dir":Shared_options["data_dir"],
    "device_id":Shared_options["device_id"],

    # Give a name to the exp, which will be used in the path
    "exp_id":dataset+"_"+exp,
}
args=Shared_options
Shared_options['exp_id']=exp
# Init the argument
options = fairlib.BaseOptions()
state = options.get_state(args=args, silence=True)
print(state)
fairlib.utils.seed_everything(2022)

# Init Model
# customer with
# https://hanxudong.github.io/fairlib/backup/tutorial_notebooks/tutorial_colored_MNIST.html?highlight=mnist


model = fairlib.networks.get_main_model(state)

model.train_self()
print('Finished training!')
sys.exit(1)
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
print(f"Predicted class index: {predicted_labels}")
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

save_path_new= os.path.join(results_dir,project_dir,dataset,exp,"results.pkl")
print (save_path_new)
print (save_path)
print ( Shared_options["Performance_metric_name"], Shared_options["selection_criterion"])
#Shared_options["Performance_metric_name"]='Performance'
results = analysis.model_selection(
    # exp_id started with model_id will be treated as the same method, e.g, vanilla, and adv
    model_id= exp,

    # the tuned hyperparameters of a methods, which will be used to group multiple runs together.
    # This option is generally used for differentiating models with the same debiasing method but 
    # with different method-specific hyperparameters, such as the strength of adversarial loss for Adv
    # Random seeds should not be included here, such that, random runs with same hyperparameters can
    # be aggregated to present the statistics of the results. 
    index_column_names = ["BT", "BTObj", "adv_debiasing","lr", "dropout","adv_num_subDiscriminator",  "adv_diverse_lambda", "adv_lambda"],

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

