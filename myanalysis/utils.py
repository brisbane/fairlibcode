import re
from fairlib.src.analysis.utils import get_dir, retrive_exp_results, retrive_all_exp_results, get_model_scores
import fairlib.src.analysis
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")
import importlib
import yaml
from fairlib.src import analysis

import yaml, string
from functools import partial


# Shared_options["selection_criterion"],
modelnoselection=partial(analysis.model_selection,  
    # the tuned hyperparameters of a methods, which will be used to group multiple runs together.
    # This option is generally used for differentiating models with the same debiasing method but
    # with different method-specific hyperparameters, such as the strength of adversarial loss for Adv
    # Random seeds should not be included here, such that, random runs with same hyperparameters can
    # be aggregated to present the statistics of the results.
    index_column_names =  ["BT", "BTObj", "adv_debiasing","adv_num_subDiscriminator","adv_lambda","lr","dropout","batch_size","weight_decay","lr_scheduler", "encoder_architecture", "n_hidden", "softmax", "hidden_size"],
    # to convenient the further analysis, we will store the resulting DataFrame to the specified path


    # Follwoing options are predefined
    results_dir= "results",
    #results_dir = "/media/user/624EF0CF7D3CCBF1/results",


    GAP_metric_name = "TPR_GAP",
    Performance_metric_name = "accuracy",

    # We use DTO for epoch selection

    checkpoint_dir= "results",
    checkpoint_name= "checkpoint_epoch",
    keep_original_metrics=False,
       override_checkpoint_name=False,
    # If retrive results in parallel
    n_jobs=16,


)
modelselection = partial (modelnoselection, selection_criterion = "DTO")

def pretty_text(textin):
      text=str(textin)
      if text == 'nan' or text == "None":
            return  '-'
      text = text.replace("adv_diverse_lambda", "$\lambdadiff$").replace("adv_lambda", "$\lambdaadv$")
      text=" ".join(  [ word[0].upper() + word[1:] for word in  text.split("_") ])
      text=text.replace("Lr", "LR").replace("N Hidden", "Hidden Layers")
      return text

def hyp_order(hyp):
  d={
    "opt_dir": 1000 ,
    "weight_decay": 22 ,
    "softmax": 2,
    "adv_debiasing": 200 ,
    "adv_diverse_lambda": 202 ,
    "adv_num_subDiscriminator": 203 ,
    "BTObj": 102,
    "dropout": 10,
    "lr": 21 ,
    "batch_size": 9 ,
    "adv_lambda": 201 ,
    "BT": 101 ,
    "lr_scheduler": 20 ,
    "adv_corr_loss": 204 ,
    "hidden_size": 5 ,
    "INLP_min_acc": 303 ,
    "INLP_n": 302 ,
    "encoder_architecture": 3,
    "n_hidden": 4 ,
    "INLP_discriminator_reweighting": 304 ,
    "INLP_by_class": 305 ,
  }
  if hyp in d:
      return d[hyp]
  return 900



def makePlot(results_dict_partial):
#results_dict_partial={ "celeba BTEO" : results_d_b ,
#                       "celeba Vanilla" : results_d_v ,
 #                    }
  plot_df = analysis.final_results_df(
    results_dict = results_dict_partial,
    pareto = True, pareto_selection = "test",
    selection_criterion = None, return_dev = True,
        Fairness_metric_name = "fairness",
    ) #.replace(to_replace = to_replace, value=value)

  analysis.tables_and_figures.make_zoom_plot(
    plot_df, dpi = 100,
    zoom_xlim=(0.7, 1.0),
    zoom_ylim=(0.7, 1.0),
    )


pretty_metric_names={'dev_performance': 'Train Accuracy',  'dev_fairness': "Train MAR GAP", 'dev_DTO': "Train DTO",
           'test_performance': 'Test Accuracy',  'test_fairness': "Test MAR GAP", 'test_DTO': "Test DTO",
           'epoch': 'Epoch'}
def convert_metric(x):
    if x in pretty_metric_names.keys():
        return pretty_metric_names[x]
    elif str(x) in pretty_metric_names.keys():
        return pretty_metric_names[str(x)]
    return x

#restruction iis a dict such as {'softmax' : False}
def get_best_result(df, restriction=False, assume_match_if_no_index=True, selection_criterion='dev_DTO'):

    if  not restriction:
      if selection_criterion.find("performance") != -1:
        return df.iloc[df[selection_criterion].argmax()]
      return df.iloc[df[selection_criterion].argmin()]
    valid_keys=[]
    for r in restriction.keys():
        if r not in df.index.names and not assume_match_if_no_index:
            print (f"{r} is not in {df.index.names} and Ive been told to return nothing if this is the case as assume_match_if_no_index={assume_match_if_no_index}")
            return None
        valid_keys.append(r)

    _df=df
    for key in valid_keys:
        value=restriction[key]
        _df = _df[np.in1d( _df.index.get_level_values ( _df.index.names.index(key)), [value]) ]
    if selection_criterion.find("performance") != -1:
        return _df.iloc[_df[selection_criterion].argmax()]
    return _df.iloc[_df[selection_criterion].argmin()]
def epoch_plot_from_single_run(dfin, plotname="results", epoch=0):

    df=dfin.copy()
    df['dev_fairness']=1 - df['dev_fairness']
    df['test_fairness']= 1 - df['test_fairness']
    evaluation_criteria='Test DTO' 
    df.rename(columns=pretty_metric_names, inplace=True)
 #   dfp1=pd.DataFrame({'Epoch': df['Epoch'], evaluation_criteria :df.pop('Test DTO')})


#    dfp=dfp1.melt('Epoch', var_name='cols', value_name='vals')

    dfm=df.melt('Epoch', var_name='cols', value_name='vals')
    #sns.catplot(x="epoch", y="vals", s=20, hue="cols", data=dfm, kind="strip", 
     #            palette="dark", alpha=.6, hue_order=sorted(df.columns)
     #          )
    plt.axvspan(df['Epoch'].min(), -1, color='grey', alpha=0.5, lw=0) 
#    dfm['_size']=((dfm['cols'] == 'Test DTO')+1)
    ax = sns.lineplot(x="Epoch", y="vals", hue="cols",data=dfm, alpha=0.7 )#size='_size',
#    ax = sns.lineplot(x=dfp.Epoch, y=dfp.vals,alpha=0.7, size=2 , label=evaluation_criteria)#size='_size',
    #sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.legend(loc= "center left", bbox_to_anchor=(0.65, 0.5), ncol=1)
#    ax.legend(loc= "center left", bbox_to_anchor=(0.05, 0.5), ncol=1)

    plt.axvline(x=epoch) 

    plt.gca().set_xlim([df['Epoch'].min(), df['Epoch'].max()+0.5])
    plt.title(plotname.replace('\\','').replace("INLP Standard Section", "Vanilla"))
    plt.savefig(f"/home/user/fairlibcode/plots/right_legend/epoch_{plotname}.png".replace(' ','_').replace('\\','').lower() )
    plt.show()
    return dfm

def epoch_plot(results_framein, plotname=None, INLP=False):
    if plotname == None:
        try:
           plotname = df.name
        except:
           plotname="data"
    results_frame=results_framein.copy()
    best = get_best_result(results_frame)
    opts =  best['opt_dir']
    epoch = best['epoch']
    with open(opts) as stream:
     try:
        dict = yaml.safe_load(stream)
     except yaml.YAMLError as exc:
        print(exc)
    results_dir= dict['model_dir']
    log_dir = results_dir
    results = {}
    splits = log_dir.split('/')
    s=0
    for i in splits: 
     
      if i =="results":
         
         break
      s+=1
    gd1="results"
    gd2=splits[s+1]+'/'+splits[s+2]
    gd3=splits[s+5]
    gd4="checkpoint_epoch"

   # gd4="BEST_checkpoint"
    gd5=splits[s+3]
    exp = get_dir(gd1,gd2,gd3,gd4,gd5)
    print (exp)
    #return exp
    df = get_model_scores(exp[0], "TPR_GAP", "accuracy")

    if INLP:
        van_epoch = get_best_result(df)['epoch']
        
        epoch_plot_from_single_run(df.copy(), f"{plotname} Standard Section", van_epoch)
        #print ("inlp")
        gd4="INLP_"+gd4
        #print("get_best_result(df):", get_best_result(df))
        df['epoch']=df['epoch'] -  van_epoch -1

        df3 = df[df['epoch'] < 0.5] 
        exp = get_dir(gd1,gd2,gd3,gd4,gd5)
        #print (exp[0])
        df2 =get_model_scores(exp[0], "TPR_GAP", "accuracy")


        df = pd.concat([df3, df2]  , ignore_index=True)
    return epoch_plot_from_single_run(df, plotname, epoch)


#Note the INLP doesnt do anything at the moment, and doesnt need to
#def print_results(results_framein, name="results", summary=True, Header=False, INLP=False):
#        #note line 267 fairlib/src/analysis/utils.py dev alteady used to select best epoch from a hyper run
#    #print ("1", name)
#    if name != 'results':
#return epoch_plot(results_frame, name, INLP)
#Note the INLP doesnt do anything at the moment, and doesnt need to
def print_results(results_framein, name="results", summary=True, Header=False, INLP=False, selection_criterion="dev_DTO" ):
        #note line 267 fairlib/src/analysis/utils.py dev alteady used to select best epoch from a hyper run
    #print ("1", name)
    results_frame=results_framein.copy()
    if name != 'results':
      try:
           plotname = results_frame.name
      except:
           results_frame.name=name  
    results_frame['dev_fairness']=1 -results_frame['dev_fairness']
    results_frame['test_fairness']= 1 - results_frame['test_fairness']
    best=get_best_result(results_frame, selection_criterion=selection_criterion)

    if not summary:
        print ("Best DTO")
        print ("dev:", results_frame.iloc[(results_frame['dev_DTO'].argmin())]['dev_DTO'])
        print ("test:",results_frame.iloc[(results_frame['dev_DTO'].argmin())]['test_DTO'])
        print ("Best performance")
        print ("dev:", results_frame.iloc[(results_frame['dev_DTO'].argmin())]['dev_performance'])
        print ("test:",results_frame.iloc[(results_frame['dev_DTO'].argmin())]['test_performance'])
        print ("Best fairness")
        print ("dev:", results_frame.iloc[(results_frame['dev_DTO'].argmin())]['dev_fairness'])
        print ("test:",results_frame.iloc[(results_frame['dev_DTO'].argmin())]['test_fairness'])
        
        print ("everything:", results_frame.iloc[(results_frame['dev_DTO'].argmin())])
        
       # print ("dev:", results_frame.iloc[(results_frame['test_DTO'].argmin())]['dev_DTO'])
       # print ("test:",results_frame.iloc[(results_frame['test_DTO'].argmin())]['test_DTO'])
        print ("hyperparameters")
        for i, x  in  zip(
                          results_frame.iloc[(results_frame['dev_DTO'].argmin())].name, 
                          results_frame.index.names ):
            print (x , i)
        for i, x  in  zip(
                          results_frame.iloc[(results_frame['test_DTO'].argmin())].name, 
                          results_frame.index.names ):
            print (x , i)


    metrics = ['dev_DTO', 'test_DTO',  'dev_performance','test_performance','dev_fairness', 'test_fairness', 'epoch']
    title="      & \\text_bf{"+ convert_metric(str(metrics[0])) +'}'
    values="\\textbf{" + name+"}"+" & {0:.3g}".format(best[metrics[0]])
    for i in metrics[1:]: 
        title+=" & "+"\\textbf{" + convert_metric(i) + '}'
        values+=" &  " + "{0:.3g}".format(best[i])
    title+=' \\\\'
    values+=' \\\\'
    if Header: print (title.replace('_', ' '))
    print (values.replace("_", " "))



def print_hyp1(results_frame, name="results", summary=True, Header=False):
        #note line 267 fairlib/src/analysis/utils.py dev alteady used to select best epoch from a hyper run
    print ("hyperparameters")
    for i, x  in  zip(
                          results_frame.iloc[(results_frame['dev_DTO'].argmin())].name, 
                          results_frame.index.names ):
        if Header:
            if type(i) == "float": print ("{0} & {1:.3g}".format(x, i))
            if type(i) != "float": print ("{0} & {1}".format(x, i))
        else:
            if type(i) == "float": print ("{0:.3g}".format(i))
            if type(i) != "float": print (" & {0}".format( i))
    metrics = ['dev_DTO', 'test_DTO',  'dev_performance','test_performance','dev_fairness', 'test_fairness']
    title="      & "+ str(metrics[0])
    values="\\textbf{" + name+"}"+" & {0:.3g}".format(best[metrics[0]])
    for i in metrics[1:]: 
        title+=" & "+"\\textbf{" + convert_metric(i) + '}'
        values+=" &  " + "{0:.3g}".format(best[i])

    title+=' \\\\'
    values+=' \\\\'
    if Header: print (title.replace('_', '\_'))
    print (values.replace("_", "\_"))
    


def print_hyp(results_dict):
    d={'opt_dir':{}}

    for k in results_dict.keys():
            best_row = get_best_result(results_dict[k])
            try:
                yaml_file=best_row.opt_dir
                #print (yaml_file)
                with open(yaml_file, 'r') as f:
                  opt = yaml.full_load(f) 
                  d['opt_dir'][k]=opt
            except:
                print("exception")
                opt={}
            #print (opt)
            for hyp, val in  (zip(  results_dict[k].index.names,   best_row.name)):
                if hyp not in d:
                    d[hyp]  ={}
                d[hyp][k]=val
                if hyp in opt:
                    #print ("Override", hyp, val, opt[hyp])
                    d[hyp][k]=opt[hyp]
                
           # print (d)
    count=0
    for i in sorted(d.keys(), key = hyp_order):
        print (  '    "'+ str(i) + '":' ,count, ","  )
        count+=1
    
    l=r"\textbf{Hyperparameter}"
    for exp in results_dict.keys():
       l+=r" & \textbf{" + exp + "}"
    l+=" \\\\ \hline \hline"
    print (l.replace('_', ' '))
    l=r"\textbf{Hyperparameter}"
    for exp in results_dict.keys():
       l+=r" & \textbf{" + re.sub(r'.*/', '', exp.replace("dev", "train")) + "}"
    l+=" \\\\ \hline \hline"
    print (l.replace('_', ' '))
    l=""

    for hyp in sorted(d.keys(), key = hyp_order):
        l="\\textbf{" + pretty_text(hyp) + "}"
        if hyp =='opt_dir': continue
        for exp in  results_dict.keys():
            
            if exp in d[hyp]:
               text=str(d[hyp][exp])
            elif hyp in d['opt_dir'][exp]:
                text = str(d['opt_dir'][exp][hyp])
            else:
                text = '-'

            l+=" & " + pretty_text(text)
        l+=" \\\\ "
        print (l.replace('_', ' '))

def print_hyp_old(results_dict):
    d={}
    for k in results_dict.keys():
            for hyp, val in  (zip(  results_dict[k].index.names,   results_dict[k].iloc[(results_dict[k]['dev_DTO'].argmin())].name)):
                if hyp not in d:
                    d[hyp]  ={}
                d[hyp][k]=val
           # print (d)
    l=r"\textbf{Hyperparameter}"
    for exp in results_dict.keys():
       l+=r" & \textbf{" + exp + "}"
    l+=" \\\\ \hline \hline"
    print (l.replace('_', ' '))
    l=r"\textbf{Hyperparameter}"
    for exp in results_dict.keys():
       l+=r" & \textbf{" + re.sub(r'.*/', '', exp.replace("dev", "train")) + "}"
    l+=" \\\\ \hline \hline"
    print (l.replace('_', ' '))
    l=""
    for hyp in d.keys():
        l="\\textbf{" + str(hyp) + "}"
        for exp in  results_dict.keys():
            if exp in d[hyp]:
               text=str(d[hyp][exp])
               if text == 'nan':
                   text = '-'
               l+=" & " + text
            else:
                l+=" & - "
        l+=" \\\\ "
        print (l.replace('_', ' '))
def print_results_p(results_frame, name="results", summary=True, Header=False, INLP=False):
  results_frame.name=name
  print_results(results_frame, name, summary, Header, INLP)

  return epoch_plot(results_frame, name, INLP)
