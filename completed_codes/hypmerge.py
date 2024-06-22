def merge( x,y):
     z = x.copy()   # start with keys and values of x
#     z.update(y)    # modifies z with keys and values of y
     for key in y.keys():
        if key == 'all': continue
        if key not in z.keys():
            z[key]=y[key]
        else:
            nestedy=y[key]
            nestedz=z[key]
            for nkey in nestedy.keys():
#                print (nkey)
                if nkey not in nestedz.keys():
                    z[key][nkey]=nestedy[nkey]
                else:
                    hynestedy=nestedy[nkey]
                    hynestedz=nestedz[nkey]
                    #print (key, nkey)
                    #print (hynestedy)
                    for hyp in hynestedy.keys():

                        if hyp not in hynestedz.keys():
                          #this should be a new array
                          z[key][nkey][hyp]=hynestedy[hyp]
                        else:
                          first_list = z[key][nkey][hyp]
                          second_list = y[key][nkey][hyp]
                          in_first = set(first_list)
                          in_second = set(second_list)

                          in_second_but_not_in_first = in_second - in_first

                          result = first_list + list(in_second_but_not_in_first)
                          z[key][nkey][hyp]=result


     return z
z={}
import hyp_hypscan_adult_bteo
z=merge(z, hyp_hypscan_adult_bteo.hyperparameters_main)
import hyp_hypscan_adult_dadv_decreasingNN
z=merge(z, hyp_hypscan_adult_dadv_decreasingNN.hyperparameters_main)
import hyp_hypscan_adult_dadv
z=merge(z, hyp_hypscan_adult_dadv.hyperparameters_main)
import hyp_hypscan_adult_inlp
z=merge(z, hyp_hypscan_adult_inlp.hyperparameters_main)
import hyp_hypscan_adult_vanilla_DecreasingNN
z=merge(z, hyp_hypscan_adult_vanilla_DecreasingNN.hyperparameters_main)
import hyp_hypscan_adult_vanilla
z=merge(z, hyp_hypscan_adult_vanilla.hyperparameters_main)
import hyp_hypscan_bffhq_bteo2
z=merge(z, hyp_hypscan_bffhq_bteo2.hyperparameters_main)
import hyp_hypscan_bffhq_bteo
z=merge(z, hyp_hypscan_bffhq_bteo.hyperparameters_main)
import hyp_hypscan_bffhq_dadv
z=merge(z, hyp_hypscan_bffhq_dadv.hyperparameters_main)
import hyp_hypscan_bffhq_inlp
z=merge(z, hyp_hypscan_bffhq_inlp.hyperparameters_main)
import hyp_hypscan_bffhq_van1
z=merge(z, hyp_hypscan_bffhq_van1.hyperparameters_main)
import hyp_hypscan_bffhq_van2
z=merge(z, hyp_hypscan_bffhq_van2.hyperparameters_main)
import hyp_hypscan_bffhq_vanilla
z=merge(z, hyp_hypscan_bffhq_vanilla.hyperparameters_main)
import hyp_hypscan_bios_bteo_ds_paper
z=merge(z, hyp_hypscan_bios_bteo_ds_paper.hyperparameters_main)
import hyp_hypscan_bios_bteopt2
z=merge(z, hyp_hypscan_bios_bteopt2.hyperparameters_main)
import hyp_hypscan_bios_bteopt3
z=merge(z, hyp_hypscan_bios_bteopt3.hyperparameters_main)
import hyp_hypscan_bios_bteo
z=merge(z, hyp_hypscan_bios_bteo.hyperparameters_main)
import hyp_hypscan_bios_cadv
z=merge(z, hyp_hypscan_bios_cadv.hyperparameters_main)
import hyp_hypscan_bios_cdadv
z=merge(z, hyp_hypscan_bios_cdadv.hyperparameters_main)
import hyp_hypscan_bios_dadv
z=merge(z, hyp_hypscan_bios_dadv.hyperparameters_main)
import hyp_hypscan_bios_inlp
z=merge(z, hyp_hypscan_bios_inlp.hyperparameters_main)
import hyp_hypscan_bios_vanilla
z=merge(z, hyp_hypscan_bios_vanilla.hyperparameters_main)
import hyp_hypscan_celeba_bm_bteo
z=merge(z, hyp_hypscan_celeba_bm_bteo.hyperparameters_main)
import hyp_hypscan_celeba_bm_dadv
z=merge(z, hyp_hypscan_celeba_bm_dadv.hyperparameters_main)
import hyp_hypscan_celeba_bm_inlp
z=merge(z, hyp_hypscan_celeba_bm_inlp.hyperparameters_main)
import hyp_hypscan_celeba_dadv
z=merge(z, hyp_hypscan_celeba_dadv.hyperparameters_main)
import hyp_hypscan_celeba_vanilla
z=merge(z, hyp_hypscan_celeba_vanilla.hyperparameters_main)
import hyp_hypscan_celebMHQ_blond_male
z=merge(z, hyp_hypscan_celebMHQ_blond_male.hyperparameters_main)
import hyp_hypscan_compas_dadv
z=merge(z, hyp_hypscan_compas_dadv.hyperparameters_main)
import hyp_hypscan_compas_inlp_DecreasingNN
z=merge(z, hyp_hypscan_compas_inlp_DecreasingNN.hyperparameters_main)

import hyp_hypscan_compas_inlp_fixed
z=merge(z, hyp_hypscan_compas_inlp_fixed.hyperparameters_main)
import hyp_hypscan_compass_bteo_decreasingnn
z=merge(z, hyp_hypscan_compass_bteo_decreasingnn.hyperparameters_main)

import hyp_hypscan_compass_bteo_fixed
z=merge(z, hyp_hypscan_compass_bteo_fixed.hyperparameters_main)
#import hyp_hypscan_compass_vanilla_LRPlateau
#z=merge(z, hyp_hypscan_compass_vanilla_LRPlateau.hyperparameters_main)
#import hyp_hypscan_compass_vanilla
#z=merge(z, hyp_hypscan_compass_vanilla.hyperparameters_main)
#import hyp_hypscan_compass_vanilla_StepLR
#z=merge(z, hyp_hypscan_compass_vanilla_StepLR.hyperparameters_main)
for k in z.keys():
#    print (f"############################################")
    if k == 'all' : continue
    if k == 'celeba_preprocessed' : continue
    kn=k.replace('_', ' ')
    kl=kn.lower()
    print("\\begin{small}")
    print("\\begin{landscape}")
    print("\\begin{table}[ht]")
    print("\\label{tab:", f"{kl}", "allhyp}")
    print("\\caption{Extended hyperparameters in the " + f"{kn}" + " dataset}")
    print('\\resizebox{\\textwidth}{!}{%')
    print("\\begin{tabular}{|l|llll|}")
    print( "\\hline")
    print ("\\textbf{Hyperparameter}          & \\textbf{Search space} \\\\")
    print()


    #print (f"Hyperparameter ranges for the {k} dataset")
    dskeys=[]
    
    for k2 in z[k].keys():
      if k2=='cdadv' : continue
      print(" & " , "\\textbf{",k2.replace('_', ' '),"}", end='')
      dskeys=list(set(z[k][k2].keys()) - set(dskeys))+dskeys

    print("\\\\")
    print ("\\hline")

    dskeys=sorted(dskeys)
    for hyp in dskeys:
        print('\\textbf{',hyp.replace('_', ' '),'}',end='')
        for k2 in z[k].keys():
           if k2=='cdadv' : continue
           text="-"
           if hyp in  sorted(z[k][k2].keys()):
              text=z[k][k2][hyp]
              
              try:
                 this=sorted(text)
                 

              except:
                 this=text
              out="["
              tot=0
              multicell=0
              first=1
              for word in this:
                  if first:
                      first=0
                  else:
                      out+=', '

                  if tot > 22:
                      tot=0
                      out+='\\\\'
                      multicell=1
                  tot+=len(str(word))+2
                  out+=str(word)
              out+=']'
              if multicell:
                  out='\\makecell{ ' + out +'}'
              print(" & ", out, end='')
           else:
                 print  (" & ", text,end='' )
        print ('\\\\' )
    print("\\hline")
    print("\\end{tabular}}")
    
    print("\\end{table}")
    print("\\end{landscape}")
    print("\\end{small}")
