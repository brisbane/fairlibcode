hyperparameters_main={'all' :
        {
        'dropout': [0.0,0.2,0.3,0.41,0.39,0.5],
        'lr': [0.000001,0.000003,0.00001,0.00003,0.0001,0.0003,0.001,0.003, 0.01,0.03,0.1],
        'batch_size':[ 2048,1024,512,256,128]
        },
        "COMPAS_race":
        {
#        'dropout': [0.4,0.3, 0.5,0.7,0.9],
#        'lr': [0.1,0.2,0.05,0.5,1],
#        'batch_size':[ 4096, 2048, 1024],
#        'weight_decay':[0.5, 0.7,1,2,10,100]
        'dropout': [0.0,0.1, 0.2, 0.5,0.75, 0.9],

        'lr': [0.01,0.1,1,0.3, 0.03],
        'lr_scheduler': ["ReduceLROnPlateau", "StepLR", "default"],
        'batch_size':[  1024, 2048, 512],
        'weight_decay':[0.0,0.3, 0.1],
        'n_hidden': [5,4],
        #Default is false
        'softmax':  [True, False]
            },
        "Bios_gender":
         {
        'dropout': [0.1,0.2,0.3,0.5, 0.4],
        'lr': [0.00003, 0.0001,0.0003,0.001,0.003, 0.01],
        'batch_size':[512,256,128],

         }
        }
