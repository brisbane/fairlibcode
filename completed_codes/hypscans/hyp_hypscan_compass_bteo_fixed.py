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
            { 'BT' : ['Resampling', 'Reweighting'],
              'BTObj': ['EO'],
        
        'dropout': [0.0, 0.1],
        'lr': [0.0003, 0.003, 0.001],
        'batch_size':[ 1024, 512, 256],
        'weight_decay':[0.1,0.0],
             'encoder_architecture': ['Fixed'],
             'n_hidden' : [ 2],
             'hidden_size': [ 300 ],
             'softmax':[ False, True]
            }
         },
        "Bios_gender":
         {
             'vanilla':
             {
            'dropout': [0.1,0.2,0.3,0.5, 0.4],
            'lr': [0.00003, 0.0001,0.0003,0.001,0.003, 0.01],
            'batch_size':[512,256,128],
             }
         }
        }
