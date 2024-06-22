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
            { 'BT' : ['Reweighting', 'Resampling', "Downsampling"],
              'BTObj': ['EO'],
              'dropout': [0.3],
              'lr': [0.001],
              #'batch_size':[512,256],
              'batch_size':[128],
              'weight_decay':[0.5],
              #'weight_decay':[0.0,0.1,0.5],
              'lr_scheduler':["ReduceLROnPlateau", "StepLR", "default"]
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
            'cdadv':
            { 'adv_corr_loss' : [True],
#              'adv_diverse_lambda': [1,10,100,1000],
              'adv_diverse_lambda': [10],
#              'adv_lambda': [0.1,0.3,0.8,1.0,3.0],
              'adv_lambda': [0.8],
              'adv_num_subDiscriminator' : [5],
              'dropout': [0.2],
              'lr': [0.001],
              'batch_size':[512],
              'batch_norm':[True, ]
            },
         },

         'celebMHQ':
         {
             'vanilla':
             {
                 #was up to 512
                'batch_size':[32,64,128],
                #was up to 0.5
                'dropout': [0.0,0.1,0.2],
#was 0.0001 to 1 
                'lr': [  0.003, 0.01,0.03],
                'weight_decay':[0.0,0.1],
                'lr_scheduler':["StepLR", "default"],
#                'softmax':[False],
             },
             'dadv':
             {
                 #was up to 512
                'batch_size':[32,64,128],
                #was up to 0.5
                'dropout': [0.0,0.1,0.2],
#was 0.0001 to 1 
                'lr': [  0.001, 0.003, 0.01,0.03, 0.1],
                'weight_decay':[0.0,0.1,0.2],
                'lr_scheduler':["StepLR", "default"],
                'adv_corr_loss' : [False],
#              'adv_diverse_lambda': [1,10,100,1000],
                'adv_diverse_lambda': [10,100,1000],
#              'adv_lambda': [0.1,0.3,0.8,1.0,3.0],
               'adv_lambda': [0.8,1.0,3.0],
               'adv_num_subDiscriminator' : [1, 3, 5],
             },
            'BTEO':
            { 'BT' : ['Reweighting', 'Resampling', "Downsampling"],
              'BTObj': ['EO'],
              'dropout': [0.0,0.1],
              'lr': [0.001, 0.01, 0.1],
              #'batch_size':[512,256],
              'batch_size':[64,128],
              'softmax':[False],
              'weight_decay':[0.1,0.0],
              #'weight_decay':[0.0,0.1,0.5],
              'lr_scheduler':["ReduceLROnPlateau", "StepLR", "default"]
            },

         },

         'celeba_preprocessed':
         {
             'vanilla':
             {
                 #was up to 512
                'batch_size':[32,64,128],
                #was up to 0.5
                'dropout': [0.0,0.1,],
#was 0.0001 to 1 
                'lr': [  0.003, 0.01,0.03],
                'weight_decay':[0.0,0.1,0.3],
                'lr_scheduler':["StepLR", "default"]
             },
             'dadv':
             {
                 #was up to 512
                'batch_size':[32,64,128],
                #was up to 0.5
                'dropout': [0.0,0.1,0.2],
#was 0.0001 to 1 
                'lr': [  0.001, 0.003, 0.01,0.03, 0.1],
                'weight_decay':[0.0,0.1,0.2],
                'lr_scheduler':["StepLR", "default"],
                'adv_corr_loss' : [False],
#              'adv_diverse_lambda': [1,10,100,1000],
                'adv_diverse_lambda': [10,100,1000],
#              'adv_lambda': [0.1,0.3,0.8,1.0,3.0],
               'adv_lambda': [0.8,1.0,3.0],
               'adv_num_subDiscriminator' : [1, 3, 5],
             },
            'BTEO':
            { 'BT' : ['Reweighting', 'Resampling', "Downsampling"],
              'BTObj': ['EO'],
              'dropout': [0.0,0.1],
              'lr': [0.001, 0.01, 0.1],
              #'batch_size':[512,256],
              'batch_size':[64,128],
              'softmax':[False],
              'weight_decay':[0.1,0.0],
              #'weight_decay':[0.0,0.1,0.5],
              'lr_scheduler':["ReduceLROnPlateau", "StepLR", "default"]
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
