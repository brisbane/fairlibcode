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
            },
            'inlp' :
            {
              'INLP_by_class' : [ True],
              'INLP_discriminator_reweighting' : [None],
              'INLP_min_acc' : [0.5],
              'INLP_n':  [50],
              'dropout': [0.1],
              'batch_size':[4096],
              'weight_decay':[0.0],
#              'lr': [0.03, 0.01, 0.003],
              'lr': [ 0.1],
              'lr_scheduler': [ "default"],
              'softmax': [False],
           #    "encoder_architecture": ["DecreasingNN"],
           #    "hidden_size": [4],
           #    "n_hidden": [4]
               "encoder_architecture": ["Fixed"],
               "hidden_size": [300],
               "n_hidden": [2]
            }
         },
        "adult":
        {

        'vanilla': {
        'dropout': [0.4,0.1, 0.0],
        'lr': [0.001, 0.003, 0.01, 0.03, 0.1],
        'batch_size':[ 4096, 2048, 1024],
        'weight_decay':[0.0, 0.1, 0.3],
        'lr_scheduler':["ReduceLROnPlateau", "StepLR", "default"]
            },
        'dadv': {
        'dropout': [0.1, 0.0],
        'lr': [0.0003,0.001,  0.01],
        'batch_size':[ 4096, 2048],
        'weight_decay':[0.0, 0.1],
        'lr_scheduler':["ReduceLROnPlateau", "StepLR", "default"],
                'adv_corr_loss' : [False],
#              'adv_diverse_lambda': [1,10,100,1000],
                'adv_diverse_lambda': [0.1,10],
#              'adv_lambda': [0.1,0.3,0.8,1.0,3.0],
               'adv_lambda': [0.1,10],
               'adv_num_subDiscriminator' : [3, 5],
              'softmax':[False],
            },

            'BTEO':
            { 'BT' : ['Reweighting', 'Resampling', 'Downsampling'],
              'BTObj': ['EO'],
        'dropout': [0.1, 0.0],
        'lr': [   0.1, 0.2, 0.06, ],
        'batch_size':[ 4096, 2048],
        'weight_decay':[0.0, 0.1],
        'lr_scheduler':["ReduceLROnPlateau", "StepLR", "default"]
            },
            'inlp' :
            {
              'INLP_by_class' : [ True],
              'INLP_discriminator_reweighting' : [None],
              'INLP_min_acc' : [0.5],
              'INLP_n':  [50],
              'dropout': [0.0, 0.1],
              'batch_size':[4096],
              'weight_decay':[0.0, 0.1],
#              'lr': [0.03, 0.01, 0.003],
              'lr': [ 0.01, 0.003],
              'lr_scheduler': ['ReduceLROnPlateau', "default"],
              'softmax': [False],
           #    "encoder_architecture": ["DecreasingNN"],
           #    "hidden_size": [4],
           #    "n_hidden": [4]
               "encoder_architecture": ["Fixed"],
               "hidden_size": [300],
               "n_hidden": [2]
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
           'inlp':
            { 
#              'INLP_by_class' : [True, False],
#              'INLP_discriminator_reweighting' : [None,'balanced'],
#              'INLP_min_acc' : [ 0.0, 0.5, 0.2],
#              'INLP_n':  [150,300,50],
              'INLP_by_class' : [ True],
              'INLP_discriminator_reweighting' : [None],
              'INLP_min_acc' : [0.5],
              'INLP_n':  [50],
             'dropout': [0.4],
              'batch_size':[512],
              'weight_decay':[0.0],
              'lr': [0.001],
              'lr_scheduler': ['default'],
              'softmax': [False]

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
                'dropout': [0.0,0.1,],
#was 0.0001 to 1 
                'lr': [  0.003, 0.01,0.03],
                'weight_decay':[0.0,0.1,0.3],
                'lr_scheduler':["StepLR", "default"]
             },
             'dadv':
             {
                 #was up to 512
                'batch_size':[128,256],
                #was up to 0.5
                'dropout': [0.0,0.1,0.2],
#was 0.0001 to 1 
                'lr': [  0.0003],
                #was up to 0.2

                'weight_decay':[0.0,0.1],
                'lr_scheduler':["StepLR", "default"],
                'adv_corr_loss' : [False],
#              'adv_diverse_lambda': [1,10,100,1000],
                'adv_diverse_lambda': [1,10],
#              'adv_lambda': [0.1,0.3,0.8,1.0,3.0],
               'adv_lambda': [0.1,0.8],
               'adv_num_subDiscriminator' : [3, 5],
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
           'inlp':
            { 
#              'INLP_by_class' : [True, False],
#              'INLP_discriminator_reweighting' : [None,'balanced'],
#              'INLP_min_acc' : [ 0.0, 0.5, 0.2],
#              'INLP_n':  [150,300,50],
              'INLP_by_class' : [ True],
              'INLP_discriminator_reweighting' : [None],
              'INLP_min_acc' : [0.5],
              'INLP_n':  [50],
              'softmax':[True],
              'batch_size':[2048],
              'dropout': [0.1],
              'lr': [ 0.01],
              'weight_decay':[0.5],
              #'lr_scheduler':[ "default"]
             }

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
                'batch_size':[64],
#                'dropout': [0.1,0.2,0.3,0.4,0.5],
#                'lr': [ 0.001, 0.003, 0.01,0.03],
                'lr': [ 3,10],
                'dropout': [0.7,0.8],
                #'weight_decay':[0.0,0.2,0.4,0.6,0.8],
                'weight_decay':[0.0,0.1],
             },
             'inlp':
             {
                'batch_size':[64],
                'dropout': [0.5],
                'lr': [3],
                'weight_decay':[0.0],
                'softmax':[False],
                #output layer -1 is just the resnet embedding?
                'n_hidden': [0],
                #so need to use the resnet embedding size (is that 512, thought it was 1000)
                'hidden_size': [512],
              'INLP_by_class' : [ True],
              'INLP_discriminator_reweighting' : [None],
              'INLP_min_acc' : [0.0],
              'INLP_n':  [50],

             }
        
        }
}
