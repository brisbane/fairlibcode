#!/usr/bin/env python3
from fairlib import datasets
#####hyperparameters
bert_batch_size=512
#######################
#datasets.prepare_dataset("bios", "./data/bios")
BIOS=False
MNIST=False
COCO=False
BFFHQ=False
CelebA_bm=False
CelebA=False
Adult=False
COMPAS=False
if BIOS:
  bios = datasets.bios.Bios(dest_folder="/home/user/miniconda3/datatest/bios", batch_size=bert_batch_size)
  bios.download_files()
elif MNIST:
#      ds = datasets.coloredMNIST.MNIST(dest_folder="/home/user/miniconda3/datatest/coloredmnist", batch_size=bert_batch_size)
 #     ds.download_files()
      datasets.prepare_dataset("coloredmnist", "/home/user/miniconda3/data/coloredmnist")
elif COCO:
  #    except:
   #        raise( "note impl")
#      datasets.prepare_dataset("coloredmnist", "/home/user/miniconda3/data/coloredmnist")
 #     ds = datasets.MSCOCO.COCO(dest_folder="/home/user/miniconda3/datatest/coco", batch_size=bert_batch_size)
#      ds.download_files()
    datasets.prepare_dataset("coco", "/home/user/miniconda3/data/coco")
elif BFFHQ:
    datasets.prepare_dataset("bffhq", "/home/user/miniconda3/data/bffhq")
elif CelebA:
    datasets.prepare_dataset("celeba", "/home/user/miniconda3/data/celeba")
elif CelebA_bm:
    datasets.prepare_dataset("celeba_bm", "/home/user/miniconda3/data/celeba_bm")
elif Adult:
    datasets.prepare_dataset("adult", "/home/user/miniconda3/data/adult")
elif COMPAS:
    datasets.prepare_dataset("COMPAS", "/home/user/miniconda3/data/compass_data")
#compas is done seperately
else:
  moji = datasets.moji.Moji(dest_folder="/home/user/miniconda3/datatest/moji")
  moji.download_files()
