# %%
import torch
import numpy as np
from lightning.pytorch import Trainer
from attr import evolve
from model.vicreg import SelfSupervisedMethod
from model.model_params import VICRegParams
from model.dataload import CustomDataloader
from model.utils import get_random_file

import warnings
warnings.filterwarnings("ignore")

# %%
# speedup
torch.set_float32_matmul_precision('high')

# set your data root folder
data_root = "data/data_segments/"
# data parameters
data_params = list()

# %%
# 0 Beef
data_params.append({
    'batch_size':30,
    'num_clusters': 5,
    'train_path': data_root + "Beef/train",
    'test_path' : data_root + "Beef/test",
    'checkpoint': 'checkpoint_beef'
    })
# 1 dist.phal.outl.agegroup
data_params.append({
    'batch_size':139,
    'num_clusters': 3,
    'train_path' : data_root + "DistalPhalanxOutlineAgeGroup/train",
    'test_path' :  data_root + "DistalPhalanxOutlineAgeGroup/test",
    'checkpoint': 'checkpoint_dist_agegroup'
    })
# 2 ECG200
data_params.append({
    'batch_size':100,
    'num_clusters': 2,
    'train_path' : data_root + "ECG200/train",
    'test_path' :  data_root + "ECG200/test",
    'checkpoint': 'checkpoint_ecg200'
    })
# 3 ECGFiveDays
data_params.append({
    'batch_size':23,
    'num_clusters': 2,
    'train_path' : data_root + "ECGFiveDays/train",
    'test_path' : data_root + "ECGFiveDays/test",
    'checkpoint': 'checkpoint_ecg5days'
    })
# 4 Meat
data_params.append({
    'batch_size':60,
    'num_clusters': 3,
    'train_path' : data_root + "Meat/train",
    'test_path' : data_root + "Meat/test",
    'checkpoint': 'checkpoint_meat'
    })
# 5 mote strain
data_params.append({
    'batch_size': 20,
    'num_clusters': 2,
    'train_path' : data_root + "MoteStrain/train",
    'test_path' : data_root + "MoteStrain/test",
    'checkpoint': 'checkpoint_motestrain'
    })
# 6 osuleaf
data_params.append({
    'resize': 428, 
    'batch_size': 64, # 200
    'num_clusters': 6,
    'train_path' : data_root + "OSULeaf/train",
    'test_path' : data_root + "OSULeaf/test",
    'checkpoint': 'checkpoint_osuleaf'
    })
# 7 plane
data_params.append({
    'batch_size': 105,
    'num_clusters': 7,
    'train_path' : data_root + "Plane/train",
    'test_path' : data_root + "Plane/test",
    'checkpoint': 'checkpoint_plane'
    })
# 8 proximal_agegroup
data_params.append({
    'batch_size': 205,
    'num_clusters': 3,
    'train_path' : data_root + "ProximalPhalanxOutlineAgeGroup/train",
    'test_path' : data_root + "ProximalPhalanxOutlineAgeGroup/test",
    'checkpoint': 'checkpoint_prox_agegroup'
    })
# 9 proximal_tw
data_params.append({
    'batch_size': 100, # 400
    'num_clusters': 6,
    'train_path' : data_root + "ProximalPhalanxTW/train",
    'test_path' : data_root + "ProximalPhalanxTW/test",
    'checkpoint': 'checkpoint_prox_tw'
    })

# %%
# inclusive
start = 0
end = 9

repeats = 1

for selector in range(start,end+1):
    # load parameters
    num_epochs=30
    train_path = data_params[selector]['train_path']
    random_file = get_random_file(train_path)
    img_size = len(np.load(random_file)['data'])
    
    def main():

        configs = {
            "vicreg": evolve(VICRegParams(), 
                            encoder_arch = "ws_resnet18", # resnet18, resnet34, resnet50
                            max_epochs=num_epochs
                            ),
        }

        for reruns in range(repeats): # number of repeats
            for name, config in configs.items():


                method = SelfSupervisedMethod(config)

                trainer = Trainer(accelerator="gpu", 
                                devices=[0], 
                                max_epochs=num_epochs,
                                logger=False)
                                # strategy=DDPStrategy(find_unused_parameters=False)) # to enable multi-gpu, but not necessary for now

                print("--------------------------------------")
                print(data_params[selector]['checkpoint'])


                train_loader = CustomDataloader(img_dir=data_params[selector]['train_path'],
                                                img_size=img_size,
                                                batch_size=data_params[selector]['batch_size'],
                                                ).get_dataloader()
                # val_loader = CustomDataloader(img_dir=data_params[selector]['test_path'],
                #                                 img_size=img_size,
                #                                 batch_size=data_params[selector]['batch_size'],
                #                                 ).get_dataloader()

                trainer.fit(model=method, train_dataloaders=train_loader)
            trainer.save_checkpoint('checkpoints/' + data_params[selector]['checkpoint'] + f'/last_{reruns}.ckpt')

    
    if __name__ == "__main__":
        main()

# %%
