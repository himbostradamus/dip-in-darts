from nni import trace
import nni
import nni.retiarii.strategy as strategy
import nni.retiarii.serializer as serializer

from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
from nni.retiarii.evaluator.pytorch import Lightning, Trainer
from nni.retiarii.evaluator.pytorch.lightning import DataLoader
from nni.retiarii.strategy import DARTS as DartsStrategy

import torch

# import sys to import from different directory
import sys
sys.path.insert(1, '/home/joe/nas-for-dip/')
from search_eval.eval_generic import SGLDES
from search_eval.optimizer.SingleImageDataset import SingleImageDataset
from search_eval.utils.common_utils import *
from search_space.search_space import SearchSpace

torch.cuda.empty_cache()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
print('CUDA available: {}'.format(torch.cuda.is_available()))

# Select the Search Strategy
# strategy = DartsStrategy()
# strategy = strategy.DartsStrategy()
# strategy = strategy.ENAS()
strategy = strategy.GumbelDARTS()
# strategy = strategy.RandomOneShot()

params={
    'max_epochs': 25000,
    'learning_rate': 0.07,
    'buffer_size': 100,
    'patience': 1000,
    'weight_decay': 5e-7,
}
optimized_params = nni.get_next_parameter()
params.update(optimized_params)
print(params)


resolution = 64
noise_type = 'gaussian'
noise_level = '0.09'
phantom =       np.load(f'/home/joe/nas-for-dip/phantoms/ground_truth/{resolution}/{45}.npy')
phantom_noisy = np.load(f'/home/joe/nas-for-dip/phantoms/{noise_type}/res_{resolution}/nl_{noise_level}/p_{45}.npy')

show_every = 1000
report_every = 250

# Create the lightning module
module = SGLDES(
                phantom=phantom, 
                phantom_noisy=phantom_noisy,
                
                learning_rate=params['learning_rate'], 
                buffer_size=params['buffer_size'],
                patience=params['patience'],
                weight_decay= params['weight_decay'],

                show_every=show_every,
                report_every=report_every,
                HPO=True,
                NAS=True,
                OneShot=True,
                SGLD_regularize=False
                )

# Create a PyTorch Lightning trainer
trainer = Trainer(
            max_epochs=params['max_epochs'],
            fast_dev_run=False,
            gpus=1,
            )
            
if not hasattr(trainer, 'optimizer_frequencies'):
    trainer.optimizer_frequencies = []


# Create the lighting object for evaluator
train_loader = DataLoader(SingleImageDataset(phantom, num_iter=1), batch_size=1)
val_loader = DataLoader(SingleImageDataset(phantom, num_iter=1), batch_size=1)

lightning = Lightning(lightning_module=module, trainer=trainer, train_dataloaders=train_loader, val_dataloaders=val_loader)


# Create a Search Space
model_space = SearchSpace(depth=4)

# fast_dev_run=False

config = RetiariiExeConfig(execution_engine='oneshot')
experiment = RetiariiExperiment(model_space, evaluator=lightning, strategy=strategy)
experiment.run(config)

print(f'\n\n----------------------------------------------------------------\n')
print(f'Top Models: {experiment.export_top_models()}')
print(f'\n\n----------------------------------------------------------------\n')