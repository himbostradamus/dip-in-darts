from .eval_lightning import LightningEval
from .phantom import generate_phantom


import torch
import nni

from lightning.pytorch import Trainer   
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


params = {
    'lr': .001,
    'num_iter': 100,
    'buffer_size': 50,
}

optimized_params = nni.get_next_parameter()
params.update(optimized_params)
print(params)

# input image (phantom)
resolution = 7
phantom = generate_phantom(resolution=resolution)

# model
# model = SimpleAutoencoder()
model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=1, out_channels=1, init_features=64, pretrained=False)

basic_early_stopping = EarlyStopping(
    monitor="variance", 
    mode="min", 
    patience=6, 
    verbose=True,
    min_delta=0
    )

# Create the lightning module
module = LightningEval(
                model, 
                phantom=phantom, 
                buffer_size=params['buffer_size'],
                num_iter=params['num_iter'],
                lr=params['lr'], 
                noise_type='gaussian', 
                noise_factor=0.15, 
                resolution=resolution, 
                )

# Create a PyTorch Lightning trainer
# trainer = Trainer(max_epochs=5)
trainer = Trainer(
            callbacks=[basic_early_stopping],
            max_epochs=100, # (max_epochs)*(num_iter) = (Total Iterations) ---> 85 * 50 = 4250 iterations
            # check_val_every_n_epoch=1 # check validation every epoch (num_iter) ---> check validation every 50 iterations
            )

# # Train the model
trainer.fit(module)

# epochs = 20
# for t in range(5,epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     trainer = Trainer(
#             callbacks=[basic_early_stopping],
#             max_epochs=20, # (max_epochs)*(num_iter) = (Total Iterations) ---> 85 * 50 = 4250 iterations
#             # check_val_every_n_epoch=1 # check validation every epoch (num_iter) ---> check validation every 50 iterations
#             )
#     trainer.fit(module)

epochs = 8
trainer = Trainer(
        callbacks=[basic_early_stopping],
        max_epochs=epochs, # (max_epochs)*(num_iter) = (Total Iterations) ---> 85 * 50 = 4250 iterations
        # check_val_every_n_epoch=1 # check validation every epoch (num_iter) ---> check validation every 'num_iter' iterations
        )
trainer.fit(module)