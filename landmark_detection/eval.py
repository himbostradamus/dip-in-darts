import torch
import numpy as np
import cnn6 as net
import CERDataset
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from training import EvalLandmark

from nni.retiarii.evaluator.pytorch import Lightning, Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

# mode settings for mode = 'pretrain2'
mode = 'pretrain2'
source_dir = 'TrainingImages2'
source_name = 'image_'
landmarks_dir = 'TrainingLabels1'
landmarks_name = 'labels_'
target_dir = 'DetectedLandmarks2'
target_name = 'imageDetected_'
params_name = '2pretrain_params.pt'
train_loss_name = '2pretrain_train_loss.npy'
val_loss_name = '2pretrain_val_loss.npy'
loss_fig = '2loss_Pretraining.png'
loss_file = '2loss.txt'

# landmarks
numOfLandmarks = 4

# settings for pytorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)

model = net.CNN()
model.to(device)
model.load_state_dict(torch.load(params_name,map_location=torch.device(device)))
# model.eval()

# load and shuffle data and define train/test dataloader
full_data = CERDataset.CERDataset(landmark_dir=landmarks_dir, image_dir=source_dir)
train_data_size = int(0.72 * len(full_data))
test_data_size = int(0.18 * len(full_data))
valid_data_size = len(full_data) - train_data_size - test_data_size
assert train_data_size + test_data_size + valid_data_size == len(full_data)
train_data, test_data, valid_data = torch.utils.data.random_split(full_data,
                                                                  [train_data_size, test_data_size, valid_data_size],
                                                                  generator=torch.Generator().manual_seed(42))

train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=8, shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size=8, shuffle=False)

module = EvalLandmark(mode=mode)
early_stop_callback = EarlyStopping(
            monitor="validation_loss", 
            min_delta=0.00, 
            patience=200, 
            verbose=False, 
            mode="max"
            )
trainer = Trainer(
            max_epochs=10000,
            fast_dev_run=False,
            gpus=1,
            callbacks=[early_stop_callback],
            )
trainer.fit(model, train_dataloader, valid_dataloader)

# construct fail example
'''

    phantom = np.zeros((128,128))
    phantom[60:100,60:100] = 1
    phantom_pt = torch.from_numpy(phantom)
    phantom_pt = torch.unsqueeze(phantom_pt, 0)
    landmarks_detected = model(torch.unsqueeze(phantom_pt, 0).to(device)).detach().numpy()
    landmarks_detected = np.resize(landmarks_detected, (numOfLandmarks,2))
    plt.imshow(phantom, cmap='gray', vmin=0, vmax=1)
    plt.scatter(landmarks_detected[:, 0], landmarks_detected[:, 1], color='red', s=1, label='detected landmarks')
    plt.savefig('fail.png')
    plt.close()

'''

for i in test_data.indices[:10]:
    image = torch.load(source_dir + '/' + source_name + str(i+1) + '.pt')
    landmarks = torch.load(landmarks_dir + '/' + landmarks_name + str(i+1) + '.pt').numpy()
    landmarks_detected = model(torch.unsqueeze(image, 0).to(device)).detach().numpy()

    landmarks = np.resize(landmarks, (numOfLandmarks, 2))
    landmarks_detected = np.resize(landmarks_detected, (numOfLandmarks,2))

    #print(landmarks_detected)
    plt.imshow(image.numpy().squeeze(), cmap='gray', vmin=0, vmax=1)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], color='green', s=1, label='correct landmarks')
    plt.scatter(landmarks_detected[:, 0], landmarks_detected[:, 1], color='red', s=1, label='detected landmarks')
    plt.savefig(target_dir + '/' + target_name + str(i+1) + '.png')
    plt.close()

if mode != 'pretrain1b':
    # plot loss
    train_loss = np.load(train_loss_name)
    val_loss = np.load(val_loss_name)
    startepoch = 0
    x = np.arange(startepoch,len(train_loss))
    plt.figure(figsize=(7,5))
    plt.semilogy(x, train_loss[startepoch:], label='training loss')
    plt.semilogy(x, val_loss[startepoch:], label='validation loss')
    plt.xlabel('training epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(loss_fig)
    plt.close()

    # calculate overall loss on the datasets

    validation_loss = validation_loop(valid_dataloader, model, torch.nn.MSELoss())
    train_loss = validation_loop(train_dataloader, model, torch.nn.MSELoss())
    test_loss = validation_loop(test_dataloader, model, torch.nn.MSELoss())

    f = open(loss_file, "w")
    f.write('validation_loss:' + str(validation_loss) + '\n')
    f.write('train_loss:' + str(train_loss) + '\n')
    f.write('test_loss:' + str(test_loss))
    f.close()

print('done')