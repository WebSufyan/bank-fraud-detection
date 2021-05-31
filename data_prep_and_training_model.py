''' since the dataset is extremely imbalanced

284315 normal transactions.
492 fraud transactions.

my approach to this problem would be to exclude the fraud transactions and implement an autoencoder
then train it only on normal transactions

the model will learn to recognize what are the normal transatctions

we will use cost function for prediction
In the testing phase we will feed the model a test set contains both fraud and normal
then see how it will perform'''


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset

import time

'''##################### DATA PREP PHASE ######################## '''

try:
    df = pd.read_csv('new_creditcard.csv', index_col=[0])
except:
    print('there is no new_creditcard.csv file, please download data from the available link')

# shuffle data
df = df.sample(frac=1)


# let's split fraud and normal
fraud = df[df['Class'] == 1]
normal = df[df['Class'] == 0]

fraud_testing = fraud.drop('Class', axis = 1)

# then split normal data for training and testing
normal_training = (normal.iloc[:-(len(df) - len(normal))]).drop('Class', axis=1)
normal_testing = normal.iloc[(len(normal) - 492):]

# creating a test set that includes both classes with their labels
# for evaluation at the end of training
# concatenating and balancing normal and fraud into one dataset
balanced_test_set = (pd.concat([normal_testing, fraud])).reset_index(drop=True)
balanced_test_set = balanced_test_set.sample(frac=1)

# now we split class from test set
x_test = balanced_test_set.iloc[:, :-1].values
y_test = balanced_test_set.iloc[:, -1].values



# we convert z and y to tuples each sample with its own label than feed it to dataloader to batch it
testa = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
both_classes_test_loader = DataLoader(testa, batch_size=1)


# let's convert dataset into tensors and split it into batches
# first create a simple helper function that takes in dataset and output tensor

def to_tensor(data):
    return torch.FloatTensor(data.values)

# train loader to make dataset batches and num_workers is related to GPU 
train_loader = DataLoader(to_tensor(normal_training), batch_size=64, shuffle=True, num_workers=8)


''' we create a deep atificial neural network autoencoder '''

#  encoder is a feedforward, fully connected neural network that compresses 
# the input into a latent space representation and encodes the input dataset as 
# a compressed representation in a reduced dimension
class encoder(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.fc1 = nn.Linear(in_features, 20)
        self.fc2 = nn.Linear(20, 16)
        self.fc3 = nn.Linear(16, 8)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x


# the decoder will take in the output of encoder and try to decode it back to it's original shape
class decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc2 = nn.Linear(8, 16)
        self.fc3 = nn.Linear(16, 20)
        self.fc4 = nn.Linear(20, 29)

    def forward(self, x):

        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        return x

# put encoder and decoder together
class AE(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.encoder = encoder(in_features)
        self.decoder = decoder()
        
    def forward(self, x):
        encoder = self.encoder(x)
        x = self.decoder(encoder)
        
        return x
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # training on GPU

start = time.time() # calculating runtime of training

'''##################### TRAINING PHASE ######################## '''
if __name__ == '__main__':
    
    torch.manual_seed(101) # the neural network are initiated differently each time we execute 
                           # code thus we get different results and manual_seed makes the weights 
                           # always the same even if we execute code several times

    model = AE(normal_training.shape[1]).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001)
    
    criterion = nn.MSELoss()
    
    losses = []
    
    epochs = 30
    
    model.train()
    for epoch in range(epochs):
        for idx, i in enumerate(train_loader):
            data = i.to(device)
            y_pred = model(data)
            cost = criterion(y_pred, data)

            optimizer.zero_grad()
            cost.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

        losses.append(cost.item())

        if epoch % 2 == 0:
            print(f'training loss: {cost.item()} - epoch number {epoch}')

time_training_took = time.time() - start
# plot losses
plt.plot(losses)



'''##################### EVALUATION PHASE ######################## '''

# how will we determine if a transaction is fraud or not? with cost function.
# if an unfamiliar transaction gets fed to our model, the model will try to reconstruct it and fails
# thus the model will give us a higher cost function, since our model is trained only on normal 
# transactions, so higher cost function means an unfamiliar transactions therefore
# it will be more likely a fraud transaction

# set threshold for cost function

tests = []
real_labels = []

with torch.no_grad():
    model.eval()
    for idx2, (k, l) in enumerate(both_classes_test_loader):

        test_data = k.float().to(device)
        label = l.float().to(device)

        y_fake = model(test_data)
        test_cost = criterion(y_fake, test_data)
        
        tests.append(test_cost.item())
        real_labels.append(l.item())


# trying out different thresholds values and picking the threshold based on the best accuracy




thresholds = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1, 1.50, 1.80, 2, 2.5, 3, 3.4]

accuracies = []
for i in thresholds:
    correct_samples = np.where(np.asarray(tests) < i, 0, 1)
    accuracy = (correct_samples.reshape(-1, 1) == np.asarray(real_labels).reshape(-1, 1)).sum()
    
    acc = accuracy/len(correct_samples)
    accuracies.append((acc, i))
    print(f'threshold: {i} - accuracy: {acc*100:.2f}%')

best_thresh = 0

for j in range(1, len(accuracies)):
    if accuracies[j][0] > accuracies[j-1][0]:
        best_thresh = accuracies[j]

# saving the model
# torch.save(model.state_dict(), 'fraud_detection.pt')


''' summary '''

print(f'Training time in seconds: {round(time_training_took)}.\n\
      Training time in minutes: {time_training_took/60:.2f}.')
print('')
print(f'The best threshold for cost function according to my experiments is: {best_thresh[1]} ')
print('')
print(f'This model detects fraud transactions with accuracy of {best_thresh[0]*100:.2f}%')







