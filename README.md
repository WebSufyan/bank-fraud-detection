# bank fraud detection
 detecting fraud transactions


we have:
284315 normal transactions.
492 fraud transactions.

my approach to this problem would be to exclude the fraud transactions and implement an autoencoder
then train it only on normal transactions

the model will learn to recognize what are the normal transatctions

we will use cost function for prediction
In the testing phase we will feed the model a test set contains both fraud and normal
then see how it will perform


how will we determine if a transaction is fraud or not? with cost function.
if an unfamiliar transaction gets fed to our model, the model will try to reconstruct it and fails
thus the model will give us a higher cost function, since our model is trained only on normal 
transactions, so higher cost function means an unfamiliar transactions therefore
it will be more likely a fraud transaction.






######### training and epochs #########

training loss: 2904.607421875 - epoch number 0

training loss: 290.11083984375 - epoch number 2

training loss: 1.6810399293899536 - epoch number 4

training loss: 1.8805577754974365 - epoch number 6

training loss: 0.814797043800354 - epoch number 8

training loss: 1.5862817764282227 - epoch number 10

training loss: 0.724291205406189 - epoch number 12

training loss: 1.0699563026428223 - epoch number 14

training loss: 1.1471915245056152 - epoch number 16

training loss: 1.0960371494293213 - epoch number 18

training loss: 1.3729445934295654 - epoch number 20

training loss: 1.0244572162628174 - epoch number 22

training loss: 0.6073285341262817 - epoch number 24

training loss: 0.9225599765777588 - epoch number 26

training loss: 0.7937610149383545 - epoch number 28

cost function is decreasing which is a good sign.





########## trying out dfferent cost function threshold and measuring accuracy on a data tha model never seen ##############

threshold: 0.4 - accuracy: 64.43%

threshold: 0.5 - accuracy: 70.12%

threshold: 0.6 - accuracy: 74.80%

threshold: 0.7 - accuracy: 80.28%

threshold: 0.8 - accuracy: 83.74%

threshold: 0.9 - accuracy: 87.60%

threshold: 1 - accuracy: 88.92%

threshold: 1.5 - accuracy: 91.57%

threshold: 1.8 - accuracy: 91.26%

threshold: 2 - accuracy: 91.36%

threshold: 2.5 - accuracy: 90.55%

threshold: 3 - accuracy: 90.14%

threshold: 3.4 - accuracy: 90.14%






######## summary ##########

training was done on my GPU which is RTX 2070 8GB VRAM

Training time in seconds: 673.
Training time in minutes: 11.22.

The best threshold for cost function according to my experiments is: 2 

This model detects fraud transactions with accuracy of 91.36%




###### link to dataset #######
https://www.kaggle.com/mlg-ulb/creditcardfraud




p.s: since data is too large to push on github, i had to delete it. if you are interested in running my code please donwload data from link above and put it on main folder of code.
p.s 2: in the training.py file i recommend executing code phase by phase, if you run all code at once you will get an error related to pytorch dataloader object that is related
GPU out of memory errors. or try lower batch_size



