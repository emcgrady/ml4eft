import pandas as pd
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import math
import torch
import argparse

parser = argparse.ArgumentParser(description='A script to train a neural network to train over the likelihood ratios of different EFT coefficients')

#Arguemnts that require command line input
parser.add_argument('--coef', default='SM*SM', help='The coefficient being used to train over')
parser.add_argument('--coord', default='px_py_pz', help='Coordinate system used for event data. Two options: px_py_pz pt_eta_phi')

#Arguments that can be adapted
parser.add_argument('--n_epochs', default=500, help='Number of epochs to train the network over')
parser.add_argument('--lr_i', default=2e-2, help='The initial learning rate for the nework')
parser.add_argument('--n_samples', default=1000000, help='Number of events to use from the dataset')
parser.add_argument('--t_size', default=0.9, help='Portion of events used to train the network')
parser.add_argument('--batch_size', default=100, help='Number of samples trained over at a time')
parser.add_argument('--factor', default=0.1, help='What to multiply the learning rate by upon adaptation')
parser.add_argument('--min_lr', default=1e-4, help='The lowest value the learning rate can reduce to')
parser.add_argument('--verbose', default=True, help='Print at what epoch and to what value the learning rate changes')

args = parser.parse_args()

coef = args.coef
coord = args.coord

n_samples = args.n_samples
n_epochs = args.n_epochs
batch_size = args.batch_size

lr = args.lr_i

factor = args.factor
patience=200
threshold=1e-4
threshold_mode='rel'
cooldown=25
min_lr = args.min_lr
verbose = args.verbose

if coord == 'px_py_pz':
    labels = ["Higgs $p_x$", "Higgs $p_y$", "Higgs $p_z$",
              "Top $p_x$", "Top $p_y$", "Top $p_z$",
              "Anti-Top $p_x$", "Anti-Top $p_y$", "Anti-Top $p_z$"]
    
if coord == 'pt_eta_phi':
    labels = ["Higgs $p_T$", "Higgs $\eta$", "Higgs $\phi$",
              "Top $p_T$", "Top $\eta$", "Top $phi$",
              "Anti-Top $p_T$", "Anti-Top $\eta$", "Anti-Top $\phi$"]
   
df = pd.read_feather('/scratch365/cmcgrad2/data/' + coord + 
                     '/dataframes/' + coef + '.feather').rename(columns={coef:'r_c'}).iloc[:n_samples,:]

t_size  = int(math.floor(n_samples*0.1))
inputs  = torch.tensor(df.iloc[t_size:,:9].values, dtype = torch.float32).cuda()
outputs = torch.tensor(df.iloc[t_size:, 9].values, dtype = torch.float32).cuda()
test    = torch.tensor(df.iloc[:t_size,:9].values, dtype = torch.float32).cuda()
corr    = torch.tensor(df.iloc[:t_size, 9].values, dtype = torch.float32).cuda()

model = torch.nn.Sequential(
    torch.nn.Linear(inputs.shape[1],500),
    torch.nn.ReLU(),
    torch.nn.Linear(500,500),
    torch.nn.ReLU(),
    torch.nn.Linear(500,1)
).cuda()

model.train()

numMiniBatch = int(math.floor(inputs.shape[0]/batch_size))
inputMiniBatches =inputs.chunk(numMiniBatch)
outputMiniBatches = outputs.chunk(numMiniBatch)

corrMiniBatches = corr.chunk(numMiniBatch)
testMiniBatches = test.chunk(numMiniBatch)

lossFunc = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters())

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=factor, patience=patience,
                                                       threshold=threshold, threshold_mode=threshold_mode,
                                                       cooldown=cooldown, min_lr=min_lr, verbose=verbose
                                                      )

lossIn = np.arange(1, n_epochs + 1, 1)
lossOut = []
lossTest = []

print('Starting training...')

for epoch in range(n_epochs):
    for minibatch in range(numMiniBatch):
        prediction = torch.squeeze(model(inputMiniBatches[minibatch]))
        testLoss = lossFunc(torch.squeeze(model(testMiniBatches[math.floor(minibatch/2)])),
                            corrMiniBatches[math.floor(minibatch/2)])
        loss = lossFunc(prediction,outputMiniBatches[minibatch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    lossOut.append(loss.tolist())
    scheduler.step(lossOut[epoch])
    lossTest.append(testLoss.tolist())

print('Traning complete!')

    
pred = model(test).cpu().detach()
res  = torch.abs(model(test).squeeze() - corr).cpu().detach()
corr = corr.cpu().detach()
test = test.cpu().detach()


plt.rcParams['figure.figsize'] = [15, 8]

plt.text(0.05, 0.4, 'Number of Epochs: ' + str("{:,}".format(n_epochs)) + '\n'
         + 'Initial Learning Rate: ' + str(lr) + '\n' 
         + 'Trainging Size: ' + str("{:,}".format(t_size)) + '\n'
         + 'Validation Size: ' + str("{:,}".format((n_samples - t_size))) + '\n'
         + 'Batch Size: ' + str(batch_size), fontsize=32, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 15})
plt.savefig(str('/scratch365/cmcgrad2/ml4eft/plots/' + coef + '/' + coord + '/network_performance/parameters.png'))
plt.clf()

plt.plot(lossIn,lossOut, 'b', label='Training')
plt.plot(lossIn,lossTest, 'tab:orange', label='Test')
plt.grid(b=True, color='grey', alpha=0.2, linestyle=':', linewidth=2)
plt.xlabel('Number of Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.yscale('log')
plt.title('Loss Function by Epoch (Log)', fontsize=16)
plt.legend(loc='best', fontsize=12)
plt.savefig(str('/scratch365/cmcgrad2/ml4eft/plots/' + coef + '/' + coord + '/network_performance/loss.png'))
plt.clf()

plt.plot(corr.numpy(), pred.numpy(), 'g.', label='Training Results')
plt.grid(b=True, color='grey', alpha=0.2, linestyle=':', linewidth=2)
plt.xlabel("Correct", fontsize=12)
plt.ylabel("Predicted", fontsize=12)
plt.title("Training Distribution Against Expected Results", fontsize=16)
plt.savefig(str('/scratch365/cmcgrad2/ml4eft/plots/' + coef + '/' + coord + '/network_performance/expected.png'))
plt.clf()

plt.hist(corr.numpy(), bins=100, label='Expected Results', color = 'blue', alpha=0.6)
plt.hist(pred.numpy(), bins=100, label='Training Results', color='green', alpha=0.6)
plt.grid(b=True, color='grey', alpha=0.2, linestyle=':', linewidth=2)
plt.yscale('log')
plt.xlabel("$r_c$", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Distribution of Results", fontsize=16)
plt.legend(loc="best", fontsize=12)
plt.savefig(str('/scratch365/cmcgrad2/ml4eft/plots/' + coef + '/' + coord + '/network_performance/distributions.png'))
plt.clf()

plt.hist((torch.abs(torch.div(res,corr))).numpy(), bins=100)
plt.grid(b=True, color='grey', alpha=0.2, linestyle=':', linewidth=2)
plt.yscale('log')
plt.xlabel("Residual", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Residual Distribution (Linear)", fontsize=16)
plt.savefig(str('plots/' + coef + '/' + coord + '/network_performance/residuals.png'))
plt.clf()
