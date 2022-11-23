import pandas as pd
import numpy as np
import torch
import argparse
import math
import matplotlib.pyplot as plt
from functions import dnnPlots, train, sliced, mom_dot, norm, shat_filter

parser = argparse.ArgumentParser(description='A script to train a neural network to train over the likelihood ratios of different EFT coefficients')

#Arguemnts that filter events
parser.add_argument('--datatype', default='lhe', help='The branch that was used in nanoAOD for df generation')
parser.add_argument('--coef', default='ctp*SM', help='The coefficient being used to train over')
parser.add_argument('--coord', default='px_py_pz', help='Coordinate system used for event data. Two options: px_py_pz pt_eta_phi')
parser.add_argument('--int_part', default='uubar', help='Initail state particles to include')
parser.add_argument('--jets', default=False, help='set to True to include extra jets')
parser.add_argument('--redundant', default=True, help='set to False to get rid of any redundant inputs')

#Arguments that change the training variables
parser.add_argument('--n_epochs',    '-e', default=100, help='Number of epochs to train the network over')
parser.add_argument('--lr_i',        '-i', default=1e-3, help='The initial learning rate for the nework')
parser.add_argument('--n_nodes',     '-n', default=600, help='number of nodes per layer in the network')
parser.add_argument('--n_samples',   '-s', default=1000000, help='Number of events to use from the dataset')
parser.add_argument('--t_size',      '-t', default=0.9, help='Portion of events used to train the network')
parser.add_argument('--batch_size',  '-b', default=100, help='Number of samples trained over at a time')
parser.add_argument('--factor',      '-f', default=0.1, help='What to multiply the learning rate by upon adaptation')
parser.add_argument('--min_lr',      '-m', default=1e-8, help='The lowest value the learning rate can reduce to')
parser.add_argument('--patience',    '-p', default=50, help='Number of epochs before changing the learning rate')
parser.add_argument('--threshold',   '-o', default=1e-4, help='Threshold for learning rate adaptation')
parser.add_argument('--cooldown',    '-c', default=25, help='Number of epochs before starting the patience count again')
parser.add_argument('--verbose',     '-v', default=True, help='Print at what epoch and to what value the learning rate changes')
parser.add_argument('--res_cutoff',  '-r', default=1, help='Maximum residual value allowed after training')
parser.add_argument('--max_iter',    '-a', default=1, help='Number of times to train the network after pruning')
parser.add_argument('--log',         '-l', default=False, help='Set to true to take the log of all outputs')
parser.add_argument('--dot_product', '-d', default=True, help='Set to True to include dot products of 4-vectors as inputs')
parser.add_argument('--normalize', default=True, help='Set to True to normalize inputs')
parser.add_argument('--sm_norm', default=False, help='Set to True to element-wise divide by the standard model')

args = parser.parse_args()

datatype = args.datatype
coef = args.coef
coord = args.coord
jets = args.jets
redundant = args.redundant
int_part = args.int_part
log = args.log
dot = args.dot_product
normalize = args.normalize
sm_norm = args.sm_norm

n_samples = args.n_samples
n_epochs = args.n_epochs
batch_size = args.batch_size
t_size = args.t_size
n_nodes = args.n_nodes

lr = args.lr_i

factor = args.factor
patience=args.patience
threshold=args.threshold
cooldown=args.cooldown
min_lr = args.min_lr
verbose = args.verbose

res_cutoff = args.res_cutoff
current = 0
max_iter = args.max_iter
   
print('loading dataframe...')
df = pd.read_feather('/scratch365/cmcgrad2/data/' + datatype + '/' +  coef + '.feather')
print('dataframe loaded!')
df.insert(1, 'SM*SM', pd.read_feather('/scratch365/cmcgrad2/data/' + datatype + '/SM*SM.feather')['SM*SM'])

if not jets:
    df = df.loc[df['No. of Jets'] == 2]

df = df.drop(['No. of Jets'], axis=1)

if not redundant:
    df = df.drop(['T-H d eta', 'A-H d eta', 'T-A d eta', 'T-H d phi', 'A-H d phi', 'T-A d phi'], axis=1)

if int_part == 'gg':
    df = df.loc[(df['Particle 1'] == 21) & (df['Particle 2'] == 21)]
    
elif int_part == 'qqbar':
    df = df.loc[(df['Particle 1'] <= 6) & (df['Particle 2'] <= 6)]
    
elif int_part == 'qg':
    df = df.loc[((df['Particle 1'] <= 6) & (df['Particle 2'] == 21) | 
                 (df['Particle 1'] == 21) & (df['Particle 2'] <= 6))]
    
elif int_part == 'uubar':
    df = df.loc[(np.abs(df['Particle 1']) == 1) & (np.abs(df['Particle 2']) == 1)]
    
df = df.drop(['Particle 1', 'Particle 2'], axis=1)

if n_samples == 'full':
    n_samples = len(df[coef])

df = df.iloc[:n_samples,:]

if dot:
    df = mom_dot(df)
    
if normalize:
    end = len(df.columns) - 1
    df.iloc[:,2:end], std, mean = norm(df.iloc[:,2:end])
    
if sm_norm:
    df[coef] = df[coef].divide(df['SM*SM'])
    
end = len(df.columns) - 1

f = open('plots/' + int_part + '/' + coef  + '/network_performance/parameters.txt','w+')

f.write(    
    'Inputs: ' + str(list(df.iloc[:,2:end].columns)) + '\n' + 
    'Datatype: ' + datatype + '\n' +
    'Initial States: ' + int_part + '\n' + 
    'Dataset: ' + coef + '\n' +
    'Coordinates: ' + coord + '\n' +
    'Number of Epochs: ' + str(n_epochs) + '\n' +
    'Initial LR: ' + str(lr) + '\n' +
    'No. of Samples: ' + str(n_samples) + '\n' +
    'Percent Trained: ' + str(t_size) + '\n' +
    'Batch Size: ' + str(batch_size) + '\n' + 
    'LR Factor: ' + str(factor) + '\n' +
    'Min LR: ' + str(min_lr) + '\n' + 
    'Threshold: ' + str(threshold) + '\n' +
    'Patience: ' + str(patience) + '\n' +
    'Cooldown: ' + str(cooldown) + '\n' +
    'Residual Cutoff: ' + str(res_cutoff) + '\n' +
    'Max Iteration: ' + str(max_iter) + '\n' +
    'Number of nodes: ' + str(n_nodes) + '\n' +
    'Normalization: ' + str(normalize) + '\n' +
    'Divide by SM*SM: ' + str(sm_norm) + '\n'+
    'Extra Jets: ' + str(jets) + '\n'
)

#while current < max_iter:
inputs, outputs, test, corr, df_test = sliced(df, t_size, batch_size, log)

model = torch.nn.Sequential(
    torch.nn.Linear(inputs.shape[1],n_nodes),
    torch.nn.ReLU(),
    torch.nn.Linear(n_nodes,n_nodes),
    torch.nn.ReLU(),
    torch.nn.Linear(n_nodes,1)
).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=factor, patience=patience,
                                                       threshold=threshold, cooldown=cooldown, min_lr=min_lr, 
                                                       verbose=verbose)
print('starting training...')
lossIn, lossOut, lossTest = train(test, corr, inputs, outputs, batch_size, model, n_epochs, optimizer, scheduler)

model = model.cpu()
corr = corr.cpu().detach()
test = test.cpu().detach()
pred = model(test).detach()

torch.save(model, 'plots/' + int_part + '/' + coef  + '/network_performance/model.pt')

if log == True:
    corr = np.exp(corr)
    pred = np.exp(pred)

residuals  = torch.abs(pred.squeeze() - corr)
outputs = outputs.cpu()
inputs = inputs.cpu()

df['Residuals'] = np.append(residuals.detach().numpy(), torch.abs(model(inputs).squeeze() - outputs).detach().numpy())

dnnPlots(lossIn, lossOut, lossTest, corr, pred, residuals, coef, coord, int_part, current, log, '') 

test, corr = shat_filter(df_test, 1e1, 3e0, True)
pred = model(test).detach()

dnnPlots(lossIn, lossOut, lossTest, corr, pred, residuals, coef, coord, int_part, current, log, 'filtered_')
    
#    df = df.loc[df['Residuals'] < res_cutoff].reset_index(drop=True)
#    current = current + 1
#    f.write('Iter ' + str(current) + ' No. of Samples: ' + str(len(df)) + '\n')

f.close()
