import pandas as pd
import numpy as np
import torch
import argparse

from functions import dnnPlots, train, sliced, mom_dot

parser = argparse.ArgumentParser(description='A script to train a neural network to train over the likelihood ratios of different EFT coefficients')

#Arguemnts that require command line input
parser.add_argument('--coef', default='SM*SM', help='The coefficient being used to train over')
parser.add_argument('--coord', default='px_py_pz', help='Coordinate system used for event data. Two options: px_py_pz pt_eta_phi')
parser.add_argument('--int_part', default='uubar', help='Initail state particles to include')

#Arguments that can be adapted
parser.add_argument('--n_epochs', default=1000, help='Number of epochs to train the network over')
parser.add_argument('--lr_i', default=1e-4, help='The initial learning rate for the nework')
parser.add_argument('--n_nodes', default=1200, help='number of nodes per layer in the network')
parser.add_argument('--n_samples', default=1657028, help='Number of events to use from the dataset')
parser.add_argument('--t_size', default=0.9, help='Portion of events used to train the network')
parser.add_argument('--batch_size', default=500, help='Number of samples trained over at a time')
parser.add_argument('--factor', default=0.1, help='What to multiply the learning rate by upon adaptation')
parser.add_argument('--min_lr', default=1e-8, help='The lowest value the learning rate can reduce to')
parser.add_argument('--patience', default=100, help='Number of epochs before changing the learning rate')
parser.add_argument('--threshold', default=1e-4, help='Threshold for learning rate adaptation')
parser.add_argument('--cooldown', default=25, help='Number of epochs before starting the patience count again')
parser.add_argument('--verbose', default=True, help='Print at what epoch and to what value the learning rate changes')
parser.add_argument('--res_cutoff', default=1, help='Maximum residual value allowed after training')
parser.add_argument('--max_iter', default=1, help='Number of times to train the network after pruning')
parser.add_argument('--log', default=False, help='Set to true to take the log of all outputs')

args = parser.parse_args()

coef = args.coef
coord = args.coord
int_part = args.int_part
log = args.log

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

print('Using ' + coef + '!')

if coord == 'px_py_pz':
    labels = ["Higgs $p_x$", "Higgs $p_y$", "Higgs $p_z$",
              "Top $p_x$", "Top $p_y$", "Top $p_z$",
              "Anti-Top $p_x$", "Anti-Top $p_y$", "Anti-Top $p_z$"]
    
if coord == 'pt_eta_phi':
    labels = ["Higgs $p_T$", "Higgs $\eta$", "Higgs $\phi$",
              "Top $p_T$", "Top $\eta$", "Top $phi$",
              "Anti-Top $p_T$", "Anti-Top $\eta$", "Anti-Top $\phi$"]
   

df = pd.read_feather('/scratch365/cmcgrad2/data/' + int_part + '/' + coord + '/dataframes/' + coef + '.feather')
df.columns = labels + ['$r_c$']
df = df.iloc[:n_samples,:]
df = mom_dot(df)
end = len(df.columns) - 1

f = open('plots/' + int_part + '/' + coef + '/' + coord + '/network_performance/parameters.txt','w+')

f.write(
    'Inputs: ' + str(list(df.iloc[:,:end].columns)) + '\n'
    'Initial States: ' + int_part + '\n' +
    'Dataset: ' + coef + '\n' +
    'Coordinates: ' + coord + '\n' +
    'Number of Epochs: ' + str(n_epochs) + '\n' +
    'Initial LR: ' + str(lr) + '\n' +
    'No. of Samples: ' + str(n_samples) + '\n' +
    'Percent Trained: ' + str(t_size) + '\n'
    'Batch Size: ' + str(batch_size) + '\n' + 
    'LR Factor: ' + str(factor) + '\n' +
    'Min LR: ' + str(min_lr) + '\n' + 
    'Threshold: ' + str(threshold) + '\n' +
    'Patience: ' + str(patience) + '\n' +
    'Cooldown: ' + str(cooldown) + '\n' +
    'Residual Cutoff: ' + str(res_cutoff) + '\n' +
    'Max Iteration: ' + str(max_iter) + '\n' +
    'Number of nodes: ' + str(n_nodes) + '\n'
)

while current < max_iter:
    inputs, outputs, test, corr = sliced(df, t_size, batch_size, log)

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
    
    lossIn, lossOut, lossTest = train(test, corr, inputs, outputs, batch_size, model, n_epochs, optimizer, scheduler)
    
    model = model.cpu()
    corr = corr.cpu().detach()
    test = test.cpu().detach()
    pred = model(test).detach()
    
    if log == True:
        corr = np.exp(corr)
        pred = np.exp(pred)
        
    residuals  = torch.abs(pred.squeeze() - corr)
    outputs = outputs.cpu()
    inputs = inputs.cpu()

    residuals = np.append(residuals.detach().numpy(), torch.abs(model(inputs).squeeze() - outputs).detach().numpy())
    
    
    df['Residuals'] = residuals
    
    dnnPlots(lossIn, lossOut, lossTest, corr, pred, residuals, coef, coord, int_part, current, log)
    df = df.loc[df['Residuals'] < res_cutoff].reset_index(drop=True)
    current = current + 1
    f.write('Iter ' + str(current) + ' No. of Samples: ' + str(len(df)) + '\n')

f.close()