import pandas as pd
import torch
import argparse

from functions import dnnPlots, train, sliced

parser = argparse.ArgumentParser(description='A script to train a neural network to train over the likelihood ratios of different EFT coefficients')

#Arguemnts that require command line input
parser.add_argument('--coef', default='SM*SM', help='The coefficient being used to train over')
parser.add_argument('--coord', default='px_py_pz', help='Coordinate system used for event data. Two options: px_py_pz pt_eta_phi')

#Arguments that can be adapted
parser.add_argument('--n_epochs', default=1000, help='Number of epochs to train the network over')
parser.add_argument('--lr_i', default=2e-1, help='The initial learning rate for the nework')
parser.add_argument('--n_samples', default=1000000, help='Number of events to use from the dataset')
parser.add_argument('--t_size', default=0.9, help='Portion of events used to train the network')
parser.add_argument('--batch_size', default=500, help='Number of samples trained over at a time')
parser.add_argument('--factor', default=0.1, help='What to multiply the learning rate by upon adaptation')
parser.add_argument('--min_lr', default=1e-4, help='The lowest value the learning rate can reduce to')
parser.add_argument('--verbose', default=True, help='Print at what epoch and to what value the learning rate changes')
parser.add_argument('--res_cutoff', default=100, help='Maximum residual value allowed after training')
parser.add_argument('--max_iter', default=4, help='Number of times to train the network after pruning')

args = parser.parse_args()

coef = args.coef
coord = args.coord

n_samples = args.n_samples
n_epochs = args.n_epochs
batch_size = args.batch_size
t_size = args.t_size

lr = args.lr_i

factor = args.factor
patience=200
threshold=1e-4
threshold_mode='rel'
cooldown=25
min_lr = args.min_lr
verbose = args.verbose

res_cutoff = args.res_cutoff
current = 0
max_iter = args.max_iter

f = open('plots/' + coef + '/' + coord + '/network_performance/parameters.txt','w+')
f.write('Dataset: ' + coef + '\n' +
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
        'Max Iteration: ' + str(max_iter) + '\n')

if coord == 'px_py_pz':
    labels = ["Higgs $p_x$", "Higgs $p_y$", "Higgs $p_z$",
              "Top $p_x$", "Top $p_y$", "Top $p_z$",
              "Anti-Top $p_x$", "Anti-Top $p_y$", "Anti-Top $p_z$"]
    
if coord == 'pt_eta_phi':
    labels = ["Higgs $p_T$", "Higgs $\eta$", "Higgs $\phi$",
              "Top $p_T$", "Top $\eta$", "Top $phi$",
              "Anti-Top $p_T$", "Anti-Top $\eta$", "Anti-Top $\phi$"]
   

df = pd.read_feather('/scratch365/cmcgrad2/data/' + coord + '/dataframes/' + coef + '.feather')
df.columns = labels + ['$r_c$']
df = df.iloc[:n_samples,:]

while current < max_iter:
    inputs, outputs, test, corr = sliced(df, t_size, batch_size)

    model = torch.nn.Sequential(
        torch.nn.Linear(inputs.shape[1],500),
        torch.nn.ReLU(),
        torch.nn.Linear(500,500),
        torch.nn.ReLU(),
        torch.nn.Linear(500,1)
    ).cuda()

    optimizer = torch.optim.Adam(model.parameters())

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=factor, patience=patience,
                                                           threshold=threshold, threshold_mode=threshold_mode,
                                                           cooldown=cooldown, min_lr=min_lr, verbose=verbose)
    
    lossIn, lossOut, lossTest = train(test, corr, inputs, outputs, batch_size, model, n_epochs, optimizer, scheduler)
    df['Residuals'] = dnnPlots(lossIn, lossOut, lossTest, test, corr, inputs, outputs, model, coef, coord, current)
    df = df.loc[df['Residuals'] < res_cutoff].reset_index(drop=True)
    current = current + 1
    f.write('Iter ' + str(current) + 'No. of Samples: ' + str(len(df)) + '\n')

f.close()