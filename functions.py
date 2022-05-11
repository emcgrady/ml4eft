import torch
import numpy as np
import pandas as pd
import math
from scipy import stats
from scipy.special import inv_boxcox
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def get_wc_ref_cross(wc_ref):
    '''returns the list of 276 WCs and cross terms evalulated
    at the initial WC values provided from wc_ref'''

    wc_ref = [1] + wc_ref

    wc_ref_cross=[]
    for i in range(len(wc_ref)):
        for j in range(i+1):
            term = wc_ref[i]*wc_ref[j]
            wc_ref_cross.append(term)

    return wc_ref_cross

def get_wc_names_cross(wc_names_lst):
    '''returns the list of names for 276 WCs and cross terms
    with the same ordering as get_wc_ref_cross'''

    wc_names_lst = ["SM"] + wc_names_lst

    wc_names_cross=[]
    for i in range(len(wc_names_lst)):
        for j in range(i+1):
            term = wc_names_lst[i]+"*"+wc_names_lst[j]
            wc_names_cross.append(term)

    return wc_names_cross

def box_cox(array):
    if np.min(array) < 0:
        array = array - np.min(array)*2
    lmbda = stats.boxcox_normmax(array, method='pearsonr')
    out = stats.boxcox(array, lmbda=lmbda, alpha=None)
    return (out, lmbda)
   
def norm(df):
    mean = df.mean()
    std  = df.std()
    out  = (df-mean)/std
    return (out, std, mean)

def norm_inv(df, std, mean):
    out = df*std+mean
    return out

def box_cox_inv(array, lmbda):
    out = inv_boxcox(array, lmbda)
    return out

def log_tf(df):
    df['r_c'] = np.log((df['r_c'] - df['r_c'].min()) + (df['r_c'] - df['r_c'].min()).nsmallest(2).iloc[-1])
    return df

def hist(df, title, filename, datatype, coef, log=True):
    stat = df.agg(['skew', 'kurtosis']).transpose()
    k2 = stats.normaltest(df['r_c'].to_numpy())[0]

    plt.rcParams['figure.figsize'] = [15, 8]

    n, bins, patches = plt.hist(df['r_c'], 100, density=False, alpha=0.75)

    xlim = np.max(bins)
    ylim = np.max(n)

    plt.xlabel('$\mathregular{r_c}$', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title(str(coef + ' Distribution of Outputs' + title), fontsize=16)
    plt.grid(True)
    if log == True:
        plt.yscale('log')
    plt.savefig(str('plots/' + coef + '/' + datatype + '/histograms/' + filename + '.png'))
    plt.show()
    return(bins, n)

def scatter(df, name, kind, axt, title, subset, datatype, coef, log=False, color='b.'):
    plt.plot(df[name].to_numpy(), df['r_c'].to_numpy(), color)
    plt.grid(b=True, color='grey', alpha=0.2, linestyle=':', linewidth=2)
    plt.xlabel(axt, fontsize=15)
    plt.xticks(fontsize=13)
    plt.ylabel('$r_c$', fontsize=15)
    plt.yticks(fontsize=13)
    plt.title(str( coef +  title + ' Dataset Output vs Input'), fontsize=16)
    if log == True:
        plt.yscale('log')
    plt.savefig(str('plots/' + coef + '/' + datatype + '/scatter_plots/' + subset + '/rc_vs_' + kind + '.png'))
    plt.show()
    
def dnnPlots(lossIn, lossOut, lossTest, test, corr, inputs, outputs, model, coef, coord, num):

    pred = model(test).cpu().detach()
    res  = torch.abs(model(test).squeeze() - corr).cpu().detach()
    corr = corr.cpu().detach()
    test = test.cpu().detach()

    pred2 = model(inputs).cpu().detach()
    res2  = torch.abs(model(inputs).squeeze() - outputs).cpu().detach()
    outputs = outputs.cpu().detach()
    inputs = inputs.cpu().detach()

    testResiduals = (torch.abs(torch.div(res,corr))).numpy()
    trainResiduals = (torch.abs(torch.div(res2,outputs))).numpy()
    residuals = np.append(trainResiduals, testResiduals)

    #Plot of the loss function
    plt.rcParams['figure.figsize'] = [15, 8]
    plt.plot(lossIn, lossOut, 'b', label='Training')
    plt.plot(lossIn, lossTest, 'tab:orange', label='Test')
    plt.grid(b=True, color='grey', alpha=0.2, linestyle=':', linewidth=2)
    plt.xlabel('Number of Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.yscale('log')
    plt.title('Loss Function by Epoch', fontsize=16)
    plt.legend(loc='best', fontsize=12)
    plt.savefig(str('plots/' + coef + '/' + coord + '/network_performance/loss_run_' + str(num) + '.png'))
    plt.clf()

    #Expected vs results
    plt.plot(corr.numpy(), pred.numpy(), 'g.', label='Training Results')
    plt.grid(b=True, color='grey', alpha=0.2, linestyle=':', linewidth=2)
    plt.xlabel("Correct", fontsize=12)
    plt.ylabel("Predicted", fontsize=12)
    plt.title("Training Distribution Against Expected Results", fontsize=16)
    plt.legend(loc="best", fontsize=12)
    plt.savefig(str('plots/' + coef + '/' + coord + '/network_performance/expected_run_' + str(num) + '.png'))
    plt.clf()

    #Distribution of results
    plt.hist(corr.numpy(), bins=250, label='Expected Results', color = 'blue', alpha=0.6)
    plt.hist(pred.numpy(), bins=250, label='Training Results', color='green', alpha=0.6)
    plt.grid(b=True, color='grey', alpha=0.2, linestyle=':', linewidth=2)
    plt.yscale('log')
    plt.xlabel("$r_c$", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Distribution of Results", fontsize=16)
    plt.legend(loc="best", fontsize=12)
    plt.savefig(str('plots/' + coef + '/' + coord + '/network_performance/distributions_run_' + str(num) + '.png'))
    plt.clf()

    #Distribution of results (Log)
    plt.hist(corr.numpy(), bins=np.logspace(np.log10(1e-10), np.log10(1e2), 250), 
             label='Expected Results', color = 'blue', alpha=0.6)
    plt.hist(pred.numpy(), bins=np.logspace(np.log10(1e-10), np.log10(1e2), 250), 
             label='Training Results', color='green', alpha=0.6)
    plt.grid(b=True, color='grey', alpha=0.2, linestyle=':', linewidth=2)
    plt.yscale('log')
    plt.xlabel("$r_c$", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Distribution of Results", fontsize=16)
    plt.xscale('log')
    plt.legend(loc="best", fontsize=12)
    plt.savefig(str('plots/' + coef + '/' + coord + '/network_performance/distributions_log_pos_run_' + str(num) + '.png'))
    plt.clf()

    #Distribution of residuals
    plt.hist(testResiduals, bins=100)
    plt.grid(b=True, color='grey', alpha=0.2, linestyle=':', linewidth=2)
    plt.yscale('log')
    plt.xlabel("Residual", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Residual Distribution (Linear)", fontsize=16)
    plt.savefig(str('plots/' + coef + '/' + coord + '/network_performance/residuals_run_' + str(num) + '.png'))
    plt.clf()

    return residuals

def train(test, corr, inputs, outputs, batch_size, model, n_epochs, optimizer, scheduler):
    
    lossFunc = torch.nn.MSELoss()
    
    model.train()

    numMiniBatch = int(math.floor(inputs.shape[0]/batch_size))
    inputMiniBatches =inputs.chunk(numMiniBatch)
    outputMiniBatches = outputs.chunk(numMiniBatch)

    corrMiniBatches = corr.chunk(numMiniBatch)
    testMiniBatches = test.chunk(numMiniBatch)


    lossOut = []
    lossTest = []
    lossIn = np.arange(1, n_epochs + 1, 1)

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


    print('Training done!') 
    
    return(lossIn, lossOut, lossTest)

def sliced(df, t_size, batch_size):
    t_size = int(math.floor(math.floor(len(df)*t_size) / batch_size) * batch_size)
    inputs  = torch.tensor(df.iloc[:t_size,:9].values, dtype = torch.float32).cuda()
    outputs = torch.tensor(df.iloc[:t_size, 9].values, dtype = torch.float32).cuda()
    test    = torch.tensor(df.iloc[t_size:,:9].values, dtype = torch.float32).cuda()
    corr    = torch.tensor(df.iloc[t_size:, 9].values, dtype = torch.float32).cuda()
    return(inputs, outputs, test, corr)