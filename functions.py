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

    wc_names_lst = ['SM'] + wc_names_lst

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
    i = len(df.columns)
    df.iloc[:,i] = np.log((df.iloc[:,i] - df.iloc[:,i].min()) + (df.iloc[:,i] - ddf.iloc[:,i].min()).nsmallest(2).iloc[-1])
    return df

def hist(df, title, filename, datatype, coef, log=True):
    i = len(df.columns)
    stat = df.agg(['skew', 'kurtosis']).transpose()
    k2 = stats.normaltest(df.iloc[:,i].to_numpy())[0]

    plt.rcParams['figure.figsize'] = [15, 8]

    n, bins, patches = plt.hist(df.iloc[:,i], 100, density=False, alpha=0.75)

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
    i = len(df.columns)
    plt.plot(df[name].to_numpy(), df.iloc[:,i].to_numpy(), color)
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
    
def dnnPlots(lossIn, lossOut, lossTest, corr, pred, res, coef, coord, int_part, num, log, prefix):

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
    plt.savefig(str('plots/' + int_part + '/' + coef + '/network_performance/loss_run_' + str(num) + '.png'))
    plt.clf()

    #Expected vs results
    _, x_bin, _, _ = plt.hist2d(corr.numpy(),pred.numpy().squeeze(), bins = [250,250], norm = colors.LogNorm())
    plt.plot([corr.numpy().min(), corr.numpy().max()], [corr.numpy().min(), corr.numpy().max()],
             '--', color='#647C90', label='Target', lw=3)    
    plt.colorbar()
    plt.xlabel("Correct", fontsize=12)
    plt.ylabel("Predicted", fontsize=12)
    plt.title("Training Distribution Against Expected Results", fontsize=16)
    plt.savefig(str('plots/' + int_part + '/' + coef + '/network_performance/' + prefix + 'expected_run_' + str(num) + '.png'))
    plt.clf()
    
    #Profile Plot
    ax = plt.axes()
    ax.set_facecolor('#00195E')
    
    cmap1 = colors.LinearSegmentedColormap.from_list("mycmap", list(zip([0.0, 0.3, 0.6, 1],
                                                                        ['#070782', '#08809E', '#0DCFFF', '#FFFFFF'])))
    
    p_x, p_mean, p_rms, p_median, one_sig, two_sig = compute_profile(corr.numpy(), pred.numpy().squeeze(), x_bin)
           
    plt.plot([corr.numpy().min(), corr.numpy().max()], [corr.numpy().min(), corr.numpy().max()],
             '-', color='#FF8000', label='Target', lw=2, zorder=10)
    plt.plot([corr.numpy().min(), corr.numpy().max()], [corr.numpy().min(), corr.numpy().max()],
             '-', color='k', lw=3, zorder=5)
    #two sigma
    plt.errorbar(p_x, p_mean, 2*p_rms,fmt=' ', ecolor='#FF009F',
                 capthick=2, elinewidth=2, capsize=5, zorder=10,  errorevery=8,
                 label=r'$2\sigma$ (%0.2f%% of prediction hits target bin)' % two_sig)
    plt.errorbar(p_x, p_mean, 2*p_rms,fmt=' ', ecolor='k',
                 capthick=3, elinewidth=3, capsize=5, zorder=5,  errorevery=8)
    #one sigma
    plt.errorbar(p_x, p_mean, p_rms,fmt=' ', ecolor='#24FF0F',
                 capthick=2,elinewidth=2, capsize=5, zorder=10,  errorevery=8,
                 label=r'$\sigma$ (%0.2f%% of prediction hits target bin)' % one_sig)
    plt.errorbar(p_x, p_mean, p_rms,fmt=' ', ecolor='k', 
                 capthick=3, elinewidth=3, capsize=5, zorder=5,  errorevery=8)
    #mean and median
    plt.plot(p_x, p_mean, 's', color='#FF0A01', label='mean', 
             markersize=10, markeredgecolor='k', markevery=8, zorder=10)
    plt.plot(p_x, p_median, '*', color='#FFFE03', label='median',
             markersize=20, markeredgecolor='k', markevery=8, zorder=10)

    plt.hist2d(corr.numpy(), pred.numpy().squeeze(), [250,250], norm = colors.LogNorm(), cmap=cmap1)
    
    
    plt.xlabel("Correct", fontsize=12) 
    plt.ylabel("Predicted", fontsize=12)
    plt.legend(fontsize=12)
    plt.colorbar()
    plt.title("Profiled Training Distribution Against Expected Results", fontsize=16)
    plt.savefig(str('plots/' + int_part + '/' + coef + '/network_performance/' + prefix + 'profile_plot_' + str(num) + '.png'))
    plt.clf() 
    
    #Profile Plot with standard devation divided by sqrt(n)
    ax = plt.axes()
    ax.set_facecolor('#00195E')
    
    p_x, p_mean, p_rms_n, p_median, one_sig, two_sig = compute_profile_normalized(corr.numpy(), pred.numpy().squeeze(), x_bin)
           
    plt.plot([corr.numpy().min(), corr.numpy().max()], [corr.numpy().min(), corr.numpy().max()],
             '-', color='#FF8000', label='Target', lw=2, zorder=10)
    plt.plot([corr.numpy().min(), corr.numpy().max()], [corr.numpy().min(), corr.numpy().max()],
             '-', color='k', lw=3, zorder=5)
    #two sigma
    plt.errorbar(p_x, p_mean, 2*p_rms_n,fmt=' ', ecolor='#FF009F',
                 capthick=2, elinewidth=2, capsize=5, zorder=10,  errorevery=8,
                 label=r'$\frac{2\sigma}{\sqrt{n}}$ (%0.2f%% of prediction hits target bin)' % two_sig)
    plt.errorbar(p_x, p_mean, 2*p_rms_n,fmt=' ', ecolor='k',
                 capthick=3, elinewidth=3, capsize=5, zorder=5,  errorevery=8)
    #one sigma
    plt.errorbar(p_x, p_mean, p_rms_n,fmt=' ', ecolor='#24FF0F',
                 capthick=2,elinewidth=2, capsize=5, zorder=10,  errorevery=8,
                 label=r'$\frac{\sigma}{\sqrt{n}}$ (%0.2f%% of prediction hits target bin)' % one_sig)
    plt.errorbar(p_x, p_mean, p_rms_n,fmt=' ', ecolor='k', 
                 capthick=3, elinewidth=3, capsize=5, zorder=5,  errorevery=8)
    #mean and median
    plt.plot(p_x, p_mean, 's', color='#FF0A01', label='mean', 
             markersize=10, markeredgecolor='k', markevery=8, zorder=10)
    plt.plot(p_x, p_median, '*', color='#FFFE03', label='median',
             markersize=20, markeredgecolor='k', markevery=8, zorder=10)

    plt.hist2d(corr.numpy(), pred.numpy().squeeze(), [250,250], norm = colors.LogNorm(), cmap=cmap1)
    
    
    plt.xlabel("Correct", fontsize=12) 
    plt.ylabel("Predicted", fontsize=12)
    plt.legend(fontsize=12)
    plt.colorbar()
    plt.title("Profiled Training Distribution Against Expected Results - Sample Standard Deviation", fontsize=16)
    plt.savefig(str('plots/' + int_part + '/' + coef + '/network_performance/' + prefix + 'profile_plot_' + str(num) + '_normalized.png'))
    plt.clf()
    
    #Profile Plot with standard devation divided by sqrt(n)
    ax = plt.axes()
    ax.set_facecolor('#00195E')
    
    p_x, p_mean, p_first, p_median, one_sig, two_sig = compute_profile_quartile(corr.numpy(), pred.numpy().squeeze(), x_bin)
           
    plt.plot([corr.numpy().min(), corr.numpy().max()], [corr.numpy().min(), corr.numpy().max()],
             '-', color='#FF8000', label='Target', lw=2, zorder=10)
    plt.plot([corr.numpy().min(), corr.numpy().max()], [corr.numpy().min(), corr.numpy().max()],
             '-', color='k', lw=3, zorder=5)
    
    #one sigma
    plt.errorbar(p_x, p_mean, p_first,fmt=' ', ecolor='#24FF0F',
                 capthick=2,elinewidth=2, capsize=5, zorder=10,  errorevery=8,
                 label=r'$2^{nd}$ to $3^{rd}$ quartiles (%0.2f%% of prediction hits target bin)' % one_sig)
    plt.errorbar(p_x, p_mean, p_first,fmt=' ', ecolor='k', 
                 capthick=3, elinewidth=3, capsize=5, zorder=5,  errorevery=8)
    #mean and median
    plt.plot(p_x, p_mean, 's', color='#FF0A01', label='mean', 
             markersize=10, markeredgecolor='k', markevery=8, zorder=10)
    plt.plot(p_x, p_median, '*', color='#FFFE03', label='median',
             markersize=20, markeredgecolor='k', markevery=8, zorder=10)

    plt.hist2d(corr.numpy(), pred.numpy().squeeze(), [250,250], norm = colors.LogNorm(), cmap=cmap1)
    
    
    plt.xlabel("Correct", fontsize=12) 
    plt.ylabel("Predicted", fontsize=12)
    plt.legend(fontsize=12)
    plt.colorbar()
    plt.title("Profiled Training Distribution Against Expected Results - Quartiles", fontsize=16)
    plt.savefig(str('plots/' + int_part + '/' + coef + '/network_performance/' + prefix + 'profile_plot_' + str(num) + '_quartile.png'))
    plt.clf()
    
    #Profile Plot with counted standard devation
    ax = plt.axes()
    ax.set_facecolor('#00195E')
    
    p_x, p_mean, p_first, p_second, p_median, one_sig, two_sig = compute_profile_counted(corr.numpy(), pred.numpy().squeeze(), x_bin)
           
    plt.plot([corr.numpy().min(), corr.numpy().max()], [corr.numpy().min(), corr.numpy().max()],
             '-', color='#FF8000', label='Target', lw=2, zorder=10)
    plt.plot([corr.numpy().min(), corr.numpy().max()], [corr.numpy().min(), corr.numpy().max()],
             '-', color='k', lw=3, zorder=5)
    
    #two sigma
    plt.errorbar(p_x, p_mean, p_second,fmt=' ', ecolor='#FF009F',
                 capthick=2, elinewidth=2, capsize=5, zorder=10,  errorevery=8,
                 label=r'middle 95.45%% (%0.2f%% of prediction hits target bin)' % two_sig)
    plt.errorbar(p_x, p_mean, p_second,fmt=' ', ecolor='k',
                 capthick=3, elinewidth=3, capsize=5, zorder=5,  errorevery=8)
    
    #one sigma
    plt.errorbar(p_x, p_mean, p_first,fmt=' ', ecolor='#24FF0F',
                 capthick=2,elinewidth=2, capsize=5, zorder=10,  errorevery=8,
                 label=r'middle 68.27%% (%0.2f%% of prediction hits target bin)' % one_sig)
    plt.errorbar(p_x, p_mean, p_first,fmt=' ', ecolor='k', 
                 capthick=3, elinewidth=3, capsize=5, zorder=5,  errorevery=8)
    #mean and median
    plt.plot(p_x, p_mean, 's', color='#FF0A01', label='mean', 
             markersize=10, markeredgecolor='k', markevery=8, zorder=10)
    plt.plot(p_x, p_median, '*', color='#FFFE03', label='median',
             markersize=20, markeredgecolor='k', markevery=8, zorder=10)

    plt.hist2d(corr.numpy(), pred.numpy().squeeze(), [250,250], norm = colors.LogNorm(), cmap=cmap1)
    
    
    plt.xlabel("Correct", fontsize=12) 
    plt.ylabel("Predicted", fontsize=12)
    plt.legend(fontsize=12)
    plt.colorbar()
    plt.title("Profiled Training Distribution Against Expected Results - Counted Standard Deviation", fontsize=16)
    plt.savefig(str('plots/' + int_part + '/' + coef + '/network_performance/' + prefix + 'profile_plot_' + str(num) + '_counted.png'))
    plt.clf()
    
    #Scatter plot of normalized standard deviation
    plt.scatter(p_x, p_rms/p_mean, label=r'$\frac{\sigma}{\mu}$')
    plt.scatter(p_x, p_rms, label=r'$\sigma$')
    plt.scatter(p_x, p_rms_n, label=r'$\frac{\sigma}{\sqrt{n}}$')
    
    plt.xlabel('Correct', fontsize=12)
    plt.yscale('log')
    plt.legend(fontsize=12)
    plt.title('Standard Deviation for Training', fontsize=16)
    plt.savefig(str('plots/' + int_part + '/' + coef + '/network_performance/' + prefix + 'standard_deviation_' + str(num) + '.png'))
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
    plt.savefig(str('plots/' + int_part + '/' + coef + '/network_performance/' + prefix + 'distributions_run_' + str(num) + '.png'))
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
    plt.savefig(str('plots/' + int_part + '/' + coef + '/network_performance/' + prefix + 'distributions_log_pos_run_' + str(num) + '.png'))
    plt.clf()

    #Distribution of residuals
    plt.hist(res, bins=100)
    plt.grid(b=True, color='grey', alpha=0.2, linestyle=':', linewidth=2)
    plt.yscale('log')
    plt.xlabel("Residual", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Residual Distribution (Linear)", fontsize=16)
    plt.savefig(str('plots/' + int_part + '/' + coef + '/network_performance/' + prefix + 'residuals_run_' + str(num) + '.png'))
    plt.clf()

    return 

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
    lossIn = np.arange(0, n_epochs + 1, 1)
    
    prediction = torch.squeeze(model(inputs))
    testLoss = lossFunc(torch.squeeze(model(test)),corr)
    loss = lossFunc(prediction,outputs)
    lossOut.append(loss.tolist())
    lossTest.append(testLoss.tolist())

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

def sliced(df, t_size, batch_size, log):
    end = len(df.columns) - 1
    t_size = int(math.floor(len(df)*t_size / batch_size) * batch_size)
    inputs  = torch.tensor(df.iloc[:t_size,2:end].values, dtype = torch.float32).cuda()
    df_test = df.iloc[t_size:,:]
    test    = torch.tensor(df_test.iloc[:, 2:end].values, dtype = torch.float32).cuda()
    
    if log == True:
        df.iloc[:,end] = np.log(df.iloc[:,end])
                
    outputs = torch.tensor(df.iloc[:t_size, end].values, dtype = torch.float32).cuda()
    corr    = torch.tensor(df.iloc[t_size:, end].values, dtype = torch.float32).cuda()
    return(inputs, outputs, test, corr, df_test)

def mom_dot(df):
    higgs_mass = 125
    top_mass = 176
    i = len(df.columns) - 1
    
    df.insert(i,  'Higgs E', np.sqrt(higgs_mass**2 +
                                     df['Higgs px']**2 +
                                     df['Higgs py']**2 +
                                     df['Higgs pz']**2))
    df.insert(i, 'Top E', np.sqrt(top_mass**2 +
                                  df['Top px']**2 +
                                  df['Top py']**2 +
                                  df['Higgs pz']**2))
    df.insert(i, 'Anti-Top E', np.sqrt(top_mass**2 + 
                                       df['Anti-Top px']**2 +
                                       df['Anti-Top py']**2 +
                                       df['Anti-Top pz']**2))
    
    df.insert(i, 'PH dot PT', (df['Higgs E']*df['Top E'] - 
                               df['Higgs px']*df['Top px'] - 
                               df['Higgs py']*df['Top py'] -
                               df['Higgs pz']*df['Top pz']))
    df.insert(i, 'PH dot PA', (df['Higgs E']*df['Anti-Top E'] - 
                               df['Higgs px']*df['Anti-Top px'] - 
                               df['Higgs py']*df['Anti-Top py'] -
                               df['Higgs pz']*df['Anti-Top pz']))
    df.insert(i, 'PT dot PA', (df['Higgs E']*df['Anti-Top E'] - 
                               df['Top px']*df['Anti-Top px'] - 
                               df['Top py']*df['Anti-Top py'] -
                               df['Top pz']*df['Anti-Top pz']))
    
    df = df.drop(['Higgs E', 'Top E', 'Anti-Top E'], axis=1)
    
    return(df)

def delta_r(df):
    i = len(df.columns) - 1
    
    df.insert(i, 'T-H d r', np.sqrt(df['T-H d eta']**2  + 
                                    df['T-H d phi']**2))
    df.insert(i, 'A-H d r', np.sqrt(df['A-H d eta']**2  + 
                                    df['A-H d phi']**2))
    df.insert(i, 'T-A d r', np.sqrt(df['T-A d eta']**2  + 
                                    df['T-A d phi']**2))
    return(df)

def compute_profile(x, y, xbins):
    one_sig = 0
    two_sig = 0
    xbinw = xbins[1]-xbins[0]
    x_array      = []
    x_slice_mean = []
    x_slice_rms  = []
    x_slice_median = []
    for i in range(xbins.size-1):
        yvals = y[ (x>xbins[i]) & (x<=xbins[i+1]) ]
        if yvals.size>0:
            xval = x[(x>xbins[i]) & (x<=xbins[i+1])]
            stdv = yvals.std()
            mn = yvals.mean()
            x_slice_median.append(np.median(yvals))
            x_array.append(xbins[i]+ xbinw/2)
            x_slice_mean.append(mn)
            x_slice_rms.append(stdv)
            if (mn + stdv > xval.min()) & (mn - stdv < xval.max()):
                one_sig = one_sig + len(xval)
            if (mn + 2*stdv > xval.min()) & (mn - 2*stdv < xval.max()):
                two_sig = two_sig + len(xval)
            
    x_array = np.array(x_array)
    x_slice_mean = np.array(x_slice_mean)
    x_slice_rms = np.array(x_slice_rms)
    one_sig = one_sig/len(x)*100
    two_sig = two_sig/len(x)*100
    return x_array, x_slice_mean, x_slice_rms, x_slice_median, one_sig, two_sig

def compute_profile_normalized(x, y, xbins):
    one_sig = 0
    two_sig = 0
    xbinw = xbins[1]-xbins[0]
    x_array      = []
    x_slice_mean = []
    x_slice_rms  = []
    x_slice_median = []
    for i in range(xbins.size-1):
        yvals = y[ (x>xbins[i]) & (x<=xbins[i+1]) ]
        if yvals.size>0:
            xval = x[(x>xbins[i]) & (x<=xbins[i+1])]
            stdv = yvals.std()/np.sqrt(len(yvals))
            mn = yvals.mean()
            x_slice_median.append(np.median(yvals))
            x_array.append(xbins[i]+ xbinw/2)
            x_slice_mean.append(mn)
            x_slice_rms.append(stdv)
            if (mn + stdv > xval.min()) & (mn - stdv < xval.max()):
                one_sig = one_sig + len(xval)
            if (mn + 2*stdv > xval.min()) & (mn - 2*stdv < xval.max()):
                two_sig = two_sig + len(xval)
            
    x_array = np.array(x_array)
    x_slice_mean = np.array(x_slice_mean)
    x_slice_rms = np.array(x_slice_rms)
    one_sig = one_sig/len(x)*100
    two_sig = two_sig/len(x)*100
    return x_array, x_slice_mean, x_slice_rms, x_slice_median, one_sig, two_sig

def compute_profile_quartile(x, y, xbins):
    one_sig = 0
    two_sig = 0
    xbinw = xbins[1]-xbins[0]
    first_quar = np.array([[], []])
    x_array      = []
    x_slice_mean = []
    x_slice_median = []
    for i in range(xbins.size-1):
        yvals = y[ (x>xbins[i]) & (x<=xbins[i+1]) ]
        if yvals.size>0:
            quart = np.quantile(yvals, [0, 0.25, 0.5, 0.75, 1])
            xval = x[(x>xbins[i]) & (x<=xbins[i+1])]
            first_quar = np.append(first_quar, [[quart[2] - quart[1]], [quart[3] - quart[2]]], axis=1)
            mn = yvals.mean()
            x_slice_median.append(quart[2])
            x_array.append(xbins[i]+ xbinw/2)
            x_slice_mean.append(mn)
            if (quart[3] > xval.min()) & (quart[1] < xval.max()):
                one_sig = one_sig + len(xval)
            if (quart[4] > xval.min()) & (quart[0] < xval.max()):
                two_sig = two_sig + len(xval)
            
    x_array = np.array(x_array)
    x_slice_mean = np.array(x_slice_mean)
    one_sig = one_sig/len(x)*100
    two_sig = two_sig/len(x)*100
    return x_array, x_slice_mean, first_quar, x_slice_median, one_sig, two_sig

def compute_profile_counted(x, y, xbins):
    one_sig = 0
    two_sig = 0
    xbinw = xbins[1]-xbins[0]
    first_quar = np.array([[], []])
    secnd_quar = np.array([[], []])
    x_array      = []
    x_slice_mean = []
    x_slice_median = []
    for i in range(xbins.size-1):
        yvals = y[ (x>xbins[i]) & (x<=xbins[i+1]) ]
        if yvals.size>0:
            quart = np.quantile(yvals, [0, 0.0275, 0.34135, 0.50, 0.84135, 0.97725, 1])
            xval = x[(x>xbins[i]) & (x<=xbins[i+1])]
            first_quar = np.append(first_quar, [[quart[3] - quart[2]], [quart[4] - quart[3]]], axis=1)
            secnd_quar = np.append(secnd_quar, [[quart[3] - quart[1]], [quart[5] - quart[3]]], axis=1)
            mn = yvals.mean()
            x_slice_median.append(quart[2])
            x_array.append(xbins[i]+ xbinw/2)
            x_slice_mean.append(mn)
            if (quart[3] > xval.min()) & (quart[1] < xval.max()):
                one_sig = one_sig + len(xval)
            if (quart[4] > xval.min()) & (quart[0] < xval.max()):
                two_sig = two_sig + len(xval)
            
    x_array = np.array(x_array)
    x_slice_mean = np.array(x_slice_mean)
    one_sig = one_sig/len(x)*100
    two_sig = two_sig/len(x)*100
    return x_array, x_slice_mean, first_quar, secnd_quar, x_slice_median, one_sig, two_sig

def shat_filter(df, filt1, filt2, log):
    
    end = len(df.columns) - 1
    
    df['coef_norm'] = df.iloc[:, end].divide(df['SM*SM'])
    df['shat'] = df['shat']/1000
    bins = get_bins(df, log)
    h, xedges, yedges, _ = plt.hist2d(df['shat'], df['coef_norm'], bins=bins, norm=colors.LogNorm())
    
    xtop = xedges[1:]
    ytop = yedges[1:]
    xbot = xedges[np.where(h >= filt1)[0]]
    xtop = xtop[np.where(h >= filt1)[0]]
    ybot = yedges[np.where(h >= filt1)[1]]
    ytop = ytop[np.where(h >= filt1)[1]]
    

    df2 = pd.DataFrame()
    for i in range(len(xtop)):
        df2 = pd.concat([df2, df.loc[((xbot[i] < df['shat']) & (xtop[i] > df['shat'])) &
                                    ((ybot[i] < df['coef_norm']) & (ytop[i] > df['coef_norm']))]], ignore_index=True)
    
    
    bins = get_bins(df2, log)
    print(df2)
    h, xedges, yedges, _ = plt.hist2d(df2['shat'], df2['coef_norm'], bins=bins, norm=colors.LogNorm())
    
    xtop = xedges[1:]
    ytop = yedges[1:]
    xbot = xedges[np.where(h >= filt2)[0]]
    xtop = xtop[np.where(h >= filt2)[0]]
    ybot = yedges[np.where(h >= filt2)[1]]
    ytop = ytop[np.where(h >= filt2)[1]]
    
    df3 = pd.DataFrame()
    for i in range(len(xtop)):
        df3 = pd.concat([df3, df2.loc[((xbot[i] < df2['shat']) & (xtop[i] > df2['shat'])) &
                                      ((ybot[i] < df2['coef_norm']) & (ytop[i] > df2['coef_norm']))]], ignore_index=True)
        
    df3 = df3.drop(['SM*SM', 'coef_norm', 'shat'], axis=1)
    
    end = len(df3.columns) - 1
        
    test = torch.tensor(df3.iloc[:,:end].values, dtype = torch.float32)
    corr = torch.tensor(df3.iloc[:, end].values, dtype = torch.float32)
    
    corr = corr.cpu().detach()
    test = test.cpu().detach()
    
    return test, corr
    
def get_bins(df, log):
    
    if log:
        df = df.loc[df['shat'] > 0]
        df = df.loc[df['coef_norm'] > 0]
        bins = [ np.linspace(df['shat'].min(), df['shat'].max(), num=250),
                np.logspace(np.log10(df['coef_norm'].min()), np.log10(df['coef_norm'].max()), num=250)]
    else:
        bins = [np.linspace(df['shat'].min(), df['shat'].max(), num=250), 
                np.linspace(df['coef_norm'].min(), df['coef_norm'].max(), num=250)]
    
    return bins
    