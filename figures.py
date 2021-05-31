import numpy as np
import pandas as pd
import pylab as plt
import sys

import warnings
warnings.simplefilter('ignore')

# plot settings
plt.rcParams['mathtext.default'] = 'regular'

# read in the train/validation set
train_valid = pd.read_csv(sys.argv[1])

# read in the test set
test = pd.read_csv(sys.argv[2])
k_test = np.log10(test.K)
h_test = test.H

# subdirectories
figure = 'figures'
TSV = 'TSVs'

# near-azeotropic pairs
## training set
azeo_pairs_train_valid = [
    ('propionitrile', '1,5-heptadiene'),
    ('methyl isopropyl ether', 'neopentane'),
    ('propyl alcohol', 'methyl propyl ketone'),
    ('acetonitrile', 'isopropyl alcohol'),
    ('4-methyl-1-hexene', '4,4-dimethyl-1-pentene'),
    ('acetaldehyde', 'ethylamine'),
    ('acetaldehyde', 'dimethylamine'),
    ('methyl propyl ether', '2-pentene'),
    ('propene', 'propane'),
    ('ethylamine', 'dimethylamine')
    ]

## test set
azeo_pairs_test = [
    ('1-methyl-3-buten-1-ol', '3-methyl-1-butanol'),
    ('2-hexene', '1,5-hexadiene'),
    ('propionaldehyde', 'propylamine')
    ]

# get figures
def figure3():
    fig, axes = plt.subplots(1, 4, figsize = (16, 4), dpi = 300)
    axes = axes.ravel()

    for i in range(1, 5):
        te = pd.read_csv('{}/{}/valid_K.tsv'.format(TSV, i), sep = '\t')
        GCMC = te.K
        ML = te.predict
        
        axes[i - 1].tick_params(axis = 'x', labelsize = 8)
        axes[i - 1].tick_params(axis = 'y', labelsize = 8)
        
        axes[i - 1].plot([1e-60, 1e15], [1e-60, 1e15], 'k-', lw = 0.8)
        axes[i - 1].scatter(GCMC, ML, alpha = .4, marker = '.')
        axes[i - 1].plot([1e-15, 1e-15], [1e-60, 1e15], 'k--', lw = 0.5)
        axes[i - 1].plot([1e-60, 1e15], [1e-15, 1e-15], 'k--', lw = 0.5)
        
        axes[i - 1].set_xlabel('$K_H$ from GCMC [mol/kg/Pa]', fontsize = 12)
        axes[i - 1].set_ylabel('$K_H$ from ML [mol/kg/Pa]', fontsize = 12)
        axes[i - 1].set_xscale('log')
        axes[i - 1].set_yscale('log')
        axes[i - 1].set_xlim([1e-60, 1e15])
        axes[i - 1].set_ylim([1e-60, 1e15])
        axes[i - 1].set_xticks(np.logspace(-55, 15, 8))
        axes[i - 1].set_yticks(np.logspace(-55, 15, 8))
        axes[i - 1].set_title('split #{}'.format(i))
    
    plt.tight_layout()
    plt.savefig('{}/figure3.png'.format(figure))

def figure4():
    fig, axes = plt.subplots(2, 2, figsize = (9.5, 9), dpi = 150)
    axes = axes.ravel()

    space = [1e-9, 1e17]

    for i in range(len(axes)):
        axes[i].set_xscale('log')
        axes[i].set_yscale('log')
        axes[i].set_xlim(space)
        axes[i].set_ylim(space)
        axes[i].set_xticks(np.logspace(-5, 15, 5))
        axes[i].set_yticks(np.logspace(-5, 15, 5))
        axes[i].set_xlabel('$S_{^{A}/_{B}}$ by GCMC', fontsize = 'large')
        axes[i].set_ylabel('$S_{^{A}/_{B}}$ by ML', fontsize = 'large')
        axes[i].plot(space, space, 'k-', lw = 1)

    letter = ['a', 'b', 'c', 'd']
    sel = pd.read_csv('{}/selectivity_valid_300.tsv'.format(TSV), sep = '\t')
    #S = [0.900, 0.838, 0.392, 0.404]

    for j, pair in enumerate([('propionitrile', '1,5-heptadiene'), ('methyl isopropyl ether', 'neopentane'), ('ethylamine', 'dimethylamine'), ('propene', 'propane')]):
        mol_1, mol_2 = pair

        for i in range(1, seed):
            df = pd.read_csv('{}/{}/valid_K.tsv'.format(TSV, i), sep = '\t')

            mol_1_index = df[df.molecule == mol_1].set_index('MOF')[['K', 'predict']].applymap(lambda x: x > 1e-15).all(axis = 1).values
            df_1 = df[df.molecule == mol_1].loc[mol_1_index].set_index('MOF')[['K', 'predict']]

            mol_2_index = df[df.molecule == mol_2].set_index('MOF')[['K', 'predict']].applymap(lambda x: x > 1e-15).all(axis = 1).values
            df_2 = df[df.molecule == mol_2].loc[mol_2_index].set_index('MOF')[['K', 'predict']]
            
            join = df_1.join(df_2, on = ['MOF'], how = 'inner', lsuffix = '_1', rsuffix = '_2')

            join['S_ML'] = join.predict_1 / join.predict_2
            join['S_GCMC'] = join.K_1 / join.K_2

            axes[j].scatter(join.S_GCMC, join.S_ML, marker = '.', alpha = .7, label = 'split #{}'.format(i))
            axes[j].set_title('{} / {}'.format(mol_1, mol_2))
            
        S = sel[np.logical_and(sel.mol_1 == mol_1, sel.mol_2 == mol_2)]['mean'].values[0]

        text = '({})'.format(letter[j]) + ' S' + ' = {:.3f}'.format(S)
        axes[j].text(1e-8, 1e15, text, fontsize = 'large')

    handles, labels = axes[0].get_legend_handles_labels()
    lgd = axes[0].legend(handles = handles, labels = labels, loc = 'center', ncol = 5, bbox_to_anchor = (0.5, -0.02), frameon = True, bbox_transform = fig.transFigure)
    plt.tight_layout()
    plt.savefig('{}/figure4.png'.format(figure), bbox_extra_artists = (lgd,), bbox_inches = 'tight')

def figure5():
    K = np.log10(train_valid.K)
    H = train_valid.H

    fig, ax = plt.subplots(figsize = (5, 4.5), dpi = 150)

    ax.scatter(K, H, marker = '.', alpha = .4)
    ax.set_xlabel('log($K_H$)')
    ax.set_ylabel('$\Delta H_{ads}$ [kJ/mol]')

    plt.tight_layout()
    plt.savefig('{}/figure5.png'.format(figure))

def figure6():
    fig, axes = plt.subplots(2, 2, figsize = (9.5, 9), dpi = 150)
    axes = axes.ravel()

    space = [1e-9, 1e17]

    for i in range(len(axes)):
        axes[i].set_xscale('log')
        axes[i].set_yscale('log')
        axes[i].set_xlim(space)
        axes[i].set_ylim(space)
        axes[i].set_xticks(np.logspace(-5, 15, 5))
        axes[i].set_yticks(np.logspace(-5, 15, 5))
        axes[i].set_xlabel('$S_{^{A}/_{B}}^{373}$ by GCMC', fontsize = 'large')
        axes[i].set_ylabel('$S_{^{A}/_{B}}^{373}$ by ML', fontsize = 'large')
        axes[i].plot(space, space, 'k-', lw = 1)

    letter = ['a', 'b', 'c', 'd']
    sel = pd.read_csv('{}/selectivity_valid_373.tsv'.format(TSV), sep = '\t')
    #S = [0.900, 0.838, 0.392, 0.404]

    for j, pair in enumerate([('propionitrile', '1,5-heptadiene'), ('methyl isopropyl ether', 'neopentane'), ('ethylamine', 'dimethylamine'), ('propene', 'propane')]):
        mol_1, mol_2 = pair

        for i in range(1, seed):
            K = pd.read_csv('{}/{}/valid_K.tsv'.format(TSV, i), sep = '\t')
            H = pd.read_csv('{}/{}/valid_H.tsv'.format(TSV, i), sep = '\t')

            mol_1_index = K[K.molecule == mol_1].set_index('MOF')[['K', 'predict']].applymap(lambda x: x > 1e-15).all(axis = 1).values
            K_1 = K[K.molecule == mol_1].loc[mol_1_index].set_index('MOF')[['K', 'predict']]
            H_1 = H[H.molecule == mol_1].loc[mol_1_index].set_index('MOF')[['H', 'predict']]

            mol_2_index = K[K.molecule == mol_2].set_index('MOF')[['K', 'predict']].applymap(lambda x: x > 1e-15).all(axis = 1).values
            K_2 = K[K.molecule == mol_2].loc[mol_2_index].set_index('MOF')[['K', 'predict']]
            H_2 = H[H.molecule == mol_2].loc[mol_2_index].set_index('MOF')[['H', 'predict']]

            K_1['K_373'] = henry_at_diff_temp(K_1.K, H_1.H, 300, 373)
            K_1['predict_373'] = henry_at_diff_temp(K_1.predict, H_1.predict, 300, 373)
            
            K_2['K_373'] = henry_at_diff_temp(K_2.K, H_2.H, 300, 373)
            K_2['predict_373'] = henry_at_diff_temp(K_2.predict, H_2.predict, 300, 373)
            
            join = K_1.join(K_2, on = ['MOF'], how = 'inner', lsuffix = '_1', rsuffix = '_2')

            join['S_ML'] = join.predict_373_1 / join.predict_373_2
            join['S_GCMC'] = join.K_373_1 / join.K_373_2

            axes[j].scatter(join.S_GCMC, join.S_ML, marker = '.', alpha = .7, label = 'split #{}'.format(i))
            axes[j].set_title('{} / {}'.format(mol_1, mol_2))
            
        S = sel[np.logical_and(sel.mol_1 == mol_1, sel.mol_2 == mol_2)]['mean'].values[0]

        text = '({})'.format(letter[j]) + ' S' + ' = {:.3f}'.format(S)
        axes[j].text(1e-8, 1e15, text, fontsize = 'large')

    handles, labels = axes[0].get_legend_handles_labels()
    lgd = axes[0].legend(handles = handles, labels = labels, loc = 'center', ncol = 5, bbox_to_anchor = (0.5, -0.02), frameon = True, bbox_transform = fig.transFigure)
    plt.tight_layout()
    plt.savefig('{}/figure6.png'.format(figure), bbox_extra_artists = (lgd,), bbox_inches = 'tight')

def figureS1():
    fig, axes = plt.subplots(2, 5, figsize = (20, 8), dpi = 150)
    axes = axes.ravel()

    space = [1e-9, 1e20]

    for i in range(len(axes)):
        axes[i].set_xscale('log')
        axes[i].set_yscale('log')
        axes[i].set_xlim(space)
        axes[i].set_ylim(space)
        axes[i].set_xticks(np.logspace(-5, 20, 6))
        axes[i].set_yticks(np.logspace(-5, 20, 6))
        axes[i].plot(space, space, 'k-', lw = 1)

    axes[0].set_ylabel('$S_{^{A}/_{B}}$ by ML', fontsize = 'large')
    axes[5].set_ylabel('$S_{^{A}/_{B}}$ by ML', fontsize = 'large')
    axes[7].set_xlabel('$S_{^{A}/_{B}}$ by GCMC', fontsize = 'large')

    letter = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    sel = pd.read_csv('{}/selectivity_valid_300.tsv'.format(TSV), sep = '\t')

    for j, pair in enumerate(azeo_pairs_train_valid):
        mol_1, mol_2 = pair

        for i in range(1, seed):
            df = pd.read_csv('{}/{}/valid_K.tsv'.format(TSV, i), sep = '\t')

            mol_1_index = df[df.molecule == mol_1].set_index('MOF')[['K', 'predict']].applymap(lambda x: x > 1e-15).all(axis = 1).values
            df_1 = df[df.molecule == mol_1].loc[mol_1_index].set_index('MOF')[['K', 'predict']]

            mol_2_index = df[df.molecule == mol_2].set_index('MOF')[['K', 'predict']].applymap(lambda x: x > 1e-15).all(axis = 1).values
            df_2 = df[df.molecule == mol_2].loc[mol_2_index].set_index('MOF')[['K', 'predict']]

            join = df_1.join(df_2, on = ['MOF'], how = 'inner', lsuffix = '_1', rsuffix = '_2')

            join['S_ML'] = join.predict_1 / join.predict_2
            join['S_GCMC'] = join.K_1 / join.K_2

            axes[j].scatter(join.S_GCMC, join.S_ML, marker = '.', alpha = .7, label = 'split #{}'.format(i))
            axes[j].set_title('{} - {}'.format(mol_1, mol_2))
                
        S = sel[np.logical_and(sel.mol_1 == mol_1, sel.mol_2 == mol_2)]['mean'].values[0]

        text = '({})'.format(letter[j]) + ' S' + ' = {:.3f}'.format(S)
        axes[j].text(1e-8, 1e18, text, fontsize = 'large')

    handles, labels = axes[0].get_legend_handles_labels()
    lgd = axes[0].legend(handles = handles, labels = labels, loc = 'center', ncol = 5, bbox_to_anchor = (0.5, -0.02), frameon = True, bbox_transform = fig.transFigure)
    plt.tight_layout()
    plt.savefig('{}/figureS1.png'.format(figure), bbox_extra_artists = (lgd,), bbox_inches = 'tight')

def figureS2():
    fig, axes = plt.subplots(2, 5, figsize = (20, 8), dpi = 150)
    axes = axes.ravel()

    space = [1e-9, 1e20]

    for i in range(len(axes)):
        axes[i].set_xscale('log')
        axes[i].set_yscale('log')
        axes[i].set_xlim(space)
        axes[i].set_ylim(space)
        axes[i].set_xticks(np.logspace(-5, 20, 6))
        axes[i].set_yticks(np.logspace(-5, 20, 6))
        axes[i].plot(space, space, 'k-', lw = 1)

    axes[0].set_ylabel('$S_{^{A}/_{B}}^{373}$ by ML', fontsize = 'large')
    axes[5].set_ylabel('$S_{^{A}/_{B}}^{373}$ by ML', fontsize = 'large')
    axes[7].set_xlabel('$S_{^{A}/_{B}}^{373}$ by GCMC', fontsize = 'large')

    letter = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    sel = pd.read_csv('{}/selectivity_valid_373.tsv'.format(TSV), sep = '\t')

    for j, pair in enumerate(azeo_pairs_train_valid):
        mol_1, mol_2 = pair

        for i in range(1, seed):
            K = pd.read_csv('{}/{}/valid_K.tsv'.format(TSV, i), sep = '\t')
            H = pd.read_csv('{}/{}/valid_H.tsv'.format(TSV, i), sep = '\t')

            mol_1_index = K[K.molecule == mol_1].set_index('MOF')[['K', 'predict']].applymap(lambda x: x > 1e-15).all(axis = 1).values
            K_1 = K[K.molecule == mol_1].loc[mol_1_index].set_index('MOF')[['K', 'predict']]
            H_1 = H[H.molecule == mol_1].loc[mol_1_index].set_index('MOF')[['H', 'predict']]

            mol_2_index = K[K.molecule == mol_2].set_index('MOF')[['K', 'predict']].applymap(lambda x: x > 1e-15).all(axis = 1).values
            K_2 = K[K.molecule == mol_2].loc[mol_2_index].set_index('MOF')[['K', 'predict']]
            H_2 = H[H.molecule == mol_2].loc[mol_2_index].set_index('MOF')[['H', 'predict']]

            K_1['K_373'] = henry_at_diff_temp(K_1.K, H_1.H, 300, 373)
            K_1['predict_373'] = henry_at_diff_temp(K_1.predict, H_1.predict, 300, 373)
            
            K_2['K_373'] = henry_at_diff_temp(K_2.K, H_2.H, 300, 373)
            K_2['predict_373'] = henry_at_diff_temp(K_2.predict, H_2.predict, 300, 373)
            
            join = K_1.join(K_2, on = ['MOF'], how = 'inner', lsuffix = '_1', rsuffix = '_2')

            join['S_ML'] = join.predict_373_1 / join.predict_373_2
            join['S_GCMC'] = join.K_373_1 / join.K_373_2

            axes[j].scatter(join.S_GCMC, join.S_ML, marker = '.', alpha = .7, label = 'split #{}'.format(i))
            axes[j].set_title('{} - {}'.format(mol_1, mol_2))
         
        S = sel[np.logical_and(sel.mol_1 == mol_1, sel.mol_2 == mol_2)]['mean'].values[0]
        
        text = '({})'.format(letter[j]) + ' S' + ' = {:.3f}'.format(S)
        axes[j].text(1e-8, 1e18, text, fontsize = 'large')
    
    handles, labels = axes[0].get_legend_handles_labels()
    lgd = axes[0].legend(handles = handles, labels = labels, loc = 'center', ncol = 5, bbox_to_anchor = (0.5, -0.02), frameon = True, bbox_transform = fig.transFigure)
    plt.tight_layout()
    plt.savefig('{}/figureS2.png'.format(figure), bbox_extra_artists = (lgd,), bbox_inches = 'tight')

figure3()
figure4()
figure5()
figure6()
figureS1()
figureS2()