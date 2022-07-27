import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import warnings
import textwrap


def load_phenom(y_group="_miller_mh_v1",
                BASE_FOLDER='/Users/karinsaltoun/Documents/UKBB_files'):
    """
    Load in behavioural phenotypes and auxillary files
    y_group is a relic of my naming conventions
    Output is:
    ukbb_y: Behavioural phenotype, 40681 ppl x 978 columns
            There are 977 behaviours and 1 'eid' column (called userID)
    y_desc_dict: Maps column names in ukbb_y to a descriptive
                 explanation of what they represent
                 Note that in the case of categorical values,
                 it may be necessary to go into the data showcase to determine
                 what is being encoded by the column (check 'type' of output)
    y_cat_dict: Maps column to the broad category it belongs to
    """
    # Y VALUES
    # Import from ukbb_files, which was created from funpack
    fname = os.path.join(BASE_FOLDER, f'ukbb{y_group}_phes_filled.csv')
    ukbb_y = pd.read_csv(fname, low_memory=False, index_col=0)
    fname = os.path.join(BASE_FOLDER, f'ukbb{y_group}_description.txt')
    y_desc_dict = pd.read_csv(fname, sep='\t', header=None,
                              index_col=0).to_dict()[1]
    fname = os.path.join(BASE_FOLDER, f'ukbb{y_group}_cat_map.csv')
    y_cat_dict = pd.read_csv(fname, index_col=0)

    return ukbb_y, y_desc_dict, y_cat_dict


def phenom_correlat(df_PCA_nlz, ukbb_y, y_desc_dict, y_cat_dict):
    """
    Most important function imo
    Conducts correlations between brain and 977 input phenotypes
    Will likely take the longest to run

    INPUT NOTES:
    First dataframe is brain related phenotypes
    (or generally what is being compared against phenotypes)
    It is expected to have an 'eid' column LABELED 'eid'
    to connect it to the phenotype file

    Second dataframe is behaviour phenotypes
    obtained through the load phenom function
    Should have a 'userID' column labeled 'userid'
    to connect it to the brain data

    OUTPUT NOTES:
    Returns a dataframe containing 977 rows x n columns
    There are 4 columns which describe the phenotype
    (biobank designation, category)
    one column is an index
    and 3 x m columns, where m is the number of brain-related
    columns to compare behaviour to
    r value, p value, and -log10(p) is retained
    for each behaviour/brain related combo

    COLUMN NOTES
    For a column named COL-#.0
    the # indicates the instance
    Following UKBB convention

    For a column named COL_#.0
    the # indicates the response type
    i.e. this applies for categorical responses
    what the response indicates (e.g. response values 2)
    can be found on the UKBB Datashowcase
    This naming follows PHESANT convention
    """
    if 'ukbb_y' not in locals():
        print("phenomes not found, now loading")
        ukbb_y, y_desc_dict, y_cat_dict = load_phenom()

    s1 = pd.merge(df_PCA_nlz, ukbb_y, how='inner',
                  right_on='userID', left_on='eid')

    new_use = s1.drop(columns=['eid', 'userID'])

    # MAKE data that is used in the
    # comp_nlz = np.arange(df_PCA_nlz.shape[1])
    comp_nlz = df_PCA_nlz.columns
    comp_nlz = comp_nlz[comp_nlz != 'eid']
    cols = ukbb_y.columns.values
    cols = cols[cols != 'userID']

    keys = np.concatenate([
        ['catid', 'coid', 'phesid', 'type'], [f"-logp_{c}" for c in comp_nlz],
        [f"r_{c}" for c in comp_nlz], [f"p_{c}" for c in comp_nlz]
    ])

    mnhtn_data = {key: [] for key in keys}

    for col in cols:
        # I forget why I had this code, if we get an error add this back in
        # if len(np.unique(new_use[col])) == 0:
        #    continue
        # Dealing with funpack/phesant induced quirks (lol)
        # First record column and category
        mnhtn_data['phesid'].append(col)
        col_v = col.split('_')[0].split('#')[0]
        col_v = [ii for ii in y_cat_dict.index
                 if ii.startswith(col_v + '-')][0]
        mnhtn_data['catid'].append(y_cat_dict.loc[col_v]['Cat_ID'])
        if ('_' in col):
            mnhtn_data['coid'].append(col)
        else:
            mnhtn_data['coid'].append(col_v)

        if ('#' in col):
            mnhtn_data['type'].append(col.split('#')[1])
        else:
            mnhtn_data['type'].append('')

        for comp in comp_nlz:

            # we need to deal with nan values
            keep = ~np.logical_or(np.isnan(new_use[comp]),
                                  np.isnan(new_use[col]))
            if keep.sum() >= 2:
                r, p = pearsonr(new_use[comp][keep], new_use[col][keep])
                mnhtn_data[f"r_{comp}"].append(r)
                mnhtn_data[f"p_{comp}"].append(p)
                mnhtn_data[f"-logp_{comp}"].append(-np.log10(p))
            else:
                mnhtn_data[f"r_{comp}"].append(np.nan)
                mnhtn_data[f"p_{comp}"].append(np.nan)
                mnhtn_data[f"-logp_{comp}"].append(np.nan)

    # Manually fix miscast data
    FEV_miscast = ['20152-0.0', '20153-0.0', '20154-0.0']
    clreddf = pd.DataFrame(mnhtn_data)

    for FEV in FEV_miscast:
        clreddf.loc[clreddf.coid == FEV, 'catid'] = 20

    clreddf = clreddf.sort_values(by=['catid'])
    clreddf = clreddf.reset_index(drop=True)
    clreddf['i'] = clreddf.index

    clreddf = rearrange(clreddf, y_cat_dict)
    return clreddf.copy()


def rearrange(clreddf, y_cat_dict):
    """
    Internal Function
    Likely won't need this from an outsider perspective

    This function rearranges the order of categories to put bone density
    between blood and cognitive
    instead of between physical general and cardiac

    I prefer this order bc it spaces out the cat labels on x-axis of plot
    """
    new_idx = np.arange(len(np.unique(clreddf['catid'])))
    new_idx[8] = 6
    new_idx[6:8] = new_idx[6:8] + 1

    clreddf['true_catid'] = clreddf['catid']

    # Rearrange bone density to be right before cogn for better readibility
    clreddf.loc[(clreddf.catid == 21), 'catid'] = 31

    clreddf = clreddf.sort_values(by=['catid'])
    clreddf = clreddf.reset_index(drop=True)
    clreddf['i'] = clreddf.index

    return clreddf


def cat_name(clreddf, y_cat_dict, max_chars=19):
    """
    INPUT: correlation dataframe, category dataframe
    OUTPUT: Cleaned up category names with line breaks and shorter names
    Shorten category names
    to make x axis labels more compact

    For long category labels, add a newline break (\\n)
    so category name take up two lines not one

    By default, the cut happens such that no line is greater than 19 chars
    As this is the best definition I've found

    But you can change that by changing the max_chars parameter
    """
    cat_uniq = np.unique(clreddf['catid'])
    cat_uniq = [c if c != 31 else 21 for c in cat_uniq]
    cat_name = [y_cat_dict['Cat_Name'][y_cat_dict["Cat_ID"] == c][0]
                for c in cat_uniq]
    cat_name = [c.replace(' and environment', '').replace('measures ', '')
                .replace('and', '&').replace(' & blood vessels', '')
                for c in cat_name]

    new_cat_name = [textwrap.wrap(lbl, width=max_chars, break_on_hyphens=False)
                    for lbl in cat_name]
    new_cat_name = np.asarray(['\n'.join(lbl) for lbl in new_cat_name])

    return new_cat_name


def findFDR(clreddf, col, thresBon):
    """
    Finds FDR threshold in the standard method
    As used in the sklearn function, but modified to determine
    the p value threshold in real terms
    rather than chose which features to keep, which is what sklearn does
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFdr.html
    """
    sv = np.sort(clreddf[f'p_{col}'])
    sv = sv[~np.isnan(sv)]

    # The magic part of FDR
    # Each p-val gets compared to the Bon thres of n - w tests
    # where w is the order (smallest to largest) of the p value of interest
    # and n is the total number of tests
    sel = sv[sv <= thresBon*np.arange(1, 1 + len(sv))]

    if len(sel) > 0:
        thresFDR = (sv <= sel.max()).sum() * thresBon
    else:
        thresFDR = thresBon
    return(thresFDR)


def get_colors(n_colors):
    """
    Specifically meant for the color arrangement of the color plots
    Two rearrangements from the default color palette are made
    Grey moves from color #8 to #6 (to correspond with physical general)
    Brown moves from color #6 to #9 (to correspond with bone density)
    Yellow moves from color #9 to #8 (to correspond with blood assays)
    """
    # Get default pallete
    colors = sns.color_palette('bright', n_colors=n_colors)

    # Move Grey to correct area
    c = colors[5]
    colors[5] = colors[7]
    colors[7] = c

    # Switch brown and yellow
    c = colors[7]
    colors[7] = colors[8]
    colors[8] = c
    return colors


def manhattan_plot(clreddf, lk, new_cat_name, new_idx=None,
                   thres=0.05, ylim=None, plot_height=7):
    """
    Create a manhattan plot based on info in clreddf
    ylim changes the bounds of the plot
    ylim should be a tuple
    """
    # ylim should be a tuple
    # Find FDR Thres
    n_t = clreddf.shape[0]
    thresBon = thres/n_t

    new_idx = np.arange(len(new_cat_name)) if new_idx is None else new_idx

    thresFDR = findFDR(clreddf, lk, thresBon)

    colors = get_colors(n_colors=len(new_cat_name))

    plot = sns.relplot(data=clreddf, x='i', y=f'-logp_{lk}',  edgecolor='k',
                       aspect=1.3, height=plot_height, hue='catid',
                       palette=colors, legend=None)
    t_df = clreddf.groupby('catid')['i'].median()
    t_dfm = clreddf.groupby('catid')['i'].max()[:-1]
    # Shift exercise and work and bone density
    # a bit to the right for enhanced readibility
    t_df[31] = t_df[31] + 20  # Blood Assays
    t_df[11] = t_df[11] + 20  # Exercise + Work

    plot.ax.set_ylabel('$-\\log_{10}(p)$')
    plot.ax.set_xlabel('')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot.ax.set_xticks(t_df)
        plot.ax.set_xticklabels(new_cat_name[new_idx], rotation=90, ha='right')

    for xtick, color in zip(plot.ax.get_xticklabels(), colors):
        xtick.set_color(color)

    plt.tick_params(axis='x', bottom=False)
    plot.fig.suptitle(f'Manhattan plot of {lk}')
    plt.axhline(y=-np.log10(thresFDR), color='k')
    plt.axhline(y=-np.log10(thresBon), color='k')
    [plt.axvline(x=xc, color='k', linestyle='--') for xc in t_dfm]

    if ylim:
        plot.fig.tight_layout()
        plot.set(ylim=ylim)
        plot.fig.tight_layout()
        locs, labels = plt.yticks()
        plt.yticks([*locs, -np.log10(thresBon), -np.log10(thresFDR)],
                   [*labels, 'BON', "FDR"])
    else:
        locs, labels = plt.yticks()
        plt.yticks([*locs, -np.log10(thresBon), -np.log10(thresFDR)],
                   [*labels, 'BON', "FDR"])
    plot.fig.tight_layout()
    return


def miami_plot(df_top, df_bot, lk, new_cat_name, lbls=None,
               new_idx=None,  thres=0.05, ylim=None, figsize=(10, 10)):
    """
    Creates a Miami plot, i.e. two manhattan plots,
    one above the x-axis (df_top) one below (df_bot)
    Expects TWO dataframes containing output from phenom_correlat function
    That is, all -logp and p values associated with
    all 977 behavioural phenotypes

    It is expected that the column names are the same in both dataframes
    That is, the column of interest exists in both dataframs
    and lk takes the same column from both dataframes to plot

    This function cannot plot two columns from the same dataframe

    lbls should be a tuple or list of size two
    containing the names of the groups
    e.g. Young/Old; Male/Female; High IQ/Low IQ; Lonely/Not Lonely etc.

    ylim should be a single value (e.g. 5)
    which indicates the upper/lower bound of the plot
    If unselected, it will be chosen as the largest -log(p) vals available
    """
    # Make a copy so you don't alter input dataframes
    clreddf = df_top.copy()
    clreddf2 = df_bot.copy()

    # ylim should be a single value
    # Find FDR Thres
    n_t = clreddf.shape[0]
    thresBon = thres/n_t

    new_idx = np.arange(len(new_cat_name)) if new_idx is None else new_idx

    clreddf.replace([np.inf, -np.inf], np.nan, inplace=True)
    clreddf2.replace([np.inf, -np.inf], np.nan, inplace=True)

    thresFDR = findFDR(clreddf, lk, thresBon)
    thresFDR2 = findFDR(clreddf2, lk, thresBon)

    if (clreddf2.phesid == clreddf.phesid).sum() != n_t:
        # The columns are out of order
        # We need to rearrange them
        clreddf2.index = clreddf2.phesid
        clreddf2 = clreddf2.drop('phesid', axis=1)
        clreddf2 = clreddf2.reindex(clreddf.phesid.values)
        clreddf2 = clreddf2.reset_index()
        clreddf2['i'] = clreddf2.index

    # Set up colours
    colors = get_colors(n_colors=len(new_cat_name))

    # Select/Define Limits of the graph
    if ylim:
        ymaxx = ylim
    else:
        y1 = clreddf[clreddf[f'-logp_{lk}'].notnull()][[f'-logp_{lk}']]
        y1 = np.abs(np.array(y1)).max()
        y2 = clreddf2[clreddf2[f'-logp_{lk}'].notnull()][[f'-logp_{lk}']]
        y2 = np.abs(np.array(y2)).max()
        ymaxx = max(y1, y2) + 1

    # Begin Plotting
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=clreddf, x='i', y=f'-logp_{lk}', edgecolor='k',
                    hue='catid', palette=colors, legend=None, ax=ax)

    # Define Boundarys of Categories
    # To make the lines (at max)
    # and for labels (max - 3)
    t_df = clreddf.groupby('catid')['i'].max() - 3
    t_dfm = clreddf.groupby('catid')['i'].max()[:-1]

    # Shift select cat labels for enhanced readibility
    cat_size = t_df + 3 - clreddf.groupby('catid')['i'].min()
    t_df[31] = t_df[31] + 24  # Bone Density
#     t_df[30] = t_df[30] - cat_size[30] + 25  # Blood Assays
#     t_df[51] = t_df[51] - cat_size[51] + 40  # Mental Health

    clreddf2[f'-logp_{lk}_flip'] = -1 * clreddf2[f'-logp_{lk}']
    sns.scatterplot(data=clreddf2, x='i', y=f'-logp_{lk}_flip', edgecolor='k',
                    hue='catid', palette=colors, legend=None, ax=ax)

    plt.ylim(-1*ymaxx, ymaxx)
    texts = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if lbls is not None:
            ylbl = f'{lbls[1]}\t'+'\t$-\\log_{10}(p)$\t' + f' \t{lbls[0]}'
        else:
            ylbl = '$-\\log_{10}(p)$'  # Latex formatting

        ax.set_ylabel(ylbl)
        ax.set_xlabel('')

        ax.set_xticklabels('', rotation=90, ha='left')
        ax.set_xticks(t_df)
        ax.xaxis.tick_top()

        for xtick, cat, color in zip(t_df, new_cat_name[new_idx],  colors):
            plt.text(xtick, -ymaxx + ymaxx/25, cat,
                     c=color, ha='right', rotation=90)

    plt.tick_params(axis='x', bottom=False, top=False)
    # fig.suptitle(f'Manhattan plot of {lk}');
    plt.axhline(y=-np.log10(thresFDR), color='k')
    plt.axhline(y=np.log10(thresFDR2), color='k')  # Lower Manhattan
    plt.axhline(y=-np.log10(thresBon), color='k')
    plt.axhline(y=np.log10(thresBon), color='k')  # Lower Manhattan
    plt.axhline(y=0, color='k')
    [plt.axvline(x=xc, color='k', linestyle='--') for xc in t_dfm]

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.tight_layout()
    labels = [item.get_text() for item in ax.get_yticklabels()]
    locs, _ = plt.yticks()
    new_text = [t.replace('−', '') for t in labels]
#     plt.yticks([*locs, -np.log10(thresBon), -np.log10(thresFDR),
#                 np.log10(thresBon), np.log10(thresFDR2)],
#                [*new_text, 'BON', "FDR", 'BON', "FDR"])
    plt.tight_layout()
    if lbls is not None:
        ypos = ymaxx - ymaxx/10
        texts.append(plt.text(950 - 8*len(lbls[0]), ypos, lbls[0]))
        texts.append(plt.text(950 - 8*len(lbls[1]), -1*ypos, lbls[1]))
    # adjust_text(texts)
    return


def highlight(cols, lk, clreddf,  fmt='rx'):
    """
    Will annotate the latest figure
    fmt param determines appearance (e.g. color, shape)
    cols is a selection of columns that will be highlighted

    Check here for a collection of point shape
    https://matplotlib.org/stable/api/markers_api.html
    and here for more info on how to format them (and colors)
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
    You will have to scroll to Notes - Format Strings for the second link
    """
    if not isinstance(cols, (list, np.ndarray)):
        cols = [cols]
    for col in cols:
        if col not in clreddf.coid.values:
            print(f"Column {col} not found. Skipping")
        else:
            plt.plot(clreddf[clreddf['coid'] == col]['i'].values,
                     clreddf[clreddf['coid'] == col][f'-logp_{lk}'].values,
                     fmt)
    return


def hits(clreddf, lk, y_desc_dict, y_cat_dict, to_file=False, hits_file=None,
         extra_notes=None, thresBon=None, thres=0.05, useFDR=True):
    """
    Output all hits associated with lk in clreddf

    By default outputs everything above FDR threshold
    can change to only Bon hits with useFDR=False

    By default, it will print hits to terminal

    If you want to save to a file, you need to_file=True
    By default, it prints to hits_{lk}.txt if to_file is True
    you can customize which file it goes to by passing a filepath to hits_file
    Note that hits_file doesn't do anything unles to_file is set to true

    you can add more information to output with extra_notes param

    Format is a tuple with the following value
    (x val in graph, choice, column name, -log10(p), descriptive column name)

    choice exists for categorical varaiables
    You would need to check the data showcase to confirm what it encodes

    COLUMN NOTES
    for a column named COL-#.0
    the # indicates the instance
    Following UKBB convention

    For a column named COL_#.0
    the # indicates the response type
    i.e. this applies for categorical responses
    what the response indicates (e.g. response values 2)
    can be found on the UKBB Datashowcase
    This naming follows PHESANT convention
    """
    n_t = clreddf.shape[0]
    if thresBon is None:
        thresBon = thres/n_t
    thresFDR = findFDR(clreddf, lk, thresBon)

    if to_file:
        # Save a ref to the original standard output
        original_stdout = sys.stdout

        if hits_file is None:
            hits_file = f"{extra_notes}_hits_{lk}.txt"

        with open(os.path.join(hits_file), 'w') as f:
            # Change the standard output to the file we created.
            sys.stdout = f
            # Print out all hits
            printing(clreddf, lk, y_desc_dict, y_cat_dict,
                     extra_notes, useFDR, thresBon, thresFDR)
            # Reset the standard output to its original value
            sys.stdout = original_stdout
    else:
        # Print hits to original output (terminal)
        printing(clreddf, lk, y_desc_dict, y_cat_dict,
                 extra_notes, useFDR, thresBon, thresFDR)
    return


def printing(clreddf, lk, y_desc_dict, y_cat_dict,
             extra_notes, useFDR, thresBon, thresFDR):
    """
    Auxillary function called by hits
    Created to aid readability
    and make it possible to print to file (or not)
    using the same function
    """
    # This prints out all significant columns for the current component
    print('\nComponent', lk)
    if extra_notes is not None:
        print(extra_notes)

    cn = lk
    # Find FDR Thres
    print(f"MAX: {clreddf[f'-logp_{cn}'].max():.3f}")
    print(f"N ABOVE BON ({-np.log10(thresBon):.2f}): ",
          (clreddf[f'-logp_{cn}'] > -np.log10(thresBon)).sum())
    print(f"N ABOVE FDR ({-np.log10(thresFDR):.2f}): ",
          (clreddf[f'-logp_{cn}'] > -np.log10(thresFDR)).sum())

    thresUsed = thresFDR if useFDR else thresBon
    print("Fine Grain info above ", 'FDR' if useFDR else 'Bon')
    print("Printing out " +
          str((clreddf[f'-logp_{cn}'] > -np.log10(thresUsed)).sum())
          + " hits")

    sigcol = clreddf[(clreddf[f'-logp_{lk}'] > -np.log10(thresUsed))]

    for catn in np.unique(sigcol['catid']):
        if catn == 31:
            catn = 21
        sigcol_subset = sigcol[sigcol['catid'] == catn]
        cs = [c for c in sigcol_subset['coid']]

        if len(cs) > 0:
            print(f"\n COMP {lk}",
                  y_cat_dict[y_cat_dict['Cat_ID'] == catn]
                  ['Cat_Name'][0].upper(),
                  f" ({len(sigcol_subset)} hit(s))")

            new_cs = []
            for c in cs:
                if '_' in c:
                    col_v = c.split('_')[0].split('#')[0]
                    col_v = [ii for ii in y_cat_dict.index
                             if ii.startswith(col_v+'-')][0]
                else:
                    col_v = c
                new_cs.append(col_v)
            print(*[(iv, tp,  c, f"{pp:.2f}", y_desc_dict[new_cs[ii]])
                    for ii, (iv, c, pp, tp) in enumerate(
                    sigcol_subset[['i', 'coid', f'-logp_{lk}', 'type']]
                    .values)], sep='\n')
    return
