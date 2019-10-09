def read_worldbank_csv(data_path, meta_data_path):
    """
    Read in the selected worldbank data set and join the relevant meta data.
    """
    import pandas as pd
    # Reading the csv files
    data_df = pd.read_csv(data_path, header=0, skiprows=4)
    meta_data_df = pd.read_csv(meta_data_path, header=0)
    # Perofming an inner join on the Country Code to include the Region and IncomeGroup categorizations
    merged_df = data_df.merge(right=meta_data_df[['Country Code', 'Region', 'IncomeGroup']],
                              how='inner',
                              on='Country Code')
    return merged_df

def clean_fertility_data(dataframe, sampled_years):
    """
    Generate a subset of the worldbank data to only include relevant rows and countries withou missing values.
    Calculating the countries' means, which will be analyzed, resetting the index and simplifying the column names.
    """
    # Only retaining relevant columns
    dataframe = dataframe[['Country Name', 'Country Code',
                           'Region', 'IncomeGroup']+sampled_years].copy()
    # Removing rows with any missing data then resetting the index
    dataframe.dropna(axis=0, how='any', inplace=True)
    dataframe.reset_index(inplace=True, drop=True)
    # Calculating the mean of the sampled years for each country
    dataframe['sampled_mean'] = dataframe[sampled_years].mean(axis=1)
    dataframe['sampled_mean'] = dataframe['sampled_mean'].apply(lambda x: round(x, 4))
    # Renaming the columns
    dataframe.rename(columns={'Country Name': 'country_name',
                              'Country Code': 'country_code',
                              'Region': 'region',
                              'IncomeGroup': 'income_group'},
                     inplace=True)
    return dataframe

def dist_plots(dfs_list, titles_list, sample_col):
    """
    Generating Seaborn distribution plots with the mean delineated for the desired subsets of the data.
    """
    import math
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import skew
    import numpy as np
    length = len(titles_list)
    ncols = min(length, 3)
    nrows = math.ceil(length / 3)
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(20, nrows*5))
    row = 0
    col = 0
    for i in range(length):
        if nrows > 1:
            ax = axes[row, col]
        else:
            ax = axes[col]
        data = dfs_list[i][sample_col]
        mean = round(data.mean(), 3)
        sns.distplot(data, ax = ax)
        ax.set_ylim(top=1)
        ax.axvline(x=mean, ymin=0, ymax=1.1, color='red')
        note = "Mean = "+str(mean)+", Skew = "+str(round(skew(data),3))
        ax.annotate(s=note,
                    xy=(mean, ax.get_ylim()[1]/2),
                    xytext=(30, 58),
                    textcoords='offset points')
        ax.set_title(titles_list[i])
        # interval_start = round(np.percentile(data, 5), 3)
        # interval_end = round(np.percentile(data, 95), 3)
        # interval_note = "5-95% percentile interval = "+str(interval_start)+' - '+str(interval_start)
        # ax.axvline(x=interval_start, ymin=0, ymax=1.1, color='purple')
        # ax.axvline(x=interval_end, ymin=0, ymax=1.1, color='purple')
        # ax.annotate(s=interval_note,
        #             xy=(interval_end, ax.get_ylim()[1]/2),
        #             xytext=(30, 30),
        #             textcoords='offset points')
        if col < 2:
            col += 1
        else:
            col = 0
            row += 1
    plt.subplots_adjust(hspace=0.3)

def three_way_welch_test(series_list, groups_list, alpha):
    import pandas as pd
    from itertools import combinations
    from scipy.stats import ttest_ind
    combos = list(combinations([0,1,2], r=2))
    cols = ['Group_1', 'Group_2', 't-stat', 'p_value', 'Reject']
    output = []
    for combo in combos:
        temp_list = [groups_list[combo[0]], groups_list[combo[1]]]
        stat, p = ttest_ind(series_list[combo[0]],
                            series_list[combo[1]],
                            equal_var=False)
        temp_list.append(round(stat, 4))
        temp_list.append(round(p, 4))
        if p < alpha:
            temp_list.append('True')
        else:
            temp_list.append('False')
        output.append(temp_list)
    df = pd.DataFrame(output, columns=cols)
    return df

def point_estimate_distributions(sample_input, sub_samples_list, sub_samples_titles):
    """
    Generate a distribution from random sub-samples of a given sample with mean, skew and confidence interval statistics.
    Also plot the cumulative distribution with each sub-samples mean and
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import skew

    point_estimates = []
    # create a large set of random samples from the sample
    points = 100000
    for x in range(points):
        sample = np.random.choice(
            sample_input, size=50)
        point_estimates.append(sample.mean())

    # Calculating statistics: mean, skew, confidence interval
    point_estimates = np.asarray(point_estimates)
    point_estimates.sort()
    interval_start = np.percentile(a=point_estimates, q=5)
    interval_end = np.percentile(a=point_estimates, q=95)
    samp_mean = round(np.mean(point_estimates), 3)
    samp_skew = round(skew(point_estimates), 3)

    # Create figure for plots
    fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(8, 12))
    pdf = axes[0]
    cdf = axes[1]

    # Plot the pdf and statistics
    sns.distplot(point_estimates, ax=pdf)
    pdf.set_title(
        "Estimated Sample PDF excluding SS Africa and S Asia")

    pdf.axvline(x=samp_mean, ymin=0, ymax=1, color='red')
    pdf.axvline(x=interval_start, ymin=0, ymax=1, color='purple')
    pdf.axvline(x=interval_end, ymin=0, ymax=1, color='purple')

    mean_note = "Mean = "+str(samp_mean)+", Skew = "+str(samp_skew)
    pdf.annotate(s=mean_note,
                 xy=(samp_mean, pdf.get_ylim()[1]/2),
                 xytext=(270, 40),
                 textcoords='offset points')

    confidence_note = "95% confidence interval = " + \
        str(round(interval_start, 3))+' - '+str(round(interval_end, 3))
    pdf.annotate(s=confidence_note,
                 xy=(samp_mean, pdf.get_ylim()[1]/2),
                 xytext=(270, 20),
                 textcoords='offset points')

    # Plot the cdf and statistics
    sns.distplot(point_estimates, ax=cdf,
                 hist_kws=dict(cumulative=True),
                 kde_kws=dict(cumulative=True))
    cdf.set_title(
        "Estimated Sample CDF excluding SS Africa and S Asia")

    sub_sample_means = []
    sub_sample_cdf_pcts = []
    for samp in sub_samples_list:
        samps_mean = round(samp.mean(), 3)
        sub_sample_means.append(samps_mean)
        samp_pct = round((sum((point_estimates <= samps_mean)) / points)*100, 2)
        sub_sample_cdf_pcts.append(samp_pct)

    cdf.axvline(x=sub_sample_means[0],
                ymin=0, ymax=1, color='red')
    cdf.axvline(x=sub_sample_means[1],
                ymin=0, ymax=1, color='green')
    cdf.axvline(x=sub_sample_means[2],
                ymin=0, ymax=1, color='orange')

    samp1_note = sub_samples_titles[0] + " (red): Mean = "+str(
        sub_sample_means[0]) + " & P(<= its Mean) = "+str(sub_sample_cdf_pcts[0]) + "%"
    samp2_note = sub_samples_titles[1] + " (green): Mean = "+str(
        sub_sample_means[1]) + " & P(<= its Mean) = "+str(sub_sample_cdf_pcts[1]) + "%"
    samp3_note = sub_samples_titles[2] + " (orange): Mean = "+str(
        sub_sample_means[2])+" & P(<= its Mean) = "+str(sub_sample_cdf_pcts[2]) + "%"

    cdf.annotate(s=samp1_note,
                 xy=(sub_sample_means[0], cdf.get_ylim()[1]/2),
                 xytext=(340, 60),
                 textcoords='offset points')
    cdf.annotate(s=samp2_note,
                 xy=(sub_sample_means[0], cdf.get_ylim()[1]/2),
                 xytext=(340, 40),
                 textcoords='offset points')
    cdf.annotate(s=samp3_note,
                 xy=(sub_sample_means[0], cdf.get_ylim()[1]/2),
                 xytext=(340, 20),
                 textcoords='offset points')

    plt.subplots_adjust(hspace=0.3)
    plt.show()