#%%
import numpy as np
import pandas as pd
from bokeh.plotting import figure, show, output_file, save, curdoc
from bokeh.models import ColumnDataSource, HoverTool, NumeralTickFormatter, Title
from bokeh.models.widgets import Select
from bokeh.layouts import column, row, gridplot
from datetime import datetime as dt
from math import pi, sqrt, ceil
import os
import csv
import statistics
import matplotlib.pyplot as plt


# Scenarios Titles
experiment_labels = ['scenario 1', 'scenario 2', 'scenario 3']
all_scenario_labels = ['fishes 1', 'scenario 1', 'fishes 2', 'scenario 2', 'fishes 3', 'scenario 3']

# Ranges for plots
min_range = 200
max_range = -100

# Setting the threshold to have a good sensors signal - 4 HSI possible values: 1=Good, 2=Medium, 4=Bad
hsi_threshold = 8
windows_no = 30


#%%
# 0. sequential processing of participant data
def execute_class(participant_no, experiment_order):

    # Read the file
    df = read_file(participant_no)
    
    # Clean the data and split it based on content
    df_records, df_elements, df_markers = clean_data(df)

    # Checking if the data returned is valid
    if str(type(df_records)) != "<class 'str'>":
        
        # Calculating the frequencies using the correct range
        df_prepared = extract_features(df_records)
        #visualize_plot([df_prepared], "initial", ['All'])
        
        # Split the data into baseline data and experiment data
        sections = split_sections_with_markers(df_prepared, df_markers)
        #visualize_plot(sections, "sections", all_scenario_labels)
        
        # Get basic general statistical data from sections
        stat_values_participant_df = get_stats_for_participant_sections(participant_no, sections)

        # Calculate ratios
        ratio_values_participant_df = calculate_ratios_of_participant(stat_values_participant_df)

        # Splitting data into comparable windows

        return stat_values_participant_df, ratio_values_participant_df



#%%
# 1. Load the dato into dataframe
def read_file(participant_no):
    __file__ = participant_no + '.csv'
    my_absolute_dirpath = os.path.abspath(os.path.dirname(__file__))
    file_path = my_absolute_dirpath+"\\aData\\"+__file__
    df = pd.read_csv(file_path, sep=",")
    return df

#%%
# 2. Removes unnecessary columns and checks the correctness
def clean_data(df):
    
    # Removing unnecessary columns and formatting
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], errors='coerce')
    df_cleaned = df.drop(columns=['Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z', 'AUX_RIGHT', 'Battery',
                                    'RAW_TP9', 'RAW_AF7','RAW_AF8','RAW_TP10','AUX_RIGHT'])

    # Extracting the markers from the dataframe
    df_markers = df_cleaned[df_cleaned['Elements'].str.contains('/Marker/', na=False)]
    df_markers = df_markers.reset_index(drop=True)
    df_markers = df_markers[['TimeStamp', 'Elements']]
    
    # Checking if the markers are correct - Markers are 7: 1x "marker 5", 3x "marker 1", 3x "marker 2"
    markers_no = df_markers.shape[0]
    marker_five_no = df_markers[df_markers['Elements'] == '/Marker/5'].shape[0]
    marker_one_no = df_markers[df_markers['Elements'] == '/Marker/1'].shape[0]
    marker_two_no = df_markers[df_markers['Elements'] == '/Marker/2'].shape[0]
    if not (markers_no == 7 and marker_five_no == 1 and marker_one_no == 3 and marker_two_no == 3):
        print("Markers are not of the desired number or type")
        print("Marker 5: " +str(marker_five_no))
        print("Marker 2: " +str(marker_two_no))
        print("Marker 1: " +str(marker_one_no))
        print("Total Markers: "+str(markers_no))
        return '', '', ''
    
    # Deleting all records before the initial marker timestamp (time spent positioning the sensor)
    df_start_markers = df_markers[df_markers['Elements'] == '/Marker/5']
    timestamp_start = df_start_markers['TimeStamp'].iloc[-1]
    df_formatted = df_cleaned[df_cleaned['TimeStamp'] >= timestamp_start]
    
    # Removing marker 5 from all dataframes
    df_markers = df_markers.iloc[1:]
    df_formatted = df_formatted.iloc[1:]

    # Extracting the elements - markers, blinks, clinches
    df_elements = df_formatted[df_formatted['Elements'].str.contains('/', na=False)]
    df_elements = df_elements[['TimeStamp', 'Elements']]
    df_elements = df_elements.dropna()
    df_elements = df_elements.reset_index(drop=True)

    # Extracting the records with the HeadBandOn
    df_records_headbandon = df_formatted[df_formatted['HeadBandOn'] == 1]
    df_records_headbandon = df_records_headbandon.drop(columns=['Elements'])
    df_records_headbandon = df_records_headbandon.reset_index(drop=True)
    
    # Removing low quality data
    df_records_headbandon['Sensor_Quality'] = df_records_headbandon['HSI_TP9'] + df_records_headbandon['HSI_AF7'] + df_records_headbandon['HSI_AF8'] + df_records_headbandon['HSI_TP10']
    df_records = df_records_headbandon[df_records_headbandon['Sensor_Quality'] <= hsi_threshold]
    
    # Defining and printing low quality data proportion
    lines = df_records_headbandon.shape[0]
    usable_lines = df_records.shape[0]
    print(str(usable_lines) + "/" + str(lines) + " usable lines ")
    print("Total usable lines for this participant is "+
          ('%.2f' % (100*usable_lines/lines,)).rstrip('0').rstrip('.')+
          "% - threshold: "+str(hsi_threshold))
    print()
    
    # remove unnecesary columns
    df_records = df_records.drop(columns=['HeadBandOn','HSI_TP9','HSI_AF7','HSI_AF8','HSI_TP10','Sensor_Quality'])

    return df_records, df_elements, df_markers

#%%
# 3. Transforming and extracting data, including data split and normalization
def extract_features(df_records):
    
    # Settings to change the data range
    old_range_max = 3
    old_range_min = -3
    old_range = (old_range_max - old_range_min) 
    new_range_max = 100    # 200
    new_range_min = 0   # -100
    new_range = (new_range_max - new_range_min)

    # Calculating Average Absolute Brain Waves
    df_avg_bw = df_records.drop(columns=['Delta_TP9', 'Delta_AF7', 'Delta_AF8', 'Delta_TP10', 'Theta_TP9', 'Theta_AF7', 'Theta_AF8', 'Theta_TP10', 'Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10', 'Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10', 'Gamma_TP9', 'Gamma_AF7', 'Gamma_AF8', 'Gamma_TP10'])
    original_alpha_average = (df_records['Alpha_TP9'] + df_records['Alpha_AF7'] + df_records['Alpha_AF8'] + df_records['Alpha_TP10'])/4
    original_beta_average = (df_records['Beta_TP9'] + df_records['Beta_AF7'] + df_records['Beta_AF8'] + df_records['Beta_TP10'])/4
    original_theta_average = (df_records['Theta_TP9'] + df_records['Theta_AF7'] + df_records['Theta_AF8'] + df_records['Theta_TP10'])/4

    # Converting in new range
    df_avg_bw['Alpha_Avg'] = original_alpha_average #(((original_alpha_average - old_range_min) * new_range) / old_range) + new_range_min
    df_avg_bw['Beta_Avg'] = original_beta_average #(((original_beta_average - old_range_min) * new_range) / old_range) + new_range_min
    df_avg_bw['Theta_Avg'] = original_theta_average #(((original_theta_average - old_range_min) * new_range) / old_range) + new_range_min
    
    # Calculating first ratio: Beta / (Alpha + Theta) --> task difficulty indicator + task engagement
    df_avg_bw['First_Ratio'] = df_avg_bw['Beta_Avg'] / (df_avg_bw['Alpha_Avg'] + df_avg_bw['Theta_Avg'])
        
    # Calculating second ratio: Theta / (Alpha + Beta) --> task difficulty indicator
    df_avg_bw['Second_Ratio'] = df_avg_bw['Theta_Avg'] / (df_avg_bw['Alpha_Avg'] + df_avg_bw['Beta_Avg'])

    return df_avg_bw

#%%
# 4. Splitting data based on markers
def split_sections_with_markers(df, df_markers):
    
    # Setting initial variable
    i = 0
    sections = []
    
    # Splitting the data and visualizing them separately
    prev_timestamp = "null"
    markers_timestamps = df_markers['TimeStamp'].tolist()
    for timestamp in markers_timestamps:
        
        # Taking different actions for the dataframe splitting
        if prev_timestamp != "null":
            section = df[
                df['TimeStamp'] > prev_timestamp
            ]
            section = section[
                section['TimeStamp'] <= timestamp
            ]
            
            sections.append(section)
            i=i+1
            
        prev_timestamp = timestamp

    # Visualizing the last part of the experiment
    section = df[
        df['TimeStamp'] > prev_timestamp
    ]
    sections.append(section)
    
    # Returns 6 sections: (baseline + scenario) x3
    return sections

#%%
# Get basic statistic data from sections
def get_stats_for_participant_sections(participant_no, sections):
    h = 0
    #frequency_columns = ['Alpha_Avg', 'Beta_Avg', 'Delta_Avg', 'Gamma_Avg', 'Theta_Avg']
    frequency_columns = ['Alpha_Avg', 'Beta_Avg', 'Theta_Avg']
    section_type = ''

    participant_total_df = pd.DataFrame() 
    
    for section_nr in range(len(sections)):
        # Storing the statistical data of the section
        curr_section = sections[section_nr]
        h = h + 1
        section_first_timestamp = curr_section.iloc[0]['TimeStamp']
        section_last_timestamp = curr_section.iloc[-1]['TimeStamp']
        print()

        # section type Fishes or Scenario
        if(section_nr in range(0,6,2)):
            section_type = 'F'
            # make additional processing
            # Extracting the last minute of recording 
            baseline_start_last_minute = section_last_timestamp - pd.Timedelta(minutes=1)
            df_baseline_last_minute = curr_section[curr_section['TimeStamp'] >= baseline_start_last_minute]
        else:
            section_type = 'S'

        block_section_number = ceil((section_nr+1 )/2)
        section_id = section_type + str(block_section_number)
        print("Frequency Statistical values for experiment " +str(h)+ " - " + section_id +":")
        print("Section " + str(section_id) + ". (" + str(section_first_timestamp.strftime("%y-%m-%d %H %M %S")) + 
        " - " + section_last_timestamp.strftime("%y-%m-%d %H %M %S") + "). Rows: " + str(len(curr_section.index))) 
        
        # set the column values F
        frequency_names_for_section = [ section_id + '_' + s for s in frequency_columns]
        stat_values_names = ['mean','min','max','median','mode','std']
        column_names_for_section  = []
        for i in range(len(frequency_columns)):
            for j in range(len(stat_values_names)):
                column_names_for_section.append(frequency_names_for_section[i] + '_' + stat_values_names[j])
        # TODO the last minute names

        section_frequencies_values_df = pd.DataFrame(columns = column_names_for_section)
        stat_vals = []
        for j in range(0,len(frequency_columns)):
            stat_vals.append(curr_section[frequency_columns[j]].mean())
            stat_vals.append(curr_section[frequency_columns[j]].min())
            stat_vals.append(curr_section[frequency_columns[j]].max())
            stat_vals.append(curr_section[frequency_columns[j]].median())
            stat_vals.append(curr_section[frequency_columns[j]].mode())
            stat_vals.append(curr_section[frequency_columns[j]].std())

            # additionally append last minute for the baseline
            #if(section_type == 'F'):
            #    section_frequencies_values.append(df_baseline_last_minute[frequency_columns[j]].mean())
        section_frequencies_values_df.loc[len(section_frequencies_values_df), :] = stat_vals
    
        # concatenate all the sections for one participant (grow column numbers)
        if section_nr == 0:
            participant_total_df = participant_total_df.append(section_frequencies_values_df)#, ignore_index = True)
        else:
            participant_total_df = pd.concat([participant_total_df,section_frequencies_values_df.reindex(participant_total_df.index)], axis=1) #, ignore_index=True)

    return participant_total_df

#%%
def calculate_ratios_of_participant(stat_values_df):
    stat_values_df.rename(columns=lambda x: x.strip())
    section_ratio_names = ['F1R1','F1R2','S1R1','S1R2','F2R1','F2R2','S2R1','S2R2','F3R1','F3R2','S3R1','S3R2']
    section_ratio_values_df = pd.DataFrame(columns = section_ratio_names)

    alpha_mean_col_names = ['F1_Alpha_Avg_mean', 'S1_Alpha_Avg_mean','F2_Alpha_Avg_mean', 'S2_Alpha_Avg_mean','F3_Alpha_Avg_mean', 'S3_Alpha_Avg_mean']
    beta_mean_col_names = ['F1_Beta_Avg_mean', 'S1_Beta_Avg_mean','F2_Beta_Avg_mean', 'S2_Beta_Avg_mean','F3_Beta_Avg_mean', 'S3_Beta_Avg_mean']
    theta_mean_col_names = ['F1_Theta_Avg_mean', 'S1_Theta_Avg_mean','F2_Theta_Avg_mean', 'S2_Theta_Avg_mean','F3_Theta_Avg_mean', 'S3_Theta_Avg_mean']

    participant_total_df = pd.DataFrame() 

    ratio_vals_per_participant = []
    for part_nr in range(len(stat_values_df)):
        ratio_vals_per_participant = []
        section_ratio_values_df = pd.DataFrame(columns = section_ratio_names)
        for section_nr in range(len(alpha_mean_col_names)):
            # Beta / (Alpha + Theta)
            ratio1 = stat_values_df.iloc[part_nr][beta_mean_col_names[section_nr]] / (stat_values_df.iloc[part_nr][alpha_mean_col_names[section_nr]] + stat_values_df.iloc[part_nr][theta_mean_col_names[section_nr]] )
            # Theta / (Alpha + Beta)
            ratio2 = stat_values_df.iloc[part_nr][theta_mean_col_names[section_nr]] / (stat_values_df.iloc[part_nr][alpha_mean_col_names[section_nr]] + stat_values_df.iloc[part_nr][beta_mean_col_names[section_nr]] )
            ratio_vals_per_participant.append(ratio1)
            ratio_vals_per_participant.append(ratio2)

        section_ratio_values_df.loc[len(section_ratio_values_df), :] = ratio_vals_per_participant
        
        # append 
        participant_total_df = participant_total_df.append(section_ratio_values_df, ignore_index = True)
    
    return participant_total_df







#%%
# -------------------Main Method-------------------------------------------

# Experiment order: R=Rational, S=StringUtil, U=UtilObject
total_stat_val_df = pd.DataFrame() 
total_ratio_val_df = pd.DataFrame() 

'''
participants = ['01','02']
experiments_order = [
    ['R', 'S', 'U'],
    ['U', 'R', 'S']
]
'''
participants = [
    '01', 
    '02',
    '03', 
    '04', 
    '05', 
    '06'
]
experiments_order = [
    ['R', 'S', 'U'],
    ['U', 'R', 'S'],
    ['S', 'U', 'R'],
    ['R', 'S', 'U'],
    ['U', 'R', 'S'],
    ['S', 'U', 'R']
]

# Iterating through the participants
for i in range(0, len(participants)):
    print("==== Participant "+participants[i]+" ====")
    st_val_df, rat_val_df = execute_class(participants[i], experiments_order[i])

    total_stat_val_df = total_stat_val_df.append(st_val_df, ignore_index = True)
    total_ratio_val_df = total_ratio_val_df.append(rat_val_df, ignore_index = True)

    print()

print("total_stat_val_df -------------")
print(total_stat_val_df.shape)

total_stat_val_df.to_csv('aResults/satistical_values.csv')
total_ratio_val_df.to_csv('aResults/ratio_values.csv')





doPlots = True
if (doPlots):
    # do some plots 

    ####################
    ## PER PARTICIPANT
    xlabel = 'Phase'
    # Alpha_Average Mean Values
    alpha_mean_col_names = ['F1_Alpha_Avg_mean', 'S1_Alpha_Avg_mean','F2_Alpha_Avg_mean', 'S2_Alpha_Avg_mean','F3_Alpha_Avg_mean', 'S3_Alpha_Avg_mean']
    title_start = 'Alpha_Avg Mean Participant '
    fig_sufix = 'Alpha_Mean_P'
    ylabel = 'Alpha_avg'
    plot_participant_stat_values_one_freq(total_stat_val_df,alpha_mean_col_names,title_start, fig_sufix, xlabel, ylabel, save_png = True)
    # Beata_Average Mean Values
    beta_mean_col_names = ['F1_Beta_Avg_mean', 'S1_Beta_Avg_mean','F2_Beta_Avg_mean', 'S2_Beta_Avg_mean','F3_Beta_Avg_mean', 'S3_Beta_Avg_mean']
    title_start = 'Beta_Avg Mean Participant '
    fig_sufix = 'Beta_Mean_P'
    ylabel = 'Beta_avg'
    plot_participant_stat_values_one_freq(total_stat_val_df,beta_mean_col_names, title_start, fig_sufix, xlabel, ylabel, save_png = True)
    # Theta_Average Mean Values
    theta_mean_col_names = ['F1_Theta_Avg_mean', 'S1_Theta_Avg_mean','F2_Theta_Avg_mean', 'S2_Theta_Avg_mean','F3_Theta_Avg_mean', 'S3_Theta_Avg_mean']
    title_start = 'Theta_Avg Mean Participant '
    fig_sufix = 'Theta_Mean_P'
    ylabel = 'Theta_avg'
    plot_participant_stat_values_one_freq(total_stat_val_df,beta_mean_col_names, title_start, fig_sufix,xlabel, ylabel, save_png = True)

    # the ratios 
    plot_participant_ratio_values(total_ratio_val_df, save_png = True)

    ####################
    ## PER 





# %%
def plot_participant_stat_values_one_freq(total_stat_val_df, col_names, title_start, fig_sufix, xlabel, ylabel, save_png = False):
    array_for_plot = []
    x_ticks_labels = ['F1','S1','F2','S2','F3','S3']
    for part_nr in range(0, len(total_stat_val_df)):
        array_for_plot = []
        array_for_plot.append(total_stat_val_df.iloc[part_nr][col_names[0]])
        array_for_plot.append(total_stat_val_df.iloc[part_nr][col_names[1]])
        array_for_plot.append(total_stat_val_df.iloc[part_nr][col_names[2]])
        array_for_plot.append(total_stat_val_df.iloc[part_nr][col_names[3]])
        array_for_plot.append(total_stat_val_df.iloc[part_nr][col_names[4]])
        array_for_plot.append(total_stat_val_df.iloc[part_nr][col_names[5]])
        title_str = title_start + str(part_nr)
        fig_name = fig_sufix + str(part_nr)
        plot_array(array_for_plot, title_str, x_ticks_labels, fig_name, xlabel, ylabel, save_png)


def get_array_for_plot_for_ratio(rv_df, part_nr, col_names):
    array_for_plot = []
    array_for_plot.append(rv_df.iloc[part_nr][col_names[0]])
    array_for_plot.append(rv_df.iloc[part_nr][col_names[1]])
    array_for_plot.append(rv_df.iloc[part_nr][col_names[2]])
    array_for_plot.append(rv_df.iloc[part_nr][col_names[3]])
    array_for_plot.append(rv_df.iloc[part_nr][col_names[4]])
    array_for_plot.append(rv_df.iloc[part_nr][col_names[5]])
    return array_for_plot

def plot_array(array_for_plot, title_str, x_ticks_labels, img_name, xlabel, ylabel, save_png):
    fig, ax = plt.subplots(1,1,tight_layout=True)
    plt.title(title_str, y=1.15, fontsize=14)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(range(len(array_for_plot)),x_ticks_labels,size='small')
    plt.plot(array_for_plot,  marker='o')
    plt.show()
    if save_png:
        fig.savefig('aResults/imgs/'+img_name+'.png')
    return ax


def plot_participant_ratio_values(total_ratio_val_df, save_png = False):
    array_for_plot = []
    x_ticks_labels = ['F1','S1','F2','S2','F3','S3']
    col_names = list(total_ratio_val_df)
    substr = 'R1'
    col_names_R1 = [s for s in col_names if substr in s]
    substr = 'R2'
    col_names_R2 = [s for s in col_names if substr in s]

    for part_nr in range(0, len(total_ratio_val_df)):
        #Ratio 1 
        title_start = 'Ratio 1 P '
        fig_sufix = 'ratio1_'
        xlabel = ''
        ylabel = ''
        array_for_plot = get_array_for_plot_for_ratio(total_ratio_val_df, part_nr, col_names_R1)
        title_str = title_start + str(part_nr)
        fig_name = fig_sufix + str(part_nr)
        plot_array(array_for_plot, title_str, x_ticks_labels, fig_name, xlabel, ylabel, save_png)

        #Ratio 2 
        title_start = 'Ratio 2 P '
        fig_sufix = 'ratio2_'
        xlabel = ''
        ylabel = ''
        array_for_plot = get_array_for_plot_for_ratio(total_ratio_val_df, part_nr, col_names_R2)
        title_str = title_start + str(part_nr)
        fig_name = fig_sufix + str(part_nr)
        plot_array(array_for_plot, title_str, x_ticks_labels, fig_name, xlabel, ylabel, save_png)





# %%
'''
#%%
# 5. Normalizing the data based on the baseline
def define_baseline(sections):
    
    # Declaring variables for the method execution
    h = 0
    total_lines = 0
    total_usable_lines = 0
    frequency_columns = ['Alpha_Avg', 'Beta_Avg', 'Theta_Avg']
    new_sections = []
    baselines = []
    
    # Iterating through the 3 baseline-experiment sections
    for i in range(0,6,2):
            
        # Defining variables for execution
        df_baseline = sections[i]
        
        # Extracting the last minute of recording 
        baseline_last_timestamp = df_baseline.iloc[-1]['TimeStamp']
        baseline_start_last_minute = baseline_last_timestamp - pd.Timedelta(minutes=1)
        df_baseline_last_minute = df_baseline[df_baseline['TimeStamp'] >= baseline_start_last_minute]
        
        # Storing the mean of the baseline
        baseline_frequencies = []
        print("Frequency Mean of baseline for experiment " +str(h+1)+":")
        for j in range(0,3):
            baseline_frequencies.append(df_baseline_last_minute[frequency_columns[j]].mean())
            print("-- " + frequency_columns[j] + ": " +('%.2f' % (baseline_frequencies[j],)).rstrip('0').rstrip('.'))
                     
        # Subtracting baseline from data
        for column in ['First_Ratio', 'Second_Ratio']:
            baseline_avg = df_baseline_last_minute[column].mean()
            baselines.append(baseline_avg)
        print()
                
        # Storing the elaborated sections
        new_sections.append(sections[i+1])
        h = h + 1
        
    return new_sections, baselines

#%%
# 6. Create windows
def split_in_windows(sections, baselines, k):
    
    # Defining return value - for each section, one array containing first_ratio.mean, second_ratio.mean
    all_section_means = []
    
    # Accessing every scenario
    for i in range(0, 3):
        
        section = sections[i]
        
        # Defining array useful for the execution
        first_ratio = []
        second_ratio = []
        
        # Copying the section to better manage it
        new_section = section.copy()
        
        # Iterate while we still have data after all the 30 seconds steps
        while not new_section.empty:
            
            # Retrieving a new row
            row = new_section.iloc[0]
            timestamp = row['TimeStamp'] + pd.Timedelta(seconds=k)
            
            # Storing the samples
            first_ratio.append(row['First_Ratio'])
            second_ratio.append(row['Second_Ratio'])
            
            # Excluding the 30 seconds just analyzed
            new_section = new_section[new_section['TimeStamp'] >= timestamp]

        # Appending the mean values
        first_ratio = statistics.mean(first_ratio)
        second_ratio = statistics.mean(second_ratio)
        
        # Retriving the baselines
        baseline_first_ratio = baselines[i*2]
        baseline_second_ratio = baselines[(i*2)+1]
        
        # Normalized data
        first_normalized = first_ratio - baseline_first_ratio
        second_normalized = second_ratio - baseline_second_ratio
        
        print("First Ratio:      " +('%.2f' % (first_ratio,)).rstrip('0').rstrip('.')+ "\t  -  Second Ratio:      " +('%.2f' % (second_ratio,)).rstrip('0').rstrip('.'))
        print("First Baseline:   " +('%.2f' % (baseline_first_ratio,)).rstrip('0').rstrip('.')+ "\t  -  Second Baseline:   " +('%.2f' % (baseline_second_ratio,)).rstrip('0').rstrip('.'))
        print("Result:           " +('%.2f' % (first_normalized,)).rstrip('0').rstrip('.')+ "\t  -  Result:            " +('%.2f' % (second_normalized,)).rstrip('0').rstrip('.'))
        print()
        
        # Appending the result in a return object
        all_section_means.append([first_normalized, second_normalized])
        
    return all_section_means

#%%
# END: Visualizing the data for the given simple sections
def visualize_plot(sections, extra_title, labels):
    
    # Setting initial variable
    i = 0
    
    # Visualizing all the sections
    for section in sections:
        
        # Defining variables used in method
        df = section
        title = labels[i]
        
        # Saving data in ColumnDataSource
        data = {'TimeStamp': np.array(df['TimeStamp'], dtype='i8').view('datetime64[ms]').tolist(),
                'Alpha': list(df['Alpha_Avg']),
                'Beta': list(df['Beta_Avg']),
                'Theta': list(df['Theta_Avg']),
                'TimeStamp_tooltip': [x.strftime("%Y-%m-%d %H:%M:%S") for x in df['TimeStamp']]
                }
        source = ColumnDataSource(data=data)

        # Calculating the graph ranges
        smallest = 5
        largest = 105

        # Plotting the data
        p = figure(x_range=(min(data['TimeStamp']), max(data['TimeStamp'])), y_range=(smallest, largest), plot_width=1500, plot_height=600, title=extra_title+": Plot of "+title)
        p.line(x='TimeStamp', y='Alpha', source=source, color="#3DB3FE", line_width=2, legend=dict(value="Alpha"))
        p.line(x='TimeStamp', y='Beta', source=source, color="#38A967", line_width=2, legend=dict(value="Beta"))
        p.line(x='TimeStamp', y='Theta', source=source, color="#A822F3", line_width=2, legend=dict(value="Theta"))

        # Adding hover and other visualization tools
        hover = HoverTool()
        hover.tooltips=[
            ("Value", "$y"),
            ("Timestamp", "@TimeStamp_tooltip")
        ]
        p.add_tools(hover)
        p.add_layout(Title(text="TimeStamp", align="center"), "below")
        p.add_layout(Title(text="Frequency", align="center"), "left")
        p.xaxis.major_label_orientation = np.pi / 4
        p.legend.location = 'top_right'
        
        show(p)
        i = i + 1

#%%
def save_sections(participant_no, mean_stress_values, baselines, experiment_order): 
    
    # Defining header
    header = ["First_Ratio", "First_Ratio_Baseline", "Second_Ratio", "Second_Ratio_Baseline"]
    
    # Iterating through the sections    
    for i in range(0, 3):
        
        # Creating the name of the file
        file_name = 'aResults/Percentage/results_'+experiment_order[i]+'_'+participant_no+'.csv'

        # Writing on the CSV
        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerow([mean_stress_values[i][0], baselines[i*2], mean_stress_values[i][1], baselines[(i*2)+1]])
'''
