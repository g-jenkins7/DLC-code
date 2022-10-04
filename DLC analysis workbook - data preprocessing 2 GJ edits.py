# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 14:00:59 2022

author: George
"""



#%% Dependencies
import pickle
import numpy as np
import os
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns
import pdb 
import subprocess
import re
import cv2
import scipy as sp
from scipy.signal import find_peaks
import dlc_functions as dlc_func
from scipy import interpolate


#10mg = dog
#1mg = fox
#veh = cat

#rename files in pwoershell 

# ls |Rename-Item -NewName {$_.name -replace "cat","veh"}





#%%





process_videos = 'n'


#%% Collecting behavioural data 

#set subject and session
subject_list = ['05','06','07','08','09','11','12','13','14','19','20','21','22','23','24']
session_list = ['veh','1mg','10mg']

#behavioural file informaiton 
session_file_path ='C:/Users/George/OneDrive - Nexus365/Documents/GNG - ABD/allSessions_allSubjects_ecb.p'  
all_trials_path ='C:/Users/George/OneDrive - Nexus365/Documents/GNG - ABD/all_trials_ecb.p'  
all_beh_data = pickle.load(open (session_file_path , 'rb')) 
all_trial_data = pickle.load(open (all_trials_path , 'rb')) 
all_beh_data['veh'] =all_beh_data.pop('VEH') 
all_trial_data[0].session = all_trial_data[0].session.replace({'VEH':'veh'})
mod_outcome = [all_trial_data[0].iloc[x]['outcome'] +' single' if all_trial_data[0].iloc[x]['trial type'] == 'NG Single Cue' 
               else all_trial_data[0].iloc[x]['outcome'] + ' double' if all_trial_data[0].iloc[x]['trial type'] == 'NG Double Cue'
               else all_trial_data[0].iloc[x]['outcome'] for x in all_trial_data[0].index]
all_trial_data[0].loc[:,'outcome'] = mod_outcome
#dlc file informaiton 
dlc_file_path = 'C:/Users/George/OneDrive - Nexus365/Documents/GNG - ABD/dlc analysis test dlc files'

all_data = dlc_func.get_beh_dlc_data(session_list, subject_list,all_beh_data,all_trial_data,dlc_file_path)   

#%% ffmpeg timeframe exdtraction and avg brightness data


if process_videos == 'y':
    vid_directory = 'C:/Users/George/OneDrive - Nexus365/Documents/GNG - ABD/dlc analys test videos'
    all_frame_times,    all_avg_brightness = dlc_func.process_videos(vid_directory)
    
    os.chdir(dlc_file_path)
    pickle.dump(all_frame_times, open('all_frametimes_dlc.p', 'wb'))
    pickle.dump(all_avg_brightness, open('all_avg_brightness_dlc.p', 'wb'))
else: 
    all_frame_times = pickle.load( open(dlc_file_path + '/all_frametimes_dlc.p', 'rb')) 
    all_avg_brightness = pickle.load( open(dlc_file_path + '/all_avg_brightness_dlc.p', 'rb')) 


                


#%% interpolate DLC data, find brightness peaks from the video - align to MED-file errors and convert MED events into frames


all_data,valid_points  = dlc_func.interpolate_dlc_data(all_data, 2)

period = 7 # time after trial iniation to track across
plot_peaks = 'n' # for find brightness peaks function 


all_peaks, all_onsets, mismatch_list_10 = dlc_func.find_brightness_peaks_dspk(all_frame_times,all_avg_brightness,all_data,plot_peaks,subject_list,['veh','1mg','10mg'])

all_data, all_mismatched_files = dlc_func.get_trial_frames(all_data,all_onsets,all_frame_times,period)


#uncomment to check MED-pc error alignment with brightness peak onsets detected in the video 
check_err =  True
#all_peaks = dlc_func.find_brightness_peaks_dspk(all_frame_times,all_avg_brightness,all_data,plot_peaks,subject_list,['veh'],check_err)





#%%  chopping data into trials
#avg_all_norm_medians = dlc_func.get_avg_norm_medians(all_data)

#distances = dlc_func.get_distances(all_data,avg_all_norm_medians)


restricted_traj=True #restricts points outside the box limits eg rearing for head trajectories to within confines of the box floor
all_data = dlc_func.normalise_and_track(all_data,'head',0.75,all_frame_times,all_mismatched_files,restricted_traj)

#~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.
#~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

#FIX frame chekcer

#~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.
#~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

#%% frame checker - examine whats going on in a section of the vid to check it aligns with trajecotires

check_frames  = 'no'
start_frame = 0
end_frame = 200
frame_step = 5
frame_range = [start_frame,end_frame,frame_step]

manual = True
if check_frames == 'yes':
    tag_list = ['rat06_10mg']
    vid_directory = 'C:/Users/George/OneDrive - Nexus365/Documents/GNG - ABD/dlc analys test videos'
    dlc_func.frame_checker(tag_list,vid_directory,[38910],frame_range,all_data,'frame',manual)
     
#%% 
#~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.
#~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ANALYSIS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.
#~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

trials_to_analyse =['ng1_succ','ng1_fail']# 'go1_wronglp','go2_succ', 'go2_rtex', 'go2_wronglp']
sessions_to_analyse = ['veh','1mg','10mg']
 
#all_traj_by_trial, all_nose_nan_perc_trials, all_head_nan_perc_trials = dlc_func.collect_traj_by_trialtype(trials_to_analyse,sessions_to_analyse,all_data,all_mismatched_files,scaled = False)
               
scaled_all_traj_by_trial, all_nose_nan_perc_trials, all_head_nan_perc_trials = dlc_func.collect_traj_by_trialtype(trials_to_analyse,sessions_to_analyse,all_data,all_mismatched_files,scaled = True)

avg_all_norm_medians = dlc_func.get_avg_norm_medians(all_data)


#%%
trials_to_plot = ['ng1_succ','ng1_fail']#, 'go1_wronglp','go2_succ', 'go2_rtex', 'go2_wronglp']# 'go1_wronglp','go2_succ', 'go2_rtex', 'go2_wronglp']
sessions_to_plot = ['veh','1mg','10mg']#['veh','1mg','10mg']
traj_part = 'head'

by_subject = True
subj_to_plot ='all'
dlc_func.plot_trajectories(trials_to_plot, sessions_to_plot,traj_part,scaled_all_traj_by_trial,avg_all_norm_medians,subj_to_plot,by_subject)


#%% individual trial plotting

trials_to_plot = ['go1_succ']# ['go1_succ','go1_rtex', 'go1_wronglp','go2_succ', 'go2_rtex', 'go2_wronglp']
sessions_to_plot = ['veh']#,'1mg','10mg']
num_traj = 10
plot_by_traj =False
animals_to_plot = ['06','13']#,'06','07','08','09']#,'11','12','13','14','19','20','21','22','23','24']
dlc_func.plot_indv_trajectories(trials_to_plot, sessions_to_plot,animals_to_plot,traj_part,all_traj_by_trial,avg_all_norm_medians,num_traj,all_data,plot_by_traj)



 


#%%

#%%

#%%%

#%% PDF - heat maaps
n_bins = 9

all_pdfs = {}
for tt in trials_to_plot:
    trial_type_data= all_traj_by_trial[tt]
    session_pdfs = {}
    for s in sessions_to_plot:
        session_data = trial_type_data[s]
        trial_pdf ={}
        for trial in session_data.keys():
            trial_pdf[trial],_,y_ = np.histogram2d(session_data[trial][traj_part].x,session_data[trial][traj_part].y,
                                               bins=(np.linspace(-150,150,n_bins+1),np.linspace(-0,200,n_bins+1)),density =True) 

        session_pdfs[s] = trial_pdf
    all_pdfs[tt] = session_pdfs
    #using a restricted map where area behaind the magazine is excluded


for tt in all_pdfs.keys():
    for s in all_pdfs[tt].keys():                                    
        #x =np.mean([trial_pdf[i] for i in trial_numbers])
        pdf_arr = np.array(list(all_pdfs[tt][s].values()))
        pdf_mean = pdf_arr.mean(axis=0)
        pdf_log_mean = np.log(pdf_mean)
        f, ax = plt.subplots(1,1)
        ax.imshow(pdf_log_mean, cmap='hot')#,ax=ax)
        ax.set_title(tt + ' ' + s)

#loggingg the values

#%%

trials_to_plot = ['go1_succ','go1_rtex']
sessions_to_plot = ['veh','1mg','10mg']
for tt in trials_to_plot:
    print(tt)
    trial_type_data= scaled_all_traj_by_trial[tt]
    session_pdfs = {}
    for s in sessions_to_plot:
        print(s)
        session_data = trial_type_data[s]
        indv_subjs = ['rat' + x.split('rat',1)[1][0:2] for x in session_data.keys()]
        indv_subjs_set = set(indv_subjs)
        subj_pdf ={}
        for subj in indv_subjs_set:
            trial_pdf ={}     
            subj_trials = [x for x in session_data.keys() if subj in x]
            for trial in subj_trials:
                trial_pdf[trial],_,_ = np.histogram2d(session_data[trial][traj_part].x,session_data[trial][traj_part].y,
                                                   bins=(np.linspace(-150,150,n_bins+1),np.linspace(-0,200,n_bins+1)),density=True) 
            subj_pdf[subj]= trial_pdf
        session_pdfs[s] = subj_pdf
    all_pdfs[tt] = session_pdfs

all_avg_pdf = {} 
for tt in trials_to_plot:
    session_avg_pdf ={}
    for s in sessions_to_plot:
        subj_avg_pdfs = {} 
        for subj in all_pdfs[tt][s].keys():
            subj_data = np.dstack( all_pdfs[tt][s][subj].values())  
            subj_avg_pdfs[subj] = np.mean(subj_data,axis =2)
        session_pdf  = np.dstack(subj_avg_pdfs.values())
        session_pdf_mean = np.mean(session_pdf,axis =2)
        session_avg_pdf[s] = np.where( session_pdf_mean > 0,np.log10(session_pdf_mean),-10)
    all_avg_pdf[tt] = session_avg_pdf

for tt in all_avg_pdf.keys():
    f, axs = plt.subplots(1,3)
    sess_counter = 0
    for s in all_avg_pdf[tt].keys():   
        axs[sess_counter].imshow(all_avg_pdf[tt][s], cmap='hot')#,ax=ax)
        axs[sess_counter].set_title(tt + ' ' + s)
        sess_counter +=1
        
        
# session_avg_pdf =  np.dstack(subj_avg_pdfs.values())  
# session_avg_pdf_mean = np.mean(session_avg_pdf,axis =2)
# f, ax = plt.subplots(1,1)
# ax.imshow(session_avg_pdf_mean, cmap='hot')#,ax=ax)
# #log_session_avg_pdf_mean = np.where( session_avg_pdf_mean > 0,np.log2(session_avg_pdf_mean),-2)
# log_session_avg_pdf_mean = np.log( session_avg_pdf_mean, where = session_avg_pdf_mean > 0)

# f, ax = plt.subplots(1,1)
# log_session_avg_pdf_mean[log_session_avg_pdf_mean == 0] =0.0000001 
# ax.imshow(log_session_avg_pdf_mean, cmap='hot')
# ax.set_title('log')



# hot = cm.get_cmap('hot',256)
# new_colors =hot(np.linspace(0,1,256))
# black = np.array([0,0,0,1])
# new_colors[:1,:] = black
# hotcmp  = ListedColormap(new_colors)
# _#%%

# session_avg_pdf =  np.dstack(subj_avg_pdfs.values())  
# session_avg_pdf_mean = np.mean(session_avg_pdf,axis =2)
# f, ax = plt.subplots(1,1)
# ax.imshow(session_avg_pdf_mean, cmap=hotcmp)#,ax=ax)
# log_session_avg_pdf_mean = np.where( session_avg_pdf_mean > 0,np.log(session_avg_pdf_mean),0)
# f, ax = plt.subplots(1,1)
# log_session_avg_pdf_mean[log_session_avg_pdf_mean == 0] = -0.0000001 
# ax.imshow(log_session_avg_pdf_mean, cmap=hotcmp)
# ax.set_title('log')




# for trial in session_data.keys():
#     trial_pdf[trial],_,_ = np.histogram2d(session_data[trial][traj_part].x,session_data[trial][traj_part].y,
#                                        bins=(np.linspace(-150,150,n_bins+1),np.linspace(-0,200,n_bins+1))) 
# pdf_arr = np.array(list(all_pdfs[tt][s].values()))
# pdf_mean = pdf_arr.mean(axis=0)
# pdf_log_mean = np.log(pdf_mean)
# f, ax = plt.subplots(1,1)
# ax.imshow(pdf_log_mean, cmap='hot')#,ax=ax)

#%%
trial_pdf2 ={}
for trial in session_data.keys():
    trial_pdf2[trial],_,_ = np.histogram2d(session_data[trial][traj_part].x,session_data[trial][traj_part].y,
                                       bins=(np.linspace(-150,150,n_bins+1),np.linspace(-0,200,n_bins+1))) 
pdf_arr2 = np.array(list(all_pdfs[tt][s].values()))
pdf_mean2 = pdf_arr.mean(axis=0)
pdf_log_mean2 = np.log(pdf_mean)
f, ax = plt.subplots(1,1)
ax.imshow(pdf_log_mean, cmap='hot')#,ax=ax)


#%%



x,_,_ =  np.histogram2d(session_data[trial][traj_part].x,session_data[trial][traj_part].y,
                                   bins=(np.linspace(-150,150,n_bins+1),np.linspace(-0,200,n_bins+1)),density =True) 
 
y,_,_ =  np.histogram2d(session_data[trial][traj_part].x,session_data[trial][traj_part].y,
                                   bins=(np.linspace(-150,150,n_bins+1),np.linspace(-0,200,n_bins+1))) 

f, axs = plt.subplots(1,2)
axs[0].imshow(x, cmap='hot')#,ax=ax 
axs[1].imshow(y, cmap='hot')#,ax=ax 

    
    #%% calculating trajectory length - normalise by distance between levers
    
    # CHECK need to crop traj so they end at the food magazine? maybe for correct trials - after succ trigger = go to food mag
    # for incorrect trials  - after 5 sec timeout? 
traj_distance, mean_traj_distance = dlc_func.calc_traj_length(trials_to_plot,sessions_to_plot,all_traj_by_trial,avg_all_norm_medians,traj_part)



#%%

#~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.
#NOTES ON DLC ANALYSIS PROCESS
#~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.
# get beh dlc data 
# to get trial succ times - get list to succ MED timestamps, find closest trial start time that preceeds the succ ttl and assign succ time to that trial


# processing video 
# using ffmpeg show info to get frame times
# using openCV2 to get average brightness

# interpolating data
# liklihood threshold set to 0.9 -  anything below this from DLC discarded 
# jump threshold set to 10 - differences in coords > 10*sd dev are discarded (stdev estimated from percentiles)
# isolated points, any run of less than 5 points surrounded by NaNs are discarded 


#finding error peaks
# normalise the birghtness by mean subtraction 
# remove spikes and rebounds by setting indexs where change in brihgtness is 5 < or < -5 to nan and interpolating 
# setting peak prominance to 1.53 x the average brightness from a middle portion of the session(frames 10,000 to 20,000)
# using scipy peak to find peaks  and filter them by prominences
# finding the sharpest change in d_brightness in the 250 idxs before peak to get onset of brightness and take that as error onset 

#converting med times into frames
# taking average difference in frame times and converting to seconds
# finding difference in s between frame times from med and brighness peaks from video
# discaridng files with greater than 500ms range of differences between med errors and video peaks 
# taking the average of the differences between med errors and video peaks as the difference in time between video starting and med session starting 
# adding the video start time to med times and dividing by frame times in second to get frames of beh events 
# time taken from trial start to end of tracking set to 7s (for failed trials)

#chopping data into trials 
# get medians of the box features - subtracting individual files poke median from the tracking data to normalise
# restricting trajectorys to within the floor of the box (in y domain) by setting anything greater than the y-level of the food mag to zeor 
# scaling trajectories in the x domain by relative lever distances - at the moment leaving y domain untouched
# chopping failed trials between trial start and trial start +7s 
# chopping successful trials between trials start and 0.75s after food mag roi entry following successful trial completion for succesful trials

# collecting trajectory by trial type 
# flipping trials if animals double side is 2 so that go small is always on the left and go large always on the right
