####################################                 
####    Loading ds109 dataset   ####
####        subject 2-10        ####
####################################

#### Version    : 1.0
#### Date       : 24 Feb 2017

####### Import #########
import numpy as np
import pandas as pd
import nibabel as nib

from glob import glob

def LoadDs109():

    data_r1 = sorted(glob('/home/farahana/Documents/dataset/ds109 raw/ds109/sub**/BOLD/task001_run001/bold.nii.gz'))
    data_r2 = sorted(glob('/home/farahana/Documents/dataset/ds109 raw/ds109/sub**/BOLD/task001_run002/bold.nii.gz'))

    label_r1 = sorted(glob('/home/farahana/Documents/dataset/ds109 raw/ds109/sub**/behav/task001_run001/behavdata.txt'))
    label_r2 = sorted(glob('/home/farahana/Documents/dataset/ds109 raw/ds109/sub**/behav/task001_run002/behavdata.txt'))

    ###
    ### Run 1
    ###

    data_r1_s1 = nib.load(data_r1[1]).get_data()
    data_r1_s2 = nib.load(data_r1[2]).get_data()
    data_r1_s3 = nib.load(data_r1[3]).get_data()
    data_r1_s4 = nib.load(data_r1[4]).get_data()
    data_r1_s5 = nib.load(data_r1[5]).get_data()
    data_r1_s6 = nib.load(data_r1[6]).get_data()
    data_r1_s7 = nib.load(data_r1[7]).get_data()
    data_r1_s8 = nib.load(data_r1[8]).get_data()
    data_r1_s9 = nib.load(data_r1[9]).get_data()

    dataRs_r1_s1 = np.reshape(data_r1_s1[:,:,:,:], (72*72, -1)).T
    dataRs_r1_s2 = np.reshape(data_r1_s2[:,:,:,:], (72*72, -1)).T
    dataRs_r1_s3 = np.reshape(data_r1_s3[:,:,:,:], (72*72, -1)).T
    dataRs_r1_s4 = np.reshape(data_r1_s4[:,:,:,:], (72*72, -1)).T
    dataRs_r1_s5 = np.reshape(data_r1_s5[:,:,:,:], (72*72, -1)).T
    dataRs_r1_s6 = np.reshape(data_r1_s6[:,:,:,:], (72*72, -1)).T
    dataRs_r1_s7 = np.reshape(data_r1_s7[:,:,:,:], (72*72, -1)).T
    dataRs_r1_s8 = np.reshape(data_r1_s8[:,:,:,:], (72*72, -1)).T
    dataRs_r1_s9 = np.reshape(data_r1_s9[:,:,:,:], (72*72, -1)).T

    df_r1_s1 = pd.DataFrame(dataRs_r1_s1)
    df_r1_s2 = pd.DataFrame(dataRs_r1_s2)
    df_r1_s3 = pd.DataFrame(dataRs_r1_s3)
    df_r1_s4 = pd.DataFrame(dataRs_r1_s4)
    df_r1_s5 = pd.DataFrame(dataRs_r1_s5)
    df_r1_s6 = pd.DataFrame(dataRs_r1_s6)
    df_r1_s7 = pd.DataFrame(dataRs_r1_s7)
    df_r1_s8 = pd.DataFrame(dataRs_r1_s8)
    df_r1_s9 = pd.DataFrame(dataRs_r1_s9)

    label_r1_s1 = pd.read_table(label_r1[1])
    label_r1_s2 = pd.read_table(label_r1[2])
    label_r1_s3 = pd.read_table(label_r1[3])
    label_r1_s4 = pd.read_table(label_r1[4])
    label_r1_s5 = pd.read_table(label_r1[5])
    label_r1_s6 = pd.read_table(label_r1[6])
    label_r1_s7 = pd.read_table(label_r1[7])
    label_r1_s8 = pd.read_table(label_r1[8])
    label_r1_s9 = pd.read_table(label_r1[9])

    labelCc_r1_s1 = pd.concat([label_r1_s1['CorrectAnswerCode']]*36)
    labelCc_r1_s2 = pd.concat([label_r1_s2['CorrectAnswerCode']]*36)
    labelCc_r1_s3 = pd.concat([label_r1_s3['CorrectAnswerCode']]*36)
    labelCc_r1_s4 = pd.concat([label_r1_s4['CorrectAnswerCode']]*36)
    labelCc_r1_s5 = pd.concat([label_r1_s5['CorrectAnswerCode']]*36)
    labelCc_r1_s6 = pd.concat([label_r1_s6['CorrectAnswerCode']]*36)
    labelCc_r1_s7 = pd.concat([label_r1_s7['CorrectAnswerCode']]*36)
    labelCc_r1_s8 = pd.concat([label_r1_s8['CorrectAnswerCode']]*36)
    labelCc_r1_s9 = pd.concat([label_r1_s9['CorrectAnswerCode']]*36)

    ###
    ### Run 2
    ###

    data_r2_s1 = nib.load(data_r2[1]).get_data()
    data_r2_s2 = nib.load(data_r2[2]).get_data()
    data_r2_s3 = nib.load(data_r2[3]).get_data()
    data_r2_s4 = nib.load(data_r2[4]).get_data()
    data_r2_s5 = nib.load(data_r2[5]).get_data()
    data_r2_s6 = nib.load(data_r2[6]).get_data()
    data_r2_s7 = nib.load(data_r2[7]).get_data()
    data_r2_s8 = nib.load(data_r2[8]).get_data()
    data_r2_s9 = nib.load(data_r2[9]).get_data()

    dataRs_r2_s1 = np.reshape(data_r2_s1[:,:,:,:], (72*72, -1)).T
    dataRs_r2_s2 = np.reshape(data_r2_s2[:,:,:,:], (72*72, -1)).T
    dataRs_r2_s3 = np.reshape(data_r2_s3[:,:,:,:], (72*72, -1)).T
    dataRs_r2_s4 = np.reshape(data_r2_s4[:,:,:,:], (72*72, -1)).T
    dataRs_r2_s5 = np.reshape(data_r2_s5[:,:,:,:], (72*72, -1)).T
    dataRs_r2_s6 = np.reshape(data_r2_s6[:,:,:,:], (72*72, -1)).T
    dataRs_r2_s7 = np.reshape(data_r2_s7[:,:,:,:], (72*72, -1)).T
    dataRs_r2_s8 = np.reshape(data_r2_s8[:,:,:,:], (72*72, -1)).T
    dataRs_r2_s9 = np.reshape(data_r2_s9[:,:,:,:], (72*72, -1)).T

    df_r2_s1 = pd.DataFrame(dataRs_r2_s1)
    df_r2_s2 = pd.DataFrame(dataRs_r2_s2)
    df_r2_s3 = pd.DataFrame(dataRs_r2_s3)
    df_r2_s4 = pd.DataFrame(dataRs_r2_s4)
    df_r2_s5 = pd.DataFrame(dataRs_r2_s5)
    df_r2_s6 = pd.DataFrame(dataRs_r2_s6)
    df_r2_s7 = pd.DataFrame(dataRs_r2_s7)
    df_r2_s8 = pd.DataFrame(dataRs_r2_s8)
    df_r2_s9 = pd.DataFrame(dataRs_r2_s9)

    label_r2_s1 = pd.read_table(label_r2[1])
    label_r2_s2 = pd.read_table(label_r2[2])
    label_r2_s3 = pd.read_table(label_r2[3])
    label_r2_s4 = pd.read_table(label_r2[4])
    label_r2_s5 = pd.read_table(label_r2[5])
    label_r2_s6 = pd.read_table(label_r2[6])
    label_r2_s7 = pd.read_table(label_r2[7])
    label_r2_s8 = pd.read_table(label_r2[8])
    label_r2_s9 = pd.read_table(label_r2[9])

    labelCc_r2_s1 = pd.concat([label_r2_s1['CorrectAnswerCode']]*36)
    labelCc_r2_s2 = pd.concat([label_r2_s2['CorrectAnswerCode']]*36)
    labelCc_r2_s3 = pd.concat([label_r2_s3['CorrectAnswerCode']]*36)
    labelCc_r2_s4 = pd.concat([label_r2_s4['CorrectAnswerCode']]*36)
    labelCc_r2_s5 = pd.concat([label_r2_s5['CorrectAnswerCode']]*36)
    labelCc_r2_s6 = pd.concat([label_r2_s6['CorrectAnswerCode']]*36)
    labelCc_r2_s7 = pd.concat([label_r2_s7['CorrectAnswerCode']]*36)
    labelCc_r2_s8 = pd.concat([label_r2_s8['CorrectAnswerCode']]*36)
    labelCc_r2_s9 = pd.concat([label_r2_s9['CorrectAnswerCode']]*36)

    ######## Concatenate the data and label
    data1 = pd.concat([df_r1_s1, df_r1_s2, df_r1_s3, df_r1_s4, df_r1_s5, df_r1_s6, df_r1_s7, df_r1_s8, df_r1_s9])
    data2 = pd.concat([df_r2_s1, df_r2_s2, df_r2_s3, df_r2_s4, df_r2_s5, df_r2_s6, df_r2_s7, df_r2_s8, df_r2_s9])

    label1 = pd.concat([labelCc_r1_s1, labelCc_r1_s2, labelCc_r1_s3, labelCc_r1_s4, labelCc_r1_s5, labelCc_r1_s6, labelCc_r1_s7, labelCc_r1_s8, labelCc_r1_s9])
    label2 = pd.concat([labelCc_r2_s1, labelCc_r2_s2, labelCc_r2_s3, labelCc_r2_s4, labelCc_r2_s5, labelCc_r2_s6, labelCc_r2_s7, labelCc_r2_s8, labelCc_r2_s9])
        
    data = pd.concat([data1,data2]).as_matrix()
    label = pd.concat([label1, label2]).as_matrix()

    return data, label

def LoadHaxby():
    data = sorted(glob('/home/farahana/Documents/dataset/Haxby2001/subj*/bold.nii'))

    data_s1 = nib.load(data[0])
    data_s2 = nib.load(data[1])
    data_s3 = nib.load(data[2])
    data_s4 = nib.load(data[3])
    data_s5 = nib.load(data[4])
    data_s6 = nib.load(data[5])

    ## get_data()
    data_1 = data_s1.get_data()
    data_2 = data_s2.get_data()
    data_3 = data_s3.get_data()
    data_4 = data_s4.get_data()
    data_5 = data_s5.get_data()
    data_6 = data_s6.get_data()

    #2. Label and session initialization
    label_s1 = np.recfromcsv('/home/farahana/Documents/dataset/Haxby2001/subj1/labels.txt', delimiter=' ')

    session = label_s1['chunks']
    y = label_s1['labels']

    # Initialize the 'rest' and 'house' labels
    shoe_state = y == b"shoe"
    house_state = y == b"house"

    # Divide the rest and house state for labels and data
    y_shoe = session[shoe_state]
    y_house = session[house_state]

    # Initialize shoe as [0] and house as [1] for later hot encoding
    y_shoe_one = np.zeros(y_shoe.shape[0], dtype = object)
    y_shoe_one[:] = 0

    y_house_one = np.zeros(y_house.shape[0], dtype = object)
    y_house_one[:] = 1

    # Data label appending and hot encode
    idx = 40
    data_session = np.concatenate((y_shoe_one[:], y_house_one[:]), axis=0)
    data_session = np.tile(data_session, idx)

    data_label = np.concatenate((data_session[:], data_session[:]), axis = 0)
    data_label = np.concatenate((data_label[:], data_session[:]), axis = 0)
    data_label = np.concatenate((data_label[:], data_session[:]), axis = 0)
    data_label = np.concatenate((data_label[:], data_session[:]), axis = 0)
    data_label = np.concatenate((data_label[:], data_session[:]), axis = 0)
        
    # Hot encoding
    #data_session = (np.arange(2) == data_label[:, None]).astype(np.float32)

    #3. Data Sorting 

    # Getting the x=20 of x-plane and Combining the data for both labels
    data_1_reshape = np.reshape(data_1[:,:,:,[y_shoe,y_house]],(64*64,-1)).T
    data_2_reshape = np.reshape(data_2[:,:,:,[y_shoe,y_house]],(64*64,-1)).T
    data_3_reshape = np.reshape(data_3[:,:,:,[y_shoe,y_house]],(64*64,-1)).T
    data_4_reshape = np.reshape(data_4[:,:,:,[y_shoe,y_house]],(64*64,-1)).T
    data_5_reshape = np.reshape(data_5[:,:,:,[y_shoe,y_house]],(64*64,-1)).T
    data_6_reshape = np.reshape(data_6[:,:,:,[y_shoe,y_house]],(64*64,-1)).T
    
    data = np.concatenate((data_1_reshape[:], data_2_reshape[:]), axis = 0)
    data = np.concatenate((data[:], data_3_reshape[:]), axis = 0) 
    data = np.concatenate((data[:], data_4_reshape[:]), axis = 0) 
    data = np.concatenate((data[:], data_5_reshape[:]), axis = 0) 
    data = np.concatenate((data[:], data_6_reshape[:]), axis = 0)
    

    return data, data_label
