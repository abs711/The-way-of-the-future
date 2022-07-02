
import pandas as pd
import numpy as np
import scipy.io as sio 
import os
global root_datadir
global root_dir
import torch
from pathlib import Path
from math import ceil
results_dir =[]
timestamp =[]
model_type=[]
debugPrint=False
global data_config_global
global uid_manager
global target_sensor_type
"""
v8 Data manager with IDs, target_len, label long reading, class weights
created 08/2019 
- 08/28/2019
"""

class UID_manager:
    def __init__(self):
        self.uids2index={}
        self.index2uid={}
        self.indexCount=0
    def addUID(self,this_uid) :
        if this_uid not in self.uids2index:
            self.uids2index[this_uid]=self.indexCount
            self.index2uid[self.indexCount] = this_uid
            self.indexCount +=1
            return self.indexCount -1 
        else :
            print(self.index2uid)
            return self.getIndex(this_uid)
    
    def getIndex(self,uid)   :
        return int(self.uids2index[uid])  
    
    def getUID(self,index):
        return self.index2uid[int(index.item())]

def index2UID(idx):
    global uid_manager
    idx=int(idx.item())
    print(idx)
    return uid_manager.index2uid[idx]
# reformat all the data, take three inputs, data_dir can be used elsewhere
def get_and_reformat_all_data(curr_root_dir, config_file_name, param_vals):
    # global results_dir
    
    global root_dir
    global root_datadir    
    global data_config_global
    
    global uid_manager
    
    uid_manager = UID_manager()
    
    root_dir= curr_root_dir;
    
    root_datadir=curr_root_dir + 'processed/'
    
    dataset_dict={}; 
    
    config_df=load_data_config(config_file_name)
    
    
    data_config_global =config_df
    save_param_vals_in_config(param_vals)

    dataset_dict["config"]=config_df

        #todo sample
    train_dict=load_data(config_df['train_datadir'],param_vals,bTrain=True)
    test_dict=load_data(config_df['test_datadir'],param_vals,bTrain=False)
    val_dict= load_data(config_df['val_datadir'],param_vals,bTrain=False)



    
    
    return train_dict, test_dict, val_dict, uid_manager


##  get tensors for each trial

def get_tensors_from_df(trial_df):
    
    global data_config_global        
    trial_df= trial_df.loc[:,~trial_df.columns.duplicated(keep='first')]
    #first get numpy 
    x,y,userStats,trialIDsInt,trialIDsList,frameNumsInt,relTime= get_numpy_from_df(trial_df,data_config_global)
        
    num_input_features=int(get_num_input_features(data_config_global))
    num_stats_features=int(get_num_stats_features(data_config_global))
    num_target_features=int(get_num_target_features(data_config_global))
    
    
    # convert to tensors 
    xTensor=torch.from_numpy(x).type(torch.Tensor).view(-1,num_input_features)
    yTensor = torch.from_numpy(y).type(torch.Tensor).view(-1,num_target_features)
    
    userStatsTensor=torch.from_numpy(userStats).type(torch.Tensor).view(-1,num_stats_features)
    trialIDsTensorInt=torch.from_numpy(trialIDsInt).type(torch.Tensor).view(-1)
    frameNumsIntTensor=torch.from_numpy(frameNumsInt).type(torch.IntTensor).view(-1)
    relTimeTensor=torch.from_numpy(relTime).type(torch.IntTensor).view(-1)
    
    data_dict={};
    
    data_dict['x']=xTensor
    data_dict['y']=yTensor
    data_dict['userStats']=userStatsTensor
    data_dict['trialIDsInt']=trialIDsTensorInt
    #data_dict['trialIDsList']=trialIDsList
    data_dict['frameNums']=   frameNumsIntTensor 
    data_dict['relTime']=relTimeTensor
    
    return data_dict
    
def get_numpy_from_df(fullData_df,config_df):
    global target_sensor_type
    
    input_features=config_df['input_features'].dropna().tolist()   
    stats_features=config_df['stats_features'].dropna().tolist() 
    
    target_features=config_df['target_features'].dropna().tolist() 
    
    x=fullData_df[input_features].dropna().to_numpy().astype("float32")
    print('target_sensor_type: ',target_sensor_type[0])
    
    if target_sensor_type[0] == 'action_labels':
        y=fullData_df[target_features].dropna().to_numpy().astype("long")
        print("Loading Target as Long")
    else:
        y=fullData_df[target_features].dropna().to_numpy().astype("float32")
        print("Loading Target as Float32")
    userStats=fullData_df[stats_features].dropna().to_numpy().astype("float32")
    trialIDsList=fullData_df.uid.dropna().tolist();
    
    intTrialID=uid_manager.addUID(trialIDsList[0])
    intTrialIDs=np.repeat(intTrialID,x.shape[0])
    
    relTime=fullData_df.relTime.dropna().to_numpy().astype("int32")
    
    # extract only frame numbers for samples and only if sample factor is not zero
    if int(get_field_from_config('frame_subsample_factor'))==0:
        frameNums=np.zeros((fullData_df.shape[0],),dtype=int)
    else: 
        frameNums=fullData_df.frames.dropna().str.replace('[-.a-zA-Z]','').to_numpy().astype("int32")
    
    
    return     x,y,userStats,intTrialIDs,trialIDsList,frameNums,relTime








"""
input: list of input trial directories, bTrain = true for training set else false
access: globale config dataframe to extract parameters for subsampling, trainRatio etc
output: dataframe with user stat data 
    
"""
def load_data(input_dir_df,param_vals,bTrain):
        
        allTrials_dict={}  
        global data_config_global  
                
        def appendToDict(this_dict):
            for key,val in this_dict.items():
                if key in allTrials_dict and torch.is_tensor(val):
                    allTrials_dict[key]=torch.cat([allTrials_dict[key],val],dim=0)
                else :
                    allTrials_dict[key]=val
                    

        if data_config_global.bNorm.dropna().values[0]:
            filename="jointDataNorm.csv"
        else :
            filename="jointDataRaw_ss6wLabels.csv" #"jointDataRaw.csv"
            
        input_dir_list=input_dir_df.dropna().tolist()
        
      
        trainDataRatio=float(data_config_global.trainDataRatio.dropna().values)
        sample_factor=int(get_field_from_config('sub_sample_factor'))

        rolling_window_step= get_field_from_config('window_step')

            
        csvData_df=pd.DataFrame();
        if len(input_dir_list) > 1:
                       
            for sub_dirs in input_dir_list :
                # read  csv and user stat data
                print ("loading from ",sub_dirs)
                this_csv_df=pd.read_csv(os.path.join(sub_dirs,filename) )  
                
                if trainDataRatio !=1.0 and  bTrain:
                    # only keep a certain part of data
                    num_samples=len(this_csv_df.index);
                    num_samples_keep=  ceil(trainDataRatio * num_samples)
                    if debugPrint: print ("total samples length",num_samples," but keeping ",num_samples_keep)
                    this_csv_df=this_csv_df[:num_samples_keep]
                
                if   sample_factor !=1 and bTrain:  
                    this_csv_df=sample_data(this_csv_df,sample_factor)
                    
                # data_dir/user_main_dir/activity/trialNo/Norm 
                # so go up 3 levels for user main directory
                levels_up = 3 
                user_main_dir=Path(sub_dirs).parents[levels_up-1]
                this_stats_df= pd.read_csv(os.path.join(user_main_dir,"userStats.csv") )     
            
                # repeat user stat for every datapoint in csv data
                num_rows=len(this_csv_df.index)
                this_stats_df=pd.concat([this_stats_df]*num_rows, ignore_index = True,sort=True)
                
                # merge the dataframes into one, along col axis
                this_csv_userStats_df=pd.concat([this_csv_df,this_stats_df],axis=1, ignore_index = False,sort=True)
                
                if debugPrint: print ("csv_user shape",sub_dirs,this_csv_userStats_df.shape)
                
                #
                this_trial_dict=get_tensors_from_df(this_csv_userStats_df)
                
                
                this_trial_dict=rolling_window_with_prediction(this_trial_dict,rolling_window_step)
                appendToDict(this_trial_dict)
                # merge all trials into one, along row axis

                if debugPrint: print ("all trials dict x,y,frames,user stats shape",allTrials_dict['x'].shape,allTrials_dict['y'].shape,
                       allTrials_dict['frameNums'].shape, allTrials_dict['ids'].shape )
                #csvData_df=csvData_df.append(this_csv_userStats_df)
        else :
            
            print ("single file loading from ",input_dir_list[0])

            this_csv_df = pd.read_csv(os.path.join(input_dir_list[0],filename)) 
             
            if   sample_factor !=1 and bTrain:
                this_csv_df=sample_data(this_csv_df,sample_factor)
            

            # data_dir/user_main_dir/activity/trialNo/Norm 
            # so go up 3 levels for user main directory
            
                
            levels_up = 3 
            user_main_dir=Path(input_dir_list[0]).parents[levels_up-1]
            this_stats_df = pd.read_csv(os.path.join(user_main_dir,"userStats.csv")) 
            
            # repeat user stat for every datapoint in csv data
 
            num_rows=len(this_csv_df.index)
            this_stats_df=pd.concat([this_stats_df]*num_rows, ignore_index = True,sort=True)
            this_csv_userStats_df=pd.concat([this_csv_df,this_stats_df],axis=1, ignore_index = False,sort=True)

            this_trial_dict=get_tensors_from_df(this_csv_userStats_df)
            this_trial_dict=rolling_window_with_prediction(this_trial_dict,rolling_window_step)
            appendToDict(this_trial_dict)
       
        return allTrials_dict

   
# load the config_file according to the path
def load_data_config(config_file_name):

        def concatenate_all_values_with(root_datadir,values_list,bNorm=True):    
            if bNorm:
                return  [root_datadir + value + 'norm/'for value in values_list]
            else :
                return  [root_datadir + value + 'raw/'for value in values_list]
              # return  [root_datadir + value for value in values_list]
        
        global  root_dir 
        global root_data_dir 
        global target_sensor_type
        config_file = root_dir + config_file_name
        input_features = []
        target_features = []

        df1 = pd.read_csv(config_file)
        # set Index as the first column
        df2 = df1.set_index("Index", drop=True)
        
        #global data_config_global
        #data_config_global=df2.copy(deep=True)
        #Normalize or not
        b_NormalizedData= (df2.loc['b_NormalizedData'].dropna() == 'TRUE').bool()

        user_stats_type=df2.loc["user_stats_type"].dropna().tolist()
        
        # vision details 
        # frame sub sample factor w.r.t to kinematics samples; 
        # frame sub sample =2 will drop every other frame corresponding to each kinematic sample
        # frame sub_sample = 0 will not extract any frames at all
#        frame_subsample_factor=df2.loc["frame_subsample_factor", :]
#        frame_subsample_factor = np.array(frame_subsample_factor)[0] 

        trainDataRatio=float(df2.loc['trainDataRatio'].dropna().values[0])
        trainDataRatio=np.array(trainDataRatio)
        
        
        # extract jointname, axis, and sensorname accordingly
        postfix = df2.loc["input_axis", :]
        postfix = np.array(postfix)
        postfix = [axis for axis in postfix if str(axis) != 'nan']


        input_sensor_type = df2.loc["input_sensor_type", :]
        input_sensor_type = np.array(input_sensor_type)
        input_sensor_type = [sensor_type for sensor_type in input_sensor_type if str(sensor_type) != 'nan']

        input_joints = df2.loc["input_joints", :]
        input_joints = np.array(input_joints)
        input_joints = [joints for joints in input_joints if str(joints) != 'nan']

        target_sensor_type = df2.loc["target_sensor_type", :]
        target_sensor_type = np.array(target_sensor_type)
        target_sensor_type = [sensor_type for sensor_type in target_sensor_type if str(sensor_type) != 'nan']

        target_joint = df2.loc["target_joint", :]
        target_joint = np.array(target_joint)
        target_joint = [joint for joint in target_joint if str(joint) != 'nan']

        target_axis = df2.loc["target_axis", :]
        target_axis = np.array(target_axis)
        target_axis = [axis for axis in target_axis if str(axis) != 'nan']

#        sub_sample_factor = df2.loc["sub_sample_factor", :]
#        sub_sample_factor = np.array(sub_sample_factor)[0]

        # concatanate all the input_names with  features and save it in the a list
        for sensor_type in target_sensor_type:
            if sensor_type == "action_labels":
                print("CLASSIFICATION TARGET LABELS")
                for joint in target_joint:
                    joint_name = joint + "_" + 'labels'  
                    target_features.append(joint_name)
            else:
                for joint in target_joint:
                    joint_name = sensor_type + "_" + joint
                    for axis in target_axis:
                        fullname_y = joint_name + "_" + axis
                        target_features.append(fullname_y)

        # concatnate all the label_names with  features and save it in the a list
        for sensor_type in input_sensor_type:
            for joints in input_joints:
                joint_name = sensor_type + "_" + joints
                for axis in postfix:
                    fullname_x = joint_name + "_" + axis
                    input_features.append(fullname_x)
        
          
        train_datadir_list= concatenate_all_values_with(root_datadir, df2.loc['train_dir'].dropna().tolist(),b_NormalizedData )    
        test_datadir_list=concatenate_all_values_with(root_datadir, df2.loc['test_dir'].dropna().tolist(),b_NormalizedData) 
        val_datadir_list=concatenate_all_values_with(root_datadir, df2.loc['val_dir'].dropna().tolist(),b_NormalizedData) 
    
    
        
        # create a clean config dataframe     
        config_df=pd.DataFrame()
        if len(input_features)>len(train_datadir_list):
            
            config_df['input_features'] =pd.Series(input_features)
            config_df['train_datadir']=pd.Series(train_datadir_list)
        
        else :
            if debugPrint: print ("Number of training dirs",len(train_datadir_list))
            config_df['train_datadir']=pd.Series(train_datadir_list)
            config_df['input_features'] =pd.Series(input_features)
            

        config_df['stats_features'] =pd.Series(user_stats_type)
#        config_df['frame_subsample_factor']=pd.Series(frame_subsample_factor)
        config_df['trainDataRatio']=pd.Series(trainDataRatio)
        config_df['bNorm']=b_NormalizedData
        
        config_df['target_features'] =pd.Series(target_features)
#        config_df['sub_sample_factor']=pd.Series(sub_sample_factor)
        config_df['test_datadir']=pd.Series(test_datadir_list)
        config_df['val_datadir']=pd.Series(val_datadir_list)
        
        return config_df
   
   

 
"""
helper internal functions
returns number of input features
"""    
def get_num_input_features(config_df):

    return config_df['input_features'].dropna().shape[0]    

def get_num_target_features(config_df):

    return config_df['target_features'].dropna().shape[0]  

def get_num_stats_features(config_df):

    return config_df['stats_features'].dropna().shape[0]   
 
def get_data_from_mat(datadir,inputMatFile):
     
    matfile=datadir + 'matInputs/'+inputMatFile
    
    print (matfile)
      #matfile='/media/raiv/Data_linux/GDrive_linux/DeepLearning/data/xSens_Phase2/results/norm/cLSTM/02_08_17_12/inputs/inputsAndTarget.mat'
    this_trial_ip_mat_dict=sio.loadmat(matfile)
#    trainData_dict, testData_dict, valData_dict=dm.get_and_reformat_all_data(datadir_gen,config_file)  
# 
    # search for jAngles idx using sensor type       
    trainX_Stream =   this_trial_ip_mat_dict['trainX_Stream'].astype(np.float32)
    trainY_Stream =   np.reshape(this_trial_ip_mat_dict['trainY_Stream'],(-1,)).astype(np.float32)
    
    testX_Stream =    this_trial_ip_mat_dict['testX_Stream'].astype(np.float32)
    testY_Stream =   np.reshape(this_trial_ip_mat_dict['testY_Stream'],(-1,)).astype(np.float32)
    
    valX_Stream =    this_trial_ip_mat_dict['valX_Stream'].astype(np.float32)
    valY_Stream =    np.reshape(this_trial_ip_mat_dict['valY_Stream'],(-1,)).astype(np.float32)

     
    return trainX_Stream,trainY_Stream,testX_Stream,testY_Stream,valX_Stream,valY_Stream
     
#data Saving  functions and external
##############################################################################
''' 
'''

def get_field_from_config(field):
    # if mostly float tpes, cast it?
    global data_config_global
    
    return data_config_global[field].dropna().values[0]

def get_data_config_df():
    
    global data_config_global
    
    return  data_config_global

def get_data_dir():
    
    global root_datadir
    return root_datadir

def getUniqueFileName(scope,var_name):
     
    global timestamp
    global model_type
    global results_dir
       
#    print ("type resultsdir",type(results_dir[0])) 
#    print ("type timestamp",type(timestamp)) 
#    print ("type model_type",type(model_type)) 
    
 
     
    timstamped_filename = results_dir[0] + scope + "/" 
    
    if debugPrint: print ("timstamped_filename",timstamped_filename)
    if not os.path.exists(timstamped_filename):
         os.makedirs(timstamped_filename)
    
    
    timstamped_filename=timstamped_filename+ var_name  
    
    
    
    return str(timstamped_filename);

def make_all_model_dirs(folder_name) :    

    if not os.path.exists(folder_name):
         os.makedirs(folder_name)
         
def get_model_results_dir():   
     return results_dir[0] + 'model/'
     
def get_results_dir():     
    return results_dir[0]

def get_checkpoints_dir():     
    return results_dir[0] +'checkpoints/'

'''
auto method creates sub folder within previously defined 
results folder with preset model type and time stamp etc
'''
def set_results_dir_auto(folder_annotation,date_first=False):
     
     global timestamp
     global model_type
     if date_first:
         results_dir_auto = root_dir +'results/' + model_type +'/'+ timestamp+'_'+ folder_annotation +'/'
     else:
         results_dir_auto = root_dir +'results/' + model_type +'/'+ folder_annotation+'_'+ timestamp +'/'

        
     if debugPrint: print("Results auto set to",results_dir_auto)
     set_results_dir_manual(results_dir_auto)
     
     
     
'''
manual method overrides all prior calls and replaces results dir
completely
'''     
def set_results_dir_manual(results_dir_str):
     global results_dir
     print (results_dir_str)
     results_dir.insert(0,results_dir_str)
     make_folder_in_results(results_dir[0])
     if debugPrint: print("Results folder created in",results_dir[0])

     
def make_folder_in_results(folder_name) :    

    if not os.path.exists(folder_name):
         os.makedirs(folder_name)
         
def save_tensors_as_mat(matfilename_str_list,vars_dict):  

    np_dict=dict.fromkeys(vars_dict)
    not_a_tensor=[];
    for key in np_dict.keys():
        try:
            np_dict[key]=vars_dict[key].detach().numpy()
        except:
            not_a_tensor.append(key)
            
            continue
    
    for not_tensor_key in not_a_tensor:
        print ("deleting",not_tensor_key)
        del np_dict[not_tensor_key]
        
    save_as_mat(matfilename_str_list,np_dict)
    
    
    return 

def save_as_mat(matfilename_str_list,vars_dict):
    ''' Desc: saves variables as matfile
       Arguments: filename_str [scope (i.e foldername),variable name] 
                 vars_dict:dictionary of variables
       returns:   nada 
       
   '''
    
    scope    =   matfilename_str_list[0];
    var_name =  matfilename_str_list[1];
    

    print('Saving %s in %s folder: '%(var_name,scope))

    matlabFileName = getUniqueFileName(scope,var_name)
     
    sio.savemat(matlabFileName,vars_dict)   
    return

def save_input_tensors_mat(matfilename_str_list,vars_dict):
    
    #create a numpy dict
    np_dict= {}
    
    np_dict['x']=vars_dict['x'].numpy()
    np_dict['y']=vars_dict['y'].numpy()
    np_dict['userStats']=vars_dict['userStats'].numpy()
    
    save_as_mat(matfilename_str_list,np_dict)
    
    return


def set_global_flags(m_type,tstamp,s_type):
        #''' Desc: sets the global flags needed to save various results once and for all from the main file
        #     Arguments: model_type,timestamp
        #     returns:    
        #     @todo :      maybe make it dictionary     
        #'''

     
     global timestamp
     global model_type
     global scheduler_type   
     
     timestamp=tstamp
     model_type = m_type
     scheduler_type = s_type
     return 

def save_settings(param_vals)     :
     #data_config_df=pd.read_csv(config_file,index_col=0,header=0,engine='python')

     #p_df=pd.DataFrame.from_dict(param_vals)
     global data_config_global
     
#     p_df=pd.DataFrame.from_dict(dict([ (k,pd.Series(v)) for k,v in param_vals.items() ]),orient='index')
#     data_config_global=data_config_global.append(p_df)
     
     timstamped_path=results_dir[0] 
     if not os.path.exists(timstamped_path):
         os.makedirs(timstamped_path)
         
     data_config_global.to_csv(path_or_buf=timstamped_path +'run_settings.csv ')
     
     return 

def save_param_vals_in_config(param_vals):
    
    global data_config_global
     
    p_df=pd.DataFrame.from_dict(dict([ (k,pd.Series(v)) for k,v in param_vals.items() ]),orient='columns')
    data_config_global=data_config_global.append(p_df)
     
     

"""" data procesing functions
"""
#sample data frame
#keeps index same as original data frame, RangeIndex(step-1) reset

def sample_data(data, sub_sample_factor):

        stride = int(sub_sample_factor)
        index = range(0, data.shape[0], stride)
        sampled_data_df= data.iloc[index, :]
        sampled_data_df.reset_index(drop=True, inplace=True) # reset index to step 1
        
        

        return sampled_data_df
    
def unique(list1): 
      
        # insert the list to the set 
        list_set = set(list1) 
        # convert the set to the list 
        unique_list = (list(list_set)) 
        print ("Unique ID in this set")
        for x in unique_list: 
            print (x)
        return   unique_list
    
def tranform_IDList_toInt(trialIDs):
        
         uniq_list= unique(trialIDs)
         intTrialIDs=trialIDs.copy()
         for uniqueID_idx, uniqueID in enumerate(uniq_list):
             for i, x in enumerate(trialIDs): 
                 if x == uniqueID:
                     intTrialIDs[i]  = uniqueID_idx 
         return np.array(intTrialIDs)


def find_matched_ID_seq(trialIDsInt_win)  :  
    
    
    id_seq_list=trialIDsInt_win.tolist()
    def is_seq_valid(lst):
            return lst[1:] == lst[:-1] 
       
    matched_seq_indices=[];    
    for idx,id_seq in enumerate(id_seq_list):
        
        if (is_seq_valid(id_seq)):
            matched_seq_indices.append(idx)
    
    matched_seq_indices=torch.LongTensor(matched_seq_indices)
                   
    return matched_seq_indices

def pytorch_rolling_window_IDbased(data_dict, param_vals,step_size=1):
    
    
    input_window= param_vals['seq_length'][0]
    target_window=param_vals['target_len'][0]
    pred_window= int(target_window/2)


    x = data_dict['x']
    y = data_dict['y']
    userStats=data_dict['userStats'] 
    trialIDsInt = data_dict['trialIDsInt']
    l = data_dict['trialIDsList']
    frameNums=data_dict['frameNums']
    
    num_target_features=y.shape[1]
    
    # remove last indices from inputs; ones with no target preds
    x=x[:-pred_window,:]; userStats=userStats[:-pred_window,:];
    trialIDsInt=trialIDsInt[:-pred_window];  frameNums=frameNums[:-pred_window]
    
    # remove first indices from target, ones with no correspondin inputs
    y=y[pred_window:,:]

    x_win=x.unfold(0,input_window,step_size)
    y_win=y.unfold(0,input_window,step_size)  
    userStats_win=userStats.unfold(0,input_window,step_size)
    trialIDsInt_win=trialIDsInt.unfold(0,input_window,step_size)
    frameNums_win=frameNums.unfold(0,input_window,step_size)
    
    matched_seq_indices=find_matched_ID_seq(trialIDsInt_win)
    # find all samples with differing IDs
    
    x_winID=torch.index_select(x_win,0,matched_seq_indices)
    y_winID=torch.index_select(y_win,0,matched_seq_indices)
    userStats_winID=torch.index_select(userStats_win,0,matched_seq_indices)

    frameNums_winID=torch.index_select(frameNums_win,0,matched_seq_indices)
    ID_winID=torch.index_select(trialIDsInt_win,0,matched_seq_indices)
    
    
    trialIDsListID = [ l[idx] for idx in matched_seq_indices.cpu().numpy().tolist() ]
    
    
    
    
    
    userStats_winID=userStats_winID[:,:,-1]
    
    y_winID=y_winID[:,:,-target_window:input_window]   # y_winID=y_winID[:,:,-1]
    ID_winID=ID_winID[:,-1]
    
    
    # permute to get batchsize, seq_len, features
    x_winID=x_winID.permute(0,2,1);
    y_winID=y_winID.permute(0,2,1);
    #if num_target_features==1: y_winID.view(-1)
    
    del x,x_win,y, y_win,trialIDsInt,trialIDsInt_win,userStats_win,frameNums_win

    return {'x':x_winID,'y':y_winID,'userStats':userStats_winID, 'ids':ID_winID, 'idList':trialIDsListID, 'frameNums':frameNums_winID}

def rolling_window(data_dict,step_size):
    
    """ input and output sequence end at same time point t
    input will be t-i to t ; outout will be t-j to t
    
    
    
    """
    global data_config_global
    
    
    input_window= int(data_config_global['seq_length'].dropna().values[0])
    target_window=int(data_config_global['target_len'].dropna().values[0])

    step_size=int(step_size)

    x = data_dict['x']
    y = data_dict['y']
    userStats=data_dict['userStats'] 
    ids = data_dict['trialIDsInt']
#    trialIDsList = data_dict['trialIDsList']
    frameNums=data_dict['frameNums']
    relTime=data_dict['relTime']
    
    num_target_features=y.shape[1]
    
    # remove last indices from inputs; ones with no target preds
    
    x_unfold=x.unfold(0,input_window,step_size)
    y_unfold=y.unfold(0,input_window,step_size)  
    userStats_unfold=userStats.unfold(0,input_window,step_size)
    ids=ids.unfold(0,input_window,step_size)
    frameNums_unfold=frameNums.unfold(0,input_window,step_size)
    relTime_unfold=relTime.unfold(0,input_window,step_size)
 
    # remove columns not needed
    userStats_unfold=userStats_unfold[:,:,-1]
    y_unfold=y_unfold[:,:,-target_window:input_window]   # y_winID=y_winID[:,:,-1]
    ids=ids[:,-1] 
    
    
    # permute to get batchsize, seq_len, features
    x_unfold=x_unfold.permute(0,2,1);
    y_unfold=y_unfold.permute(0,2,1);
    #if num_target_features==1: y_winID.view(-1)
    
    del x,y,userStats,frameNums

    return {'x':x_unfold,'y':y_unfold,'userStats':userStats_unfold, 'ids':ids, 'frameNums':frameNums_unfold,'relTime':relTime_unfold}

def rolling_window_with_prediction(data_dict,step_size):
    
    """ rolls input and output windows"
     
    """
    global data_config_global
    
    
    input_window= int(data_config_global['seq_length'].dropna().values[0])
    target_window=int(data_config_global['target_len'].dropna().values[0])
    pred_window= int(data_config_global['pred_horizon'].dropna().values[0])
    step_size=int(step_size)

    x = data_dict['x']
    y = data_dict['y']
    userStats=data_dict['userStats'] 
    ids = data_dict['trialIDsInt']
#    trialIDsList = data_dict['trialIDsList']
    frameNums=data_dict['frameNums']
    relTimeX=data_dict['relTime']
    relTimeY=data_dict['relTime']
    num_target_features=y.shape[1]
    
    num_target_features=y.shape[1]
    output_window=  input_window + pred_window +  target_window -1

    # remove last indices from inputs; ones with no target preds
    cut_bottom=target_window+pred_window-1
    if cut_bottom > 0:
        x=x[:-cut_bottom,:]; userStats=userStats[:-cut_bottom,:];
        ids=ids[:-cut_bottom];  frameNums=frameNums[:-cut_bottom]
        relTimeX=relTimeX[:-cut_bottom]
    
    # remove first indices from target, ones with no correspondin inputs
    
    #y=y[pred_window:,:];
    #relTimeY=relTimeY[pred_window:]
    
    x_unfold=x.unfold(0,input_window,step_size)
    userStats_unfold=userStats.unfold(0,input_window,step_size)
    ids=ids.unfold(0,input_window,step_size)
    frameNums_unfold=frameNums.unfold(0,input_window,step_size)
    relTimeX=relTimeX.unfold(0,input_window,step_size)
    
    y_unfold=y.unfold(0,output_window,step_size)  
    relTimeY=relTimeY.unfold(0,output_window,step_size)


      # remove columns not needed
    userStats_unfold=userStats_unfold[:,:,-1]
    ids=ids[:,-1] 
    
    # first seq len indices -1 are history
    y_hist=y_unfold[:,:,0:input_window-1]
    relTimeY_hist=relTimeY[:,0:input_window-1]  # y_winID=y_winID[:,:,-1]

    #last target len indices are the actual targets pred horizon apart
    y_target=y_unfold[:,:,-target_window:output_window]   # y_winID=y_winID[:,:,-1]
    relTimeY_target=relTimeY[:,-target_window:output_window]   # y_winID=y_winID[:,:,-1]


    y_cat=torch.cat((y_hist,y_target),dim=2)
    relTimeY_cat=torch.cat((relTimeY_hist,relTimeY_target),dim=1)
    # permute to get batchsize, seq_len, features
    x_unfold=x_unfold.permute(0,2,1);
    y_cat=y_cat.permute(0,2,1);

    return {'x':x_unfold,'y':y_cat,'userStats':userStats_unfold, 'ids':ids, 'frameNums':frameNums_unfold, 'relTimeX':relTimeX, 'relTimeY':relTimeY_cat}

"""Collects input tensor by averaging predictions 
"""
def collect_window(input_rolled):
    
    def sum_kernel(kernel_window):
        kernel_flipped= kernel_window.flip(1)
    
#        print (torch.mean(kernel_flipped.diag()))
        
        return torch.mean(kernel_flipped.diag())
#        for col_idx in range(window_len):
#            print (kernel_flipped.diag(col_idx))
    input_rolled=input_rolled.detach()
    batch_size=input_rolled.shape[0]
    window_len=input_rolled.shape[1]
    averaged_seq=[]
    for idx_batch in range(batch_size):
      averaged_seq.append(sum_kernel(input_rolled[idx_batch:idx_batch+window_len,:]))
    
    return torch.stack(averaged_seq)


def get_weights4classes():
    global root_datadir
    weights_dir = root_datadir + '/classweights'
    wdf = pd.read_csv(weights_dir + '/ClassWeights_ss6.csv')
    wdf = wdf.set_index("joint_ids",drop=True)
    joint_labels = wdf.index.tolist()
    weight_list = []
    for joint in joint_labels:
        weight_list.append(wdf.loc[joint].dropna().values.tolist())

    category_sizes = torch.tensor(weight_list)
    category_sizes = category_sizes.float()
    weight = category_sizes.sum(1, keepdim=True) / category_sizes
    return weight
	  
