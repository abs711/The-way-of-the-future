import torch
import sys
sys.path.append("../../utilsX")
 
import datamanager_v7 as dm
 
import os
import copy
import numpy as np
from itertools import product
from collections import defaultdict
from collections import OrderedDict
import time
import torch.utils.data as utils 
from torch.autograd import Variable
import sys
from pytorchtools import EarlyStopping

from torch_utils import *
from log_utils import *
from train_eval_functions import evaluate
from loss_modules import * #RMSELoss
from metrics import *
from random import randrange
debugPrint=0


import device_utils
device = device_utils.getCudaDevice() 
#sanity_check=True
#single_batch= False
#clipGrads=False
#indie_ckpt = False
#substring_list = ['Conv','Vision','Scene','Optical','Spatiotemporal']
def SetTrainingOpts(sanityflag,use_single_batch,overlap_datatransfers,model_name,backprop_vision=False,clipGradients=False,indie_checkpoints=False,vision_substrings =['Conv','Vision','Scene','Optical','Spatiotemporal'],numscenesamples=1):
    global sanity_check,single_batch,non_blocking,unnecessary_model_name,clipGrads,indie_ckpt,substring_list,num_scenesamples,backprop_vis
    sanity_check,single_batch,non_blocking,unnecessary_model_name,backprop_vis,clipGrads,indie_ckpt,substring_list,num_scenesamples= sanityflag,use_single_batch,overlap_datatransfers,model_name,backprop_vision,clipGradients,indie_checkpoints,vision_substrings,numscenesamples
    return 

def GetTrainingOpts():
    global sanity_check,substring_list,num_scenesamples,unnecessary_model_name,backprop_vis
    return sanity_check,substring_list,num_scenesamples,unnecessary_model_name,backprop_vis


#sanityflag,single_batch,clipGrads,indie_ckpt,substring_list = SetTrainingOpts()
##validation_patience = 1
# we will use CUDA if it is available
USE_CUDA = torch.cuda.is_available()
#os.environ['CUDA_VISIBLE_DEVICES'] = device_utils.getCudaDeviceidx()




def my_collate(batch):
    #inputSeq,userStat,frames_tensor,target
    
    (batch_inputSeq,batch_userStat,batch_frames_tensor,batch_target)=zip(*batch)
   
    batch_inputSeq=torch.stack(batch_inputSeq)
    batch_userStat=torch.stack(batch_userStat)
    batch_frames_tensor=torch.stack(batch_frames_tensor)
    batch_target=torch.stack(batch_target)
    
    return batch_inputSeq,batch_userStat,batch_frames_tensor,batch_target
        

def decayed_LearningRate(model_params,epoch):
    base_lr = model_params['learning_rate']
    decay_rate = model_params['lr_decay_rate']
    step = model_params['lr_decay_step']
    assert 1 <= epoch
    if 1 <= epoch <= step:
        return base_lr
    elif step <= epoch <= step * 2:
        return base_lr * decay_rate
    elif step * 2 <= epoch <= step * 3:
        return base_lr * decay_rate * decay_rate
    else:
        return base_lr * decay_rate * decay_rate * decay_rate    
        
    

def get_batches4models(batch,model_type):

       if any(substring in model_type for substring in substring_list):
               batchX,batchUserStat,batchFrames,batchYSeq=batch
               return batchX,batchUserStat,batchFrames,batchYSeq
       else:
               batchX,batchUserStat,batchYSeq=batch
               return batchX,batchUserStat,None,batchYSeq


def load_frames4models(batchFrames,model_type):
       
          if 'Scene' in model_type:
                  #frame_id = 4#randrange(seq_len)
                          #print('Frame selected: ', frame_id)
                  #batchFrames=batchFrames[:,frame_id,:,:,:]
                          
                  batchFrames=batchFrames.contiguous().to(device,non_blocking=non_blocking)
##                  print(batchFrames.size())
          else:
                  batchFrames=batchFrames.contiguous().to(device,non_blocking=non_blocking)
##                  print(batchFrames.size())
          return batchFrames


def forwardpass4models(batchX,batchFrames,batchY_history,batchY, batchUserStat,phase,model,model_type):

       if any(substring in model_type for substring in substring_list):
               #tic_fp = time.clock()
               #y_preds=model(phase,batchX,batchFrames,y_history=y_history,batchY=batchY,batchUserStat=batchUserStat) 
               y_preds=model(phase,batchX,batchFrames) 
               #toc_fp = time.clock()
               #print('Forward Pass Time',toc_fp-tic_fp)
       else:
               y_preds=model(batchX,batchY_history,batchY, batchUserStat,phase)
       return y_preds


def train_iters(epoch, batch_idx, batch,model,model_type,push2gpu_flag):
        
       phase='train'
       seq_len=model.seq_len 
       target_len=model.target_len
       #print("IN TRAIN ITERS")
    

       batchX,batchUserStat,batchFrames,batchYSeq = get_batches4models(batch,model_type) 
       # first seq len -1 indices are history
       batchY_history=batchYSeq[:,0:seq_len-1,:]; batchY=batchYSeq[:,-target_len:,:]
     
##       print('batchY.type(): ',batchY.type())
       if USE_CUDA and push2gpu_flag:

          if single_batch: push2gpu_flag = False 
          #tic_batch_gpuload=time.clock()
          if batch_idx == 0 and epoch == 0:
                 print('ON GPU DEVICE '+str(device)+' '+str(torch.cuda.current_device()))
          batchX=batchX.contiguous().to(device,non_blocking=non_blocking)
          batchY=batchY.contiguous().to(device,non_blocking=non_blocking)
          #batchY_history=batchY_history.to(device,non_blocking=non_blocking)          
          #batchUserStat=batchUserStat.to(device,non_blocking=non_blocking)
          if any(substring in model_type for substring in substring_list):
                  batchFrames=load_frames4models(batchFrames,model_type)

       
       y_preds= forwardpass4models(batchX,batchFrames,batchY_history,batchY, batchUserStat,phase,model,model_type)
       #torch.cuda.empty_cache() 
       
       return  batchY_history,batchY,y_preds                   

def eval_iters( batch_idx, batch,model,model_type,phase):
    
    model.eval()
    
    with torch.no_grad():
       seq_len=model.seq_len 
       target_len=model.target_len
       batchX,batchUserStat,batchFrames,batchYSeq = get_batches4models(batch,model_type) 

       # first seq len -1 indices are history
       batchY_history=batchYSeq[:,0:seq_len-1,:]; batchY=batchYSeq[:,-target_len:,:]

       
       if USE_CUDA:
          if batch_idx == 0 :
                 print('Eval ON GPU DEVICE '+str(device)+' '+str(torch.cuda.current_device()))
          batchX=batchX.contiguous().to(device,non_blocking=non_blocking)
          batchY=batchY.contiguous().to(device,non_blocking=non_blocking)
          #batchY_history=batchY_history.to(device,non_blocking=non_blocking)          
          #batchUserStat=batchUserStat.to(device,non_blocking=non_blocking)
          if any(substring in model_type for substring in substring_list):
                  batchFrames=load_frames4models(batchFrames,model_type)

       y_preds= forwardpass4models(batchX,batchFrames,batchY_history,batchY, batchUserStat,phase,model,model_type)
       #torch.cuda.empty_cache() 

       return batchY_history,batchY,y_preds
   
def evaluate(model,phase, model_type,dataloader_eval):
     torch.cuda.empty_cache() 
 

     yTests_temp = []; yPred_temp = [];
     model.set_phase(phase)  
     with torch.no_grad():
#        batch = next(iter(dataloader_eval))
#         batch_idx = 1
         for batch_idx,batch in enumerate(dataloader_eval): 
             if debugPrint : print ("Pre eval iters allocated and cached ", batch_idx,torch.cuda.memory_allocated()/1e+9,torch.cuda.memory_cached()/1e+9)
    
             batchY_history,batchY,y_preds= eval_iters(batch_idx,batch,model,model_type,phase)

             yTests_temp.append(batchY.detach().cpu()); yPred_temp.append(y_preds.detach().cpu())
##             yTests_temp.append(batchY_history.detach().cpu()); yPred_temp.append(y_preds.detach().cpu())
             toc_batch=time.clock()
             
             
           
         yPreds =torch.cat(yPred_temp)
         yTests = torch.cat(yTests_temp)
     return yTests,yPreds 

             
def train(train_dict,test_dict,val_dict,model_type,param_vals, uid_manager):
 
    torch.cuda.empty_cache() 
    param_vals=scriptable_params(param_vals) 

    param_vals = OrderedDict(sorted(param_vals.items(), key=lambda t: t[0]))
    model_params=param_vals.fromkeys(param_vals.keys())
    print(model_params.keys())

    #stores final results
    log_avg_dict=defaultdict(list)
    
    cnt = 0; 
  
    for p_tups in product(*param_vals.values()):
         print ("\n New Param set \n") 
         
         trial_predictions=getLogger() 
           
         cnt = cnt + 1; 
         temp_params_dict=dict(zip(param_vals.keys(),p_tups) )
         model_params.update(temp_params_dict) 
         print(model_params)
         
         print ("runner xtrain y train shapes",train_dict['x'].shape, train_dict['y'].shape)
         print ("runner xtest y test shapes",test_dict['x'].shape, test_dict['y'].shape)
         print ("runner xVal y Val shapes",val_dict['x'].shape,val_dict['y'].shape)
          
         # training set loaded into TensorDataset, 
         # only batches will be loaded onto GPU to save GPU memory
         dataloaders= getDataloader(train_dict,test_dict,val_dict,model_type,model_params, uid_manager)
         tic = time.clock() 
         if debugPrint : print ("Mory allocated and cached",torch.cuda.memory_allocated()/1e+9,torch.cuda.memory_cached()/1e+9)
         for iter_trial in range(model_params['num_trials'])   :
              
              
              """
              #I/P FORMAT:
                  (model_type,model_params,dataloaders=None,override_opt=False,custom_optimizer=None,override_scheduler=False,custom_scheduler=None)
                  Define custom optimizer and scheduler and set corresponding Flags to True  
              """
              saveModel= getModule(model_type,model_params)         # to save all the model in trial
              model = getModule(model_type,model_params) 
              optimizer = model.getOptimizer()
              #class_weights = dm.get_weights4classes()
              #print('class_weights type', class_weights.type())
              #loss_fn =torch.nn.MSELoss(reduction='mean')
##              criterion = MultiLabelCrossEntropyLoss(class_weights)
##              accuracy_metric = SequenceMultiClassMetric(model_params)
              print(unnecessary_model_name)
              print(model)
              criterion = torch.nn.MSELoss(reduction='mean')#RMSELoss()
              #if USE_CUDA: criterion= criterion.to(device,non_blocking=non_blocking)              
              
              if debugPrint : print ("Model, loss mem allocated and cached",torch.cuda.memory_allocated()/1e+9,torch.cuda.memory_cached()/1e+9)

              # early stopping pre reqs
              patience=model_params['EarlyStoppingPatience'];               # to track the training loss as the model trains
              train_losses = []; valid_losses = []; avg_train_losses = []; avg_valid_losses = [] ; lr_schedule = [];

              if indie_ckpt:
                  checkpoint_path = dm.get_checkpoints_dir()
                  dm.make_folder_in_results(checkpoint_path)
              else:
                  checkpoint_path = dm.get_results_dir()
                  
              early_stopping = EarlyStopping(patience=patience, verbose=False,delta=model_params['Early_Stopping_delta'],checkpoint_path=checkpoint_path,param_num=cnt,trial=iter_trial)
              
 
              n_epochs=model_params['num_epochs']; epoch_len = len(str(n_epochs))
              tic_trials = time.clock() 
              print ("Training")

              push2gpu_flag = True
              
              sanity_batch = [(1,next(iter(dataloaders['train'])))]
              #batch = next(iter(dataloaders['train']))
              #batch_idx = 1
              for epoch in range(n_epochs):
                   tic_EPC = time.clock()
                   if sanity_check == False:
                       torch.cuda.empty_cache() 
                   model.set_phase('train')
##                   accuracy_metric = SequenceMultiClassMetric(model_params)

                   lr_schedule.append(optimizer.param_groups[0]['lr'])
                   #tic_batch=time.clock()

                   rangevar = sanity_batch  if single_batch else enumerate(dataloaders['train'])
                   #rangevar =  enumerate(dataloaders['train'])         
                   for batch_idx,batch in rangevar:
                       #print("batch_total_time:",time.clock()-tic_batch)
                       #tic_batch=time.clock()
                       

                        
                       if debugPrint : print ("Pre train iters allocated and cached ", batch_idx,torch.cuda.memory_allocated()/1e+9,torch.cuda.memory_cached()/1e+9)
                       
                       phase='train'

                       #print('batch:',batch_idx)
                       #print('Entering Train_Iter')    
                       batchY_history,batchY,y_pred= train_iters(epoch, batch_idx,batch,model,model_type,push2gpu_flag)
                       

##                       print('Pred is Cuda', y_pred.is_cuda)
##                       print('True is Cuda', batchY.is_cuda)
##                       print("ypred shape",y_pred.shape)
##                       print("batchY shape",batchY.shape)
                       #print('Computing Loss')  
                       loss = criterion(y_pred.view(batchY.shape), batchY)                       

##                       loss = criterion(y_pred, batchY_history)                       

                       # Backward pass
                       #print('BACKWARD PASS')
                       loss.backward()

                       
                       if model_params['minibatch_GD'] == True:
                           #print('Gradient Step')  
                           model.stepOptimizer()
                           model.zero_grad()
                           #optimizer = model.getOptimizer()
                       
##                       accuracy_metric.record_output(y_pred.data, batchY_history.data, y_pred.size(0))
                            
##                       print("Accuracy this batch: ",accuracy_metric.report())
#                       iter_losses[e_i * iter_per_epoch + t_i // t_cfg.batch_size] = loss
                        
                       # Clip gradients: gradients are modified in place
                       if clipGrads:
                           _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
                           _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)
    
                       # Update parameters
##                       model.stepOptimizer()
                       
                       # record training loss
##                       train_losses.append(loss.item())
                       #toc_batch=time.clock()
                       #print("batch_total_time:",toc_batch-tic_batch)
                   if model_params['minibatch_GD'] == False:
                       model.stepOptimizer()
                       model.zero_grad()
                       
                   if epoch% model_params['valid_patience']==0: print ("Epoch:",epoch,"loss:",loss.item())
                   #print ("Epoch:",epoch,"loss:",loss.item())
                   ######################    
                   # validate the model #
                   ######################
                   if epoch % model_params['valid_patience'] == 0:
                        
##                        accuracy_metric_eval = SequenceMultiClassMetric(model_params)
                        ## phase set to train as this is within training loop and 
                        ## not to be recorded
                        if sanity_check == False:
                            yVal,yVal_pred=evaluate(model,phase, model_type,dataloaders['val'])
                            if USE_CUDA:
                                yVal=yVal#.contiguous().to(device,non_blocking=non_blocking)
                                yVal_pred=yVal_pred#.contiguous().to(device,non_blocking=non_blocking)

##                        print('Pred is Cuda', yVal_pred.is_cuda)
##                        print('True is Cuda', yVal_pred.is_cuda)
                            this_epoch_Val_loss=criterion(yVal_pred, yVal)

                        if model_params['valid_patience'] == 1 and model_params['use_scheduler']==True:
                            # Update parameters
                            if sanity_check == False:
                                model.stepScheduler(this_epoch_Val_loss)  #$$$
                                scheduler = model.getScheduler()
                            else:
                                model.stepScheduler(loss)
                                scheduler = model.getScheduler()

##                            print('SCHEDULER DICT: ',scheduler.state_dict())
                            
                        # record training loss
                        train_losses.append(loss.item())


##                        accuracy_metric_eval.record_output(yVal_pred.data, yVal.data, y_pred.size(0))
##                        print("Accuracy this batch: ",accuracy_metric_eval.report())
                        if sanity_check == False:
                            valid_losses.append(this_epoch_Val_loss.item())
                        
                           # print training/validation statistics 
                        # calculate average loss over this  epoch
                        train_loss = np.average(train_losses)
                        avg_train_losses.append(train_loss)

                        
                        if sanity_check == False:
                            valid_loss = np.average(valid_losses)                        
                            avg_valid_losses.append(valid_loss)

                            print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' + f'train_loss: {train_loss:.5f} ' + f'valid_loss: {valid_loss:.5f}')
                        toc_EPC = time.clock()
                        print("Epoch ", epoch, "! time elapsed in this epoch: ", toc_EPC-tic_EPC)
                        if sanity_check == False:                        
                            print(print_msg)
                        print('Optimizer_LR: ',optimizer.param_groups[0]['lr'])#,'; Scheduler_LR: ', scheduler.get_lr()[0])

                   if sanity_check == False:
                       early_stopping(valid_loss, model)   #$$$ 
                   else:
                       #print("TRAIN LOSS:",train_loss)
                       early_stopping(train_loss, model)#valid_loss, model)   #$$$
        
                   if early_stopping.early_stop:
                        print("Early stopping")
                        break 
                           
 
                           
        
                   # clear lists to track next epoch
                   train_losses = []
                   valid_losses = [] 
              
                  
               
                    

              #<-------- each epoch ends here   

              # load the last checkpoint with the best model
              if indie_ckpt:
                  model.load_state_dict(torch.load(checkpoint_path+'checkpoint'+'_param_'+str(cnt)+'_trial_'+str(iter_trial)+'.pt'))
              else:
                  model.load_state_dict(torch.load(checkpoint_path+'checkpoint.pt'))
              
              #avg_train_losses, avg_valid_losses 
              print ("training done")
              
              print ("Saving model for param # ", cnt, " trial # ", iter_trial)
              saveModel.load_state_dict(model.state_dict()) # copy weights
              saveTorchModel(saveModel,model_type, cnt, iter_trial)   
              torch.cuda.empty_cache()
              this_trial_Val_loss = 0
              this_trial_test_loss = 0
              if sanity_check == False:
              # evaluate end of trial losses
                      phase='val'

##                     accuracy_metric_eval = SequenceMultiClassMetric(model_params)

                      yVal,yVal_pred=evaluate(model,phase,model_type,dataloaders['val'])
                      if USE_CUDA:
                          yVal=yVal#.contiguous().to(device,non_blocking=non_blocking)
                          yVal_pred=yVal_pred#.contiguous().to(device,non_blocking=non_blocking)
                      this_trial_Val_loss=criterion(yVal_pred, yVal).item()
##              accuracy_metric_eval.record_output(yVal_pred.data, yVal.data, y_pred.size(0))
##              print("Accuracy this batch: ",accuracy_metric_eval.report())
                        
                      print ("val RMSE error",this_trial_Val_loss)
              
                      del yVal, yVal_pred
              
              #test error   
                      phase='test'

##              accuracy_metric_eval = SequenceMultiClassMetric(model_params)
                      y_tests,y_preds=evaluate(model,phase,model_type,dataloaders['test'])
                      if USE_CUDA:
                          y_tests=y_tests.contiguous().to(device,non_blocking=non_blocking)
                          y_preds=y_preds.contiguous().to(device,non_blocking=non_blocking)
                      if any(substring in model_type for substring in substring_list):
                          save_vision_preds(model,model_type,phase,cnt, iter_trial)

                      this_trial_test_loss=criterion(y_pred.view(batchY.shape), batchY).item()

                      print ("trial", iter_trial, " test loss",this_trial_test_loss)           
##              accuracy_metric_eval.record_output(y_preds.data, y_tests.data, y_preds.size(0))
##              report_test, correct_percentage_test, correct_preds_test = accuracy_metric_eval.final_report()
##              print("Test Accuracy: ",report_test)
##              report_train, correct_percentage_train, correct_preds_train = accuracy_metric.final_report()
#              y_tests_np, y_preds_np=format_target(model_type,yTests,yPreds)
##              print("Train Accuracy: ",report_train)    
             
                      print ("yTests and preds shape",y_tests.shape, y_preds.shape)
              # save every trial pred and loss   
                      trial_predictions['y_preds'].append(y_preds.detach().cpu().numpy());
                      trial_predictions['y_tests'].append(y_tests.detach().cpu().numpy())
              #trial_predictions['x_frames'].append(x_frames  )

              trial_predictions['trial_info'].append(dict(model_params))
              trial_predictions['test_Loss'].append(this_trial_test_loss)
              trial_predictions['val_Loss'].append(this_trial_Val_loss)
              trial_predictions['Epoch_vs_TrainLoss'].append(avg_train_losses)
              trial_predictions['Epoch_vs_ValLoss'].append(avg_valid_losses)
              trial_predictions['LR_Schedule'].append(lr_schedule)
##              trial_predictions['Train_Report'].append(correct_percentage_train.detach().cpu().numpy())
##              trial_predictions['Test_Report'].append(correct_percentage_test.detach().cpu().numpy())
##              trial_predictions['Train_Confusion'].append(correct_preds_train.detach().cpu().numpy())
##              trial_predictions['Test_Confusion'].append(correct_preds_test.detach().cpu().numpy())
#              # for seq to seq full samples will be target sequences
#              trial_predictions['test_full_samples'].append(yTests.cpu().detach().numpy() )
#              trial_predictions['pred_full_samples'].append(yPreds.cpu().detach().numpy() )

              
              # memory management
              del model,saveModel
              if sanity_check==False:
                      del y_tests, y_preds                    
              torch.cuda.empty_cache()
              dm.save_as_mat(['outputs','LR_Schedules'],trial_predictions)
              """ <-- each trial ends here """
         
          
         dm.save_as_mat(['outputs','predictions' + str(cnt)],trial_predictions)    

          #avg loss for these parameters for all trials
         
         avg_error =  float(np.mean(trial_predictions['test_Loss']));

         print ("\n model params",temp_params_dict.items())  
         print ("\n Avg loss out: %f, this param set # %d"%(avg_error,cnt))  

#        # save avg loss for these parameters
#         temp_params_dict['param_idx']= cnt; temp_params_dict['error']= avg_error
#         log_avg_dict = append_to_log(log_avg_dict,temp_params_dict)

         toc = time.clock()
         print ("time elapsed for this parameter set",toc - tic)     
              
              
              
         """ <-- param set ends here """ 
         
    
    #  save average performance all parameter combinations in one file  
#    dm.save_as_mat(['hyperparams','average'],log_avg_dict)      
    return trial_predictions
