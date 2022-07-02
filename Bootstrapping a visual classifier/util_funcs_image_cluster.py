# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 21:42:17 2020

@author: vijet
"""
import sys
from pathlib import Path, PurePath
import torch

from fastai.vision import *
import pandas as pd
import numpy as np
import time
from PIL import Image as pil_img
#import cv2   
from datetime import datetime
from scipy.spatial.distance import cosine
import scipy.io as sio
timestamp= datetime.now().strftime("%m_%d_%H_%M")
from sklearn.manifold import TSNE
import plotly_express as px
#from annoy import AnnoyIndex
from sklearn.cluster import KMeans 
from sklearn import mixture
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

data_dir = '../../../data/vision_pretraining/sampled_classified'  
frames_data_dir = '../../../data/vision_pretraining/processed'  

expt_dir=''
tmfs = get_transforms()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

class Hook():
    "Create a hook on `m` with `hook_func`."
    def __init__(self, m:nn.Module, hook_func:HookFunc, is_forward:bool=True, detach:bool=True):
        self.hook_func,self.detach,self.stored = hook_func,detach,None
        f = m.register_forward_hook if is_forward else m.register_backward_hook
        self.hook = f(self.hook_fn)
        self.removed = False

    def hook_fn(self, module:nn.Module, input:Tensors, output:Tensors):
        "Applies `hook_func` to `module`, `input`, `output`."
        if self.detach:
            input  = (o.detach() for o in input ) if is_listy(input ) else input.detach()
            output = (o.detach() for o in output) if is_listy(output) else output.detach()
        self.stored = self.hook_func(module, input, output)

    def remove(self):
        "Remove the hook from the model."
        if not self.removed:
            self.hook.remove()
            self.removed=True

    def __enter__(self, *args): return self
    def __exit__(self, *args): self.remove()
        
def get_output(module, input_value, output):
    return output.flatten(1)

def get_input(module, input_value, output):
    return list(input_value)[0]

def get_named_module_from_model(model, name):
    for n, m in model.named_modules():
        if n == name:
            return m
    return None


""" Load data to bunch"""

def get_train_databunch_from_csv(csv_files_list,batch_size):
    
    li = []

    for filename in csv_files_list:
        csv_file=Path(data_dir,'data_csv',filename)

        df = pd.read_csv(csv_file, index_col=None, header=0)
        li.append(df)

    data_df = pd.concat(li, axis=0, ignore_index=True)

    print ("merdeg train df size",data_df.shape)
    
    data_source_unsplit = (ImageList.from_df(df=data_df, path=frames_data_dir, cols='frame_file')
                        .split_none()
                        .label_from_df(cols='cluster_label')
                  )
    data_source = (ImageList.from_df(df=data_df, path=frames_data_dir, cols='frame_file')
                        .split_by_rand_pct(valid_pct=0.1)
                        .label_from_df(cols='cluster_label')
                  )
    
    data = data_source.transform(tmfs, size=224).databunch(bs=batch_size,device=device).normalize(imagenet_stats)
    data_unsplit = data_source_unsplit.transform(tmfs, size=224).databunch(bs=batch_size,device=device).normalize(imagenet_stats)


    print (data)
    print (device)

    
    return data_df,data,data_unsplit



def get_unsplit_databunch_from_csv(csv_files_list,inf_bs):
    
    
    li = []

    for filename in csv_files_list:
        csv_file=Path(data_dir,'data_csv',filename)

        df = pd.read_csv(csv_file, index_col=None, header=0)
        li.append(df)

    data_df = pd.concat(li, axis=0, ignore_index=True)
    data_df=data_df.sort_values('dataset_index')

    print ("merdeg inference df size",data_df.shape)
    data_df.head()
    data_bunch = (ImageList.from_df(df=data_df, path=frames_data_dir, cols='frame_file')
                        .split_none()
                        .label_from_df(cols='cluster_label')
                  )
    
    data_bunch = data_bunch.transform(tmfs, size=224).databunch(bs=inf_bs,num_workers=0).normalize(imagenet_stats)

    print  (data_bunch)
    print (device)

    return data_df,data_bunch




"""" saving """""


def  pickle_save_dataframe(expt_dir,data_df,save_as):
    
    data_df.to_pickle(Path(expt_dir,save_as).with_suffix('.pkl'))
        
    return 
    

def pickle_read_dataframe(expt_dir,saved_pickle) :
    
    pickle_file=Path(expt_dir,saved_pickle).with_suffix('.pkl')
    print ("Reading pickle file:",pickle_file)
    data_df=pd.read_pickle(pickle_file)
    
    return data_df

def read_results_df(rel_expt_dir,file_name):
    
    results_full_path=Path(data_dir,'results',rel_expt_dir,file_name)
    print ("Reading dataframe",results_full_path)
    results_df=pd.read_pickle(results_full_path)

    return results_df

    
def save_results_df(expt_dir,result_df, save_as='inf_results'):
    

    result_df.to_pickle(Path(expt_dir,save_as).with_suffix('.pkl'))
    result_df.to_csv(Path(expt_dir,save_as).with_suffix('.csv'))
    
    result_dict=result_df.to_dict(orient='list')
    sio.savemat(Path(expt_dir,save_as).with_suffix('.mat'),result_dict)


    
    return result_df 
def save_figs(expt_dir,figs, save_as):
    
    with open(Path(expt_dir,save_as+'.pkl'), 'wb') as f:  
        pickle.dump(figs, f)
        
    png_path=str(Path(expt_dir,save_as+'.png') )   
   
    figs.write_image(png_path,scale=3)    
        
    
def save_results_with_figures(expt_dir,results_list, save_as='inf_results'):
    
    """ saves final results as pibkle, csv,matfile"""
    
     # Saving the objects:
    result_df=results_list[0]  #to do dict 
    result_figs=results_list[1:2]
    
    with open(Path(expt_dir,save_as+'figs.pkl'), 'wb') as f:  
        pickle.dump(result_figs, f)

    result_df.to_pickle(Path(expt_dir,save_as).with_suffix('.pkl'))
    result_df.to_csv(Path(expt_dir,save_as).with_suffix('.csv'))
    
    result_dict=result_df.to_dict(orient='list')
    sio.savemat(Path(expt_dir,save_as).with_suffix('.mat'),result_dict)
    
#    plt.plot(result_df.img_cluster_labels)
#    plt.plot(result_df.classifier_labels)
        

    
    return result_df

def read_img_rep_matrix_from_df(results_df):
    
    print ("loading presaved img reps from dataframe" )
    img_rep_matrix_list=list(results_df.img_repr.array)
    num_samples=len(img_rep_matrix_list)
    num_features=len(img_rep_matrix_list[0])
    img_rep_matrix=np.empty((num_samples,num_features))
    for idx,sample in enumerate(img_rep_matrix_list):
       img_rep_matrix[idx,:]=  np.asarray(sample)
       
       
    return    img_rep_matrix


def create_expt(anno='_test'):
    
    global expt_dir
    expt_dir = os.path.join(data_dir,'results/', str(timestamp) + anno) 

    os.mkdir( expt_dir );
    
    return expt_dir



def set_expt_dir(rel_expt_dir):
    global expt_dir
    
    expt_dir = os.path.join(data_dir,'results/', rel_expt_dir) 

    return expt_dir




####    
"""Classifier pretrained
"""
    

def train_model_scratch(train_databunch,b_pretrain=False, n_iters=50,lr=1e-04,save_anno=""):
    
    print ("Training from scratch",b_pretrain)
    learner = cnn_learner(train_databunch, models.resnet18, pretrained=False, metrics=accuracy)
    learner.path=Path(data_dir)       
    learner.unfreeze()

    learner.lr_find()
    learner.recorder.plot()
    print("train params lr, iters",lr,n_iters)
    learner.fit_one_cycle(n_iters*2, max_lr=lr)
    learner.save('resnet18-224-unfreezed'+save_anno)
#    learner.load('resnet18-224-freezed'+save_anno)

   
    
    print("Unfrozen model saved ",learner.path)

#    learner.export('trained_model.pkl')

    
    return learner 



def train_model(train_databunch,b_pretrain=True, n_iters=50,lr=1e-04,save_anno=""):
    
    print ("Pretrain or not",b_pretrain)
    learner = cnn_learner(train_databunch, models.resnet18, pretrained=b_pretrain, metrics=accuracy)
    learner.path=Path(data_dir)       
        
    learner.lr_find()
    learner.recorder.plot()
    print("train params lr, iters",lr,n_iters)
    learner.fit_one_cycle(n_iters, max_lr=lr)
    learner.save('resnet18-224-freezed'+save_anno)
#    learner.load('resnet18-224-freezed'+save_anno)

    
    print ("unfreezing and training for iters: ",n_iters)
    learner.unfreeze()
    learner.lr_find()
    learner.recorder.plot()
    learner.fit_one_cycle(n_iters, slice(1e-5, 1e-2/5))   
    
    
    print("Unfrozen model saved ",learner.path)

    learner.save('resnet18-224-unfreezed'+save_anno)
#    learner.export('trained_model.pkl')

    
    return learner 

def get_pretrained_model_db(data_db,saved_model='resnet18-224-unfreezed')    :
    
        #        learner = cnn_learner(inf_databunch, models.resnet18, pretrained=False, metrics=accuracy)
        print ("loading pretrained model",saved_model)

        learner = cnn_learner(data_db, models.resnet18, metrics=accuracy)
        learner.path=Path(data_dir)
        learner.load(saved_model)
        
        data_model_dir=Path(data_dir,'models')
        learner.export(Path(data_model_dir,'trained_model.pkl'))
        learner=load_learner(data_model_dir,'trained_model.pkl')

        return learner
        
def get_classes_probs(learner, data_bunch):
        
    num_samples=len(data_bunch.train_ds)    
    
    inf_predictions_dict={}
    inf_probs_dict={}
#    probs=np.empty((num_samples,3));preds=np.empty(num_samples);img_ids=np.empty(num_samples)
    
    for i in range(num_samples):

        img_full_id=Path(data_bunch.train_dl.items[i])
        rel_img_id=img_full_id.relative_to(frames_data_dir)

        
        img=data_bunch.train_ds[i][0]
        
        x,y_pred,y_prob=learner.predict(img)
    
        inf_predictions_dict[str(rel_img_id)] = y_pred.detach().cpu().numpy().reshape(-1)[0]
        inf_probs_dict[str(rel_img_id)] =y_prob.detach().cpu().numpy()


    inf_preds_df=pd.DataFrame.from_dict(inf_predictions_dict,orient='index',columns=['classifier_labels'])
    inf_probs_df=pd.DataFrame.from_dict(inf_probs_dict,orient='index',columns=['class_probs_0','class_probs_1','class_probs_2'])
     
    inf_preds_df=inf_preds_df.join(inf_probs_df)
    inf_preds_df['frame_file']=inf_preds_df.index
    return inf_preds_df


def classify_add_df(learner,data_db,data_df):
    
    print ("classifying")
    preds_probs_df =get_classes_probs(learner,data_db)

    data_df=data_df.merge(preds_probs_df,on='frame_file')


        
    return data_df    



""" Img reps and TSNE
"""
def get_clean_str(img_ids_fmt3648):
    
    num_samples=img_ids_fmt3648.shape[0]
    
#    img_id_str=np.empty((num_samples))
    img_id_str=[]
    for idx in range(num_samples):
        img_id_str.append(str(img_ids_fmt3648[idx]))
        
    return    img_id_str 
        
        


def _get_img_rep_batch(batch_idx,batch,inference_dataloader,model,hook):

        with torch.no_grad():

            xb,yb=batch
#            bs = xb.shape[0]
            bs=inference_dataloader.batch_size
            img_ids =get_clean_str(inference_dataloader.items[batch_idx*bs: (batch_idx+1)*bs])

            model_preds = model.eval()(xb)
            img_reprs = hook.stored.cpu().numpy()
            
#            print("img ids",batch_idx,len(img_ids))

            #img_reprs = img_reprs.reshape(bs, -1)


            return img_ids,img_reprs.tolist()


def get_image_reps(expt_dir,inference_data,linear_output_layer,model):
    
    img_repr_map = {}
    all_img_repr_map={}
    print ("getting image embed vectors")
    all_ids=[];all_reps=[];
    inference_dataloader = inference_data.train_dl.new(drop_last =False, shuffle=False)

    with Hook(linear_output_layer, get_output, True, True) as hook:
        for batch_idx, batch in enumerate(inference_dataloader):
            torch.cuda.empty_cache() 

            img_ids,img_reprs=_get_img_rep_batch(batch_idx,batch,inference_dataloader,model,hook)
        
            all_ids=all_ids+img_ids;all_reps =all_reps + img_reprs;
            for img_id, img_repr in zip(img_ids, img_reprs):
                full_path=Path(img_id)
                rel_path=full_path.relative_to(frames_data_dir)
                img_repr_map[str(rel_path)] = img_repr
                
    for img_id, img_repr in zip(all_ids, all_reps)  :

          full_path=Path(img_id)
          rel_path=full_path.relative_to(frames_data_dir)
          all_img_repr_map[str(rel_path)] = img_repr
          
    img_repr_df = pd.DataFrame(img_repr_map.items(), columns=['frame_file', 'img_repr'])
    
    all_img_repr_df = pd.DataFrame(all_img_repr_map.items(), columns=['frame_file', 'img_repr'])


    img_repr_df['kine_labels'] = [inference_data.classes[x] for x in inference_data.train_ds.y.items[0:img_repr_df.shape[0]]]
#    img_repr_df['kine_labels']=img_repr_df['kine_labels']-1
    
#    img_repr_df.to_pickle(Path(data_dir,expt_dir,'img_reps_df.pkl'))
    
    return img_repr_df


    


def plotly_tsne(img_repr_df,clr_labels,expt_dir,save_as):
    
    fig=px.scatter_3d(img_repr_df, x='tsne1', y='tsne2', z='tsne3',color=clr_labels,
                     width=600, height=400)

    fig.update_layout(
    margin=dict(l=5, r=5, t=5, b=40),
    )
    fig.show()
    
    
#    fig.write_image(os.path.join(expt_dir,save_as+'.png'))
    
    

    return fig.to_dict() 
def get_img_reps_mat_add_df(bImageRepsNow,expt_dir,data_df, img_databunch,linear_output_layer,model,presaved_file='img_reps_df.pkl'):
   
    " get img rep matrix and add to base df"
    
       #get image reps
    if bImageRepsNow:
        img_repr_df=get_image_reps(expt_dir,img_databunch,linear_output_layer,model)   
    else :   
        print ("loading presaved reps")
        img_repr_df=pd.read_pickle(Path(data_dir,expt_dir,presaved_file))
        
    
    data_df = pd.merge(data_df, img_repr_df, on='frame_file')   

    img_repr_matrix = np.array([list(x) for x in img_repr_df['img_repr'].values])

    return   data_df,img_repr_matrix

###TSNE

def get_tsne(img_repr_matrix,k):
    tsne = TSNE(n_components=k, verbose=10, init='pca', perplexity=30, n_iter=500, n_iter_without_progress=100)
    tsne_results = tsne.fit_transform(img_repr_matrix)
    return tsne_results

def get_tsne_data_add_df(base_df,repr_matrix):

    """ get tsne representation and add to base df """
    
    tsne_results=get_tsne(repr_matrix,k=3)
    base_df['tsne1'] = tsne_results[:,0]
    base_df['tsne2'] = tsne_results[:,1]
    base_df['tsne3'] = tsne_results[:,2]
    
    return base_df


    
"""""
Clustering 

"""""
#def get_k_means_model(img_repr_matrix,n_clusters=3):
#    
#    print ("getting k means model...")
#    kmeans = KMeans(n_clusters,n_init=30) 
#    img_cluster_labels=kmeans.fit_predict(img_repr_matrix)
#    cluster_centers=kmeans.cluster_centers_    
#
#    
#    return kmeans,img_cluster_labels,cluster_centers

def fit_cluster_model(img_repr_matrix, cluster_model_params):
    
    
    if cluster_model_params['model_type']=='kmeans':
        print ("getting k means model...")
        cluster_model = KMeans(cluster_model_params['n_clusters'],n_init=30) 
        cluster_model=cluster_model.fit(img_repr_matrix)
        
    if cluster_model_params['model_type']=='DBSCAN':
        print ("getting DBSCAN model...")
        cluster_model= DBSCAN(eps=0.5,min_samples= cluster_model_params['min_samples'],metric=cluster_model_params['metric'])
#        cluster_model=cluster_model.fit(img_repr_matrix)
        
    if cluster_model_params['model_type']=='Agglomerative':
        print ("getting Agglomerative model...")
        cluster_model= AgglomerativeClustering(affinity=cluster_model_params['affinity'], n_clusters=None,linkage='average', distance_threshold=cluster_model_params['distance_threshold'])
#        cluster_model=cluster_model.fit(img_repr_matrix)   
        
    elif cluster_model_params['model_type']=='gaussian':
        
        print ("getting gaussian mixture")
        # fit a Gaussian Mixture Model 
        cluster_model = mixture.GaussianMixture(n_components=cluster_model_params['n_components'], covariance_type='full',n_init=10)
        cluster_model=cluster_model.fit(img_repr_matrix)
    
    elif cluster_model_params['model_type']=='bayes_gaussian':
        
        print ("getting bayesian gaussian mixture")
        # fit a Gaussian Mixture Model 
        cluster_model = mixture.BayesianGaussianMixture(n_components=cluster_model_params['n_components'], covariance_type='full',n_init=10)
        cluster_model=cluster_model.fit(img_repr_matrix)

    
    return cluster_model



def repeat_cluster_centers(cluster_centers,img_cluster_labels)   :
    
    num_samples=img_cluster_labels.shape[0]
    cluster_center_vect=np.zeros((num_samples,512))
    for idx,cluster_lbl in enumerate(img_cluster_labels):
        cluster_center_vect[idx]=cluster_centers[cluster_lbl]


    return cluster_center_vect.tolist()   

def predict_clusters_add_df(cluster_model,cluster_model_params,repr_df,cond_str=''):

    repr_matrix=read_img_rep_matrix_from_df(repr_df)
    model_type=cluster_model_params['model_type']
    if model_type=='kmeans':
        print ("predicting k means model...")
        
        img_cluster_labels=cluster_model.predict(repr_matrix)
        cluster_dists=cluster_model.transform(repr_matrix)
        cluster_centers=cluster_model.cluster_centers_    
    
        cluster_centers_repeated=repeat_cluster_centers(cluster_centers,img_cluster_labels)

        repr_df['kmeans_labels'+cond_str]=img_cluster_labels
        repr_df['kmeans_dist_0'+cond_str]=cluster_dists[:,0]
        repr_df['kmeans_dist_1'+cond_str]=cluster_dists[:,1]
        repr_df['kmeans_dist_2'+cond_str]=cluster_dists[:,2]
    
        repr_df['kmeans_centers'+cond_str]=cluster_centers_repeated
        
        
    elif model_type=='DBSCAN':
        
        print ("predicting using DBSCAN model")
        
        img_cluster_labels=cluster_model.fit_predict(repr_matrix)
        cluster_components=cluster_model.components_
        cluster_core_sample_indices=cluster_model.core_sample_indices_   
        
        repr_df['DBSCAN_labels'+cond_str]=img_cluster_labels
        
        print ("DBSCAN components shape",cluster_components.shape)
        print ("DBSCAN core sample indices  shape",cluster_core_sample_indices.shape)

    elif model_type=='Agglomerative':
        
        print ("predicting using Agglomerative model")
        
        img_cluster_labels=cluster_model.fit_predict(repr_matrix)
#        _n_clusters=cluster_model.n_clusters_
#        cluster_core_sample_indices=cluster_model.core_sample_indices_   
        
        repr_df['Agglomerative_labels'+cond_str]=img_cluster_labels        
        
    elif model_type=='gaussian':
        
        print ("predicting using gaussian mixture")
        
        img_cluster_labels=cluster_model.predict(repr_matrix)
        cluster_probs=cluster_model.predict_proba(repr_matrix)
        cluster_centers=cluster_model.means_   
        
        cluster_centers_repeated=repeat_cluster_centers(cluster_centers,img_cluster_labels)
        
        repr_df['gaussian_labels'+cond_str]=img_cluster_labels
        repr_df['gaussian_prob_0'+cond_str]=cluster_probs[:,0]
        repr_df['gaussian_prob_1'+cond_str]=cluster_probs[:,1]
        repr_df['gaussian_prob_2'+cond_str]=cluster_probs[:,2]
    
        repr_df['gaussian_centers'+cond_str]=cluster_centers_repeated
        
    
    elif model_type=='bayes_gaussian':
        
        print ("predicting bayesian gaussian mixture")
        img_cluster_labels=cluster_model.predict(repr_matrix)
        cluster_probs=cluster_model.predict_proba(repr_matrix)
        cluster_centers=cluster_model.means_ 
       
        
        cluster_centers_repeated=repeat_cluster_centers(cluster_centers,img_cluster_labels)
 
           
        repr_df['bayes_gaussian_labels'+cond_str]=img_cluster_labels
        repr_df['bayes_gaussian_prob_0'+cond_str]=cluster_probs[:,0]
        repr_df['bayes_gaussian_prob_1'+cond_str]=cluster_probs[:,1]
        repr_df['bayes_gaussian_prob_2'+cond_str]=cluster_probs[:,2]
    
        repr_df['bayes_gaussian_centers'+cond_str]=cluster_centers_repeated



    return repr_df    

def predict_clusters_all_models(expt_dir,data_df,cluster_input_mat,cond_str):
    
    

    clr_labels='kine_labels'
#    fig_kine_labels= plotly_tsne(data_df,clr_labels,expt_dir,save_as=cond_str+clr_labels)

    cluster_model_params= {
                            'kmeans': {'model_type':'kmeans','n_clusters':3},
                            'DBSCAN':{'model_type':'DBSCAN','eps':0.5,'min_samples':500,'metric':'cosine'},
                            'Agglomerative':{'model_type':'Agglomerative','affinity':'cosine', 'distance_threshold':0.5},
#                            'gaussian': {'model_type':'gaussian','n_components':3},
#                            'bayes_gaussian':{'model_type':'bayes_gaussian','n_components':3}
                            
                          }
    

    for key in cluster_model_params:
        print(key + ':', cluster_model_params[key])
        
        this_cluster_model= fit_cluster_model(cluster_input_mat,cluster_model_params[key])
        data_clustered_df= predict_clusters_add_df(this_cluster_model,cluster_model_params[key],data_df,cond_str)


    
    return data_clustered_df

def predict_clusters_all_models0(expt_dir,data_df,cluster_input_mat,cond_str):
    
    

    clr_labels='kine_labels'
#    fig_kine_labels= plotly_tsne(data_df,clr_labels,expt_dir,save_as=cond_str+clr_labels)

    model_type='k_means'
    kmeans,_,_= get_cluster_model(cluster_input_mat,model_type,n_clusters=3)
    data_df= predict_clusters_add_df(kmeans,model_type,data_df,cond_str)
    clr_labels='kmeans_cluster_labels'+cond_str
#    fig_kmeans= plotly_tsne(data_df,clr_labels,expt_dir,save_as=cond_str+clr_labels)


    model_type='DBSCAN '
    DBSCAN,_,_= get_cluster_model(cluster_input_mat,model_type,n_clusters=3)
    data_df= predict_clusters_add_df(DBSCAN,model_type,data_df,cond_str)
    clr_labels='dbscan_labels'+cond_str
#    fig_dbscan= plotly_tsne(data_df,clr_labels,expt_dir,save_as=cond_str+clr_labels)


    
    model_type='gaussian'
    gaussian,_,_= get_cluster_model(cluster_input_mat,model_type,n_clusters=3)
    data_df= predict_clusters_add_df(gaussian,model_type,data_df,cond_str)
    clr_labels='gaussian_cluster_labels'+cond_str
#    fig_gauss= plotly_tsne(data_df,clr_labels,expt_dir,save_as=cond_str+clr_labels)


    model_type='bayes_gaussian'
    bayes_gaussian,_,_= get_cluster_model(cluster_input_mat,model_type,n_clusters=3)
    data_df= predict_clusters_add_df(bayes_gaussian,model_type,data_df,cond_str)   
    clr_labels='bayes_gaussian_cluster_labels'+cond_str
#    fig_bayes_gauss= plotly_tsne(data_df,clr_labels,expt_dir,save_as=cond_str+clr_labels)
    
    
#    fig_dict={'fig_kine_labels':fig_kine_labels,     
#              'fig_kmeans':fig_kmeans,
#              'fig_gauss':fig_gauss,
#              'fig_bayes_gauss':fig_bayes_gauss     
#              }
    
    return data_df
def get_cluster_model(img_repr_matrix,model_type,n_clusters=3):
    
    
    
    if model_type=='kmeans':
        print ("getting k means model...")
        cluster_model = KMeans(n_clusters,n_init=30) 
        img_cluster_labels=cluster_model.fit_predict(img_repr_matrix)
        cluster_centers=cluster_model.cluster_centers_  
        
    if model_type=='DBSCAN':
        print ("getting DBSCAN model...")
        cluster_model= DBSCAN(eps=0.5, min_samples=500,metric='cosine')
        img_cluster_labels=cluster_model.fit_predict(img_repr_matrix)
        components=cluster_model.components_       
        
        
    elif model_type=='gaussian':
        
        print ("getting gaussian mixture")
        # fit a Gaussian Mixture Model 
        cluster_model = mixture.GaussianMixture(n_components=n_clusters, covariance_type='full',n_init=30)
        img_cluster_labels=cluster_model.fit_predict(img_repr_matrix)
        cluster_centers=cluster_model.means_   
    
    elif model_type=='bayes_gaussian':
        
        print ("getting bayesian gaussian mixture")
        # fit a Gaussian Mixture Model 
        cluster_model = mixture.BayesianGaussianMixture(n_components=n_clusters, covariance_type='full',n_init=30)
        img_cluster_labels=cluster_model.fit_predict(img_repr_matrix)
        cluster_centers=cluster_model.means_     

    
    return {cluster_model,img_cluster_labels,cluster_centers}  


###########
    
       
def get_full_trans_idx(trans_idx):
    full_trans_idx=[]
    for trans_i in trans_idx:
        
        
        trans_indices_3sec=np.arange(trans_i-18,trans_i)
        full_trans_idx.append(trans_indices_3sec)
    
    return list(np.hstack(full_trans_idx))

def list_avg(lst): 
    return sum(lst) / len(lst)

def medfilt (x, k):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros ((len (x), k), dtype=x.dtype)
    y[:,k2] = x
    for i in range (k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]
    return np.median (y, axis=1)


def get_transition_in_df(df,col_id):

    
    df['trans'] = df[col_id].sub(df[col_id].shift(1), axis = 0)
    
    flat_SA= df[df['trans']==1].index.to_list()
    flat_SD= df[df['trans']==2].index.to_list()

    SA_flat= df[df['trans']==-1].index.to_list()
    SD_flat= df[df['trans']==-2].index.to_list()
    
    trans_idx=flat_SA+flat_SD+SA_flat+SD_flat
    

    return trans_idx,flat_SA,flat_SD,SA_flat,SD_flat

def get_transition_as_df(df,col_id):

    
    df['trans'] = df[col_id].sub(df[col_id].shift(1), axis = 0)
    
    flat_SA= df[df['trans']==1].index.to_list()
    flat_SD= df[df['trans']==2].index.to_list()

    SA_flat= df[df['trans']==-1].index.to_list()
    SD_flat= df[df['trans']==-2].index.to_list()
    
    trans_idx=flat_SA+flat_SD+SA_flat+SD_flat
    
    trans_df=df.loc[trans_idx]
    return trans_idx,trans_df
