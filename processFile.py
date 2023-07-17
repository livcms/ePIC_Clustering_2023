import uproot
import numpy as np
import pandas as pd 
import awkward as ak
import time 

import hashlib
import xxhash
class Event: 
    def __init__(self, file, train_dir:str, test_dir:str, solution_dir:str):
        
        self.start_time = time.time() 
        self.train_dir = train_dir
        self.test_dir = test_dir 
        self.solution_dir = solution_dir

        file1 = uproot.open(file)
        self.tree1 = file1['event_tree']


    def make_file_from_evts(self, start_ev:int, stop_ev:int):
        self.start_ev  = start_ev
        self.stop_ev = stop_ev 
        self.arrs = self.tree1.arrays(entry_start=self.start_ev, entry_stop=self.stop_ev) 
        self.awk_df = ak.to_dataframe(self.arrs).reset_index()      
        self.awk_df['entry'] = self.start_ev + self.awk_df['entry']

        self.calc_truth_clusters_by_main_particle() 
        self.rename_columns() 


    def calc_truth_clusters_by_main_particle(self): 

        # the data is in a wide format, need to convert it to long to more easily select the particle with the highest energy 
        efrac = self.awk_df.filter(regex=("tower_LFHCAL_trueEfrac\w|entry"))
        id = self.awk_df.filter(regex=("tower_LFHCAL_trueID\w|entry"))

        e_melt = efrac.melt(id_vars = ['entry', 'subentry'], var_name='p_Efrac', value_name='Efrac')
        id_melt = id.melt(id_vars = ['entry', 'subentry'], var_name='p_id', value_name='id').drop(['entry', 'subentry'], axis=1)

        comb = pd.concat([e_melt, id_melt], axis=1)

        # there is an edge case about .3% of the time where there is an equal split between two particles. One of them will be randomly selected, but this should be negligible 
        # does the same as idxmax, but so much faster 
        highest_e_p = comb.sort_values(by='Efrac').drop_duplicates(['entry', 'subentry'], keep='last')


        # need to make unique ids for each event in the file - this runs quickly and ensures no duplicates 

        str_ids = highest_e_p['entry'].astype('str') + "_" + highest_e_p['id'].astype('str')
        highest_e_p['clusterID'] = [xxhash.xxh64_intdigest(x, seed=0) for x in str_ids.values] 


        # drop columns students don't need and merge with the correct cluster ids 
        noId_efrac = self.awk_df.filter(regex=("^((?!tower_LFHCAL_true).)*$"))
        noId_efrac= noId_efrac.drop(['tower_LFHCAL_NMCParticles', 'tower_LFHCAL_NContributions', 'EventID'], axis=1) 
        final = noId_efrac.merge(highest_e_p[['entry', 'subentry', 'clusterID', 'Efrac']], on=['entry', 'subentry'])

        # kaggle needs a unique id, and they can't repeat for different files, again use hash so there's no risk of a wrong overlap 

        str_ids = final['entry'].astype('str') + "_" + final['subentry'].astype('str')
        final['uniqueID'] = [xxhash.xxh64_intdigest(x, seed=0) for x in str_ids.values] 
        final['uniqueID'] = final['uniqueID'].astype(int)
        self.final = final 

    def rename_columns(self): 
        # every column starting with tower_LFHCAL annoys me 
        self.final = self.final.rename(dict(zip(list(self.final.filter(regex=("tower_LFHCAL*")).columns), ['N', 'E', 'T', 'ix', 'iy', 'iz', 'posx', 'posy', 'posz'])), axis=1)
        self.final = self.final.rename({'entry':'event', 'subentry':'hit_number'}, axis=1)

    def save_train_csv(self): 
        self.final.to_csv(self.train_dir+'train_ePIC_event'+str(self.start_ev)+"_"+str(self.stop_ev)+'.csv', index=False)
    
    def save_test_csv(self): 
        #will not have energy fraction and clusterID 
        test_df = self.final.drop(['clusterID', 'Efrac'], axis=1)
        test_df.to_csv(self.test_dir+'test_ePIC_event'+str(self.start_ev)+"_"+str(self.stop_ev)+'.csv', index=False)


    def save_solution_csv(self,file_num:int, is_private=0): 
        solution_df = self.final[['uniqueID', 'clusterID', 'E']]
        if is_private: 
            usage = "Private"
        else:  
            usage = "Public"

        solution_df['Usage'] = usage 

        solution_df.to_csv(self.solution_dir+'solution'+'_'+str(file_num)+'.csv', index=False)
