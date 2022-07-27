import os
import pandas as pd 
import numpy as np
import textwrap

CURR_PATH = os.path.abspath('/home/chloesav/scratch/Miami_bootstrap')
NEW_PATH = '/home/chloesav/scratch/Miami_bootstrap'
os.chdir(NEW_PATH)

import manhattan_plot_util8 as man_plot
os.chdir(CURR_PATH)

import importlib
os.chdir(NEW_PATH)
importlib.reload(man_plot)
os.chdir(CURR_PATH)

BASE_FOLDER = '/home/chloesav/scratch/Miami_bootstrap'
y_group = "_miller_mh_v1"

ukbb_y, y_desc_dict, y_cat_dict = man_plot.load_phenom(y_group, BASE_FOLDER)

# Compute APOE scores predicted based on subject-specific expression of each mode of HC-DN co-variation
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

data = pd.read_csv('/home/chloesav/scratch/Miami_bootstrap/chloe_data2.csv')
data = data.set_index(['eid'],drop=False)
data.index = data.index.astype('int64')

#sex
id_m = np.where((data['Sex']==1))
id_f = np.where((data['Sex']==0))
#age
id_young = np.where(data['Age']<=55.0)
id_old = np.where(data['Age']>55.0)
#sex x age 
id_young_m = np.where((data['Age']<=55.0)&(data['Sex']==1))
id_young_f = np.where((data['Age']<=55.0)&(data['Sex']==0))
id_old_m = np.where((data['Age']>55.0)&(data['Sex']==1))
id_old_f = np.where((data['Age']>55.0)&(data['Sex']==0))
#education
id_low_ed = np.where(data['Education score']<=6.69)
id_high_ed = np.where(data['Education score']>6.69)
#all 
id_all = np.where(data['eid.1']!= None)

def features (id):
    features = data.iloc[id]
    AD = features['mf']
    features_apoe = features[['e3/e3', 'e1/e3','e2/e2','e2/e3','e3/e4','e4/e4']]
    cca_modes = features[['1',
         '2',
         '3',
         '4',
         '5',
         '6',
         '7',
         '8',
         '9',
         '10',
         '11',
         '12',
         '13',
         '14',
         '15',
         '16',
         '17',
         '18',
         '19',
         '20',
         '21',
         '22',
         '23',
         '24',
         '25',
         '26',
         '27',
         '28',
         '29',
         '30',
         '31',
         '32',
         '33',
         '34',
         '35',
         '36',
         '37',
         '38',
         '39',
         '40',
         '41',
         '42',
         '43',
         '44',
         '45',
         '46',
         '47',
         '48',
         '49',
         '50']]
    risks = features[['Fluid intelligence score (R)',
              'Loneliness',
              'Lack of social support',
              'Age',
              'Sex',
              'Education score',
              'Age completed high school education',
              'Alcohol intake frequency',
              'Alcohol consumption on a typical drinking day',
              'Current tobacco smoking frequency',
              'Past tobacco smoking frequency',
              'Attend sports club or gym',
              'Attend pub or social club',
              'Attend religious group',
              'Attend adult education class',
              'Walking for pleasure',
              'Moderate exercises',
              'Strenuous sports',
              'Sleep duration',
              'Getting up in the morning',
              'Being a morning person',
              'Sleeplessness or insomnia',
              'Heart attack',
              'Angina',
              'Stroke',
              'Hypertension',
              'Diabetes diagnosed by a doctor',
              'Hearing difficulty with background noise',
              'Hearing aid user',
              'Glaucoma',
              'Cataract',
              'Macular degeneration',
              'Miserableness',
              'Fed-up feelings',
              'Mood swings',
              'Worrier / anxious feelings',
              'Nervous feelings',
              'Sensitivity / hurt feelings',
              "Tense / 'highly strung'",
              "Suffer from 'nerves'",
              'Worry too long after embarrassment',
              'Irritability',
              'Neuroticism score',
              'Happiness',
              'Feelings of nervousness or anxiety',
              'Frequency of friend / family visits',
              'Friendships satisfaction',
              'Family relationship satisfaction',
              'Number of full siblings',
              'Living in urban areas',
              'Average household income',
              'Paid employment',
              'Retirement',
              'Looking after home or family',
              'Unable to work due to sickness or disability',
              'Unemployment',
              'Doing unpaid or voluntary work',
              'Full or part-time student',
              'Number of vehicles in household',
              'Time spent watching TV',
              'Time spent using computer',
              'Number in household',
              'Physical environment score']]
    
    risks = risks.rename(columns={'Worrier / anxious feelings': 'Worrier_anxious',
                           'Fluid intelligence score (R)': 'Fluid intelligence score',
                           'Sensitivity / hurt feelings': 'Sensitivity_Hurt feelings',
                            "Tense / 'highly strung'":'Tense_Highly strung',
                           'Frequency of friend / family visits': 'Frequency of friend or family visits'})
    APOE_scores = features['APOE_ranked']
    e2_scores = features['e2_ranked']
    e4_scores = features['e4_ranked']
    igf1 = features['IGF1']
    
    return AD, features_apoe, cca_modes, risks, APOE_scores, e2_scores, e4_scores, igf1

#for females
f_fh_AD, f_apoe_dum, f_all_modes, f_risks, f_APOE_scores, f_e2_scores, f_e4_scores, f_igf1 = features(id_f)

#for males
m_fh_AD, m_apoe_dum, m_all_modes, m_risks, m_APOE_scores, m_e2_scores, m_e4_scores, m_igf1 = features(id_m)

#APOE SCORES
#z-scoring
#males
X_scaler = StandardScaler()
X_m = X_scaler.fit_transform(m_all_modes)
X_m = X_m.T
#females
X_scaler = StandardScaler()
X_f = X_scaler.fit_transform(f_all_modes)
X_f = X_f.T

X_m = np.append(X_m, data.iloc[id_m]['eid'].values.reshape(-1,1).T, axis=0)
X_f = np.append(X_f, data.iloc[id_f]['eid'].values.reshape(-1,1).T, axis=0)

for j in tqdm(range(0,1000)):
    
    rand_m, y_m = resample(X_m.T, m_APOE_scores, replace=True, random_state = j, n_samples = 17561)
    rand_f, y_f = resample(X_f.T, f_APOE_scores, replace=True, random_state = j, n_samples = 19730)
    
    #creating dataframes with randomly selected eid for males and females
    eid_m = rand_m[:,-1].reshape(-1,1)
    eid_f = rand_f[:,-1].reshape(-1,1)
    
    males_prob = []
    females_prob = []
    
    for i in range(0,25):
#         #MALES
        x = np.array([rand_m.T[i], rand_m.T[i+25]]).T
        y = y_m
        
        log_model = LinearRegression().fit(x, y)  
        #log_model.predict(x)
        odds_mode = pd.DataFrame(log_model.predict(x), columns=[f'Probability_Mode_{i+1}'])
        odds_mode_fh = odds_mode[[f'Probability_Mode_{i+1}']]
        odds_mode_fh['eid'] = eid_m
        odds_mode_fh = odds_mode_fh.sort_values([f'Probability_Mode_{i+1}'],ascending=True) 

        min_5 = odds_mode_fh.loc[odds_mode_fh[f'Probability_Mode_{i+1}'] < np.percentile(odds_mode_fh[f'Probability_Mode_{i+1}'],5)]
        max_5 = odds_mode_fh.loc[odds_mode_fh[f'Probability_Mode_{i+1}'] > np.percentile(odds_mode_fh[f'Probability_Mode_{i+1}'],95)]
        extremes_m = pd.concat([min_5,max_5],axis=0)
        
        #z-scoring APOE scores
        X_scaler = StandardScaler()
        extremes_zscored = X_scaler.fit_transform(extremes_m[[f'Probability_Mode_{i+1}']])
        extremes_m[f'Probability_Mode_{i+1}']= extremes_zscored

        #saving each bootstrap iteration
        males_prob.append(extremes_m)

        #FEMALES
        x = np.array([rand_f.T[i], rand_f.T[i+25]]).T
        y = y_f
        
        log_model = LinearRegression().fit(x, y)  
        #log_model.predict(x)
        odds_mode = pd.DataFrame(log_model.predict(x), columns=[f'Probability_Mode_{i+1}'])
        odds_mode_fh = odds_mode[[f'Probability_Mode_{i+1}']]
        odds_mode_fh['eid'] = eid_f
        odds_mode_fh = odds_mode_fh.sort_values([f'Probability_Mode_{i+1}'],ascending=True) 

        min_5 = odds_mode_fh.loc[odds_mode_fh[f'Probability_Mode_{i+1}'] < np.percentile(odds_mode_fh[f'Probability_Mode_{i+1}'],5)]
        max_5 = odds_mode_fh.loc[odds_mode_fh[f'Probability_Mode_{i+1}'] > np.percentile(odds_mode_fh[f'Probability_Mode_{i+1}'],95)]
        extremes_f = pd.concat([min_5,max_5],axis=0)
        #z-scoring APOE scores
        X_scaler = StandardScaler()
        extremes_zscored = X_scaler.fit_transform(extremes_f[[f'Probability_Mode_{i+1}']])
        extremes_f[f'Probability_Mode_{i+1}']= extremes_zscored

        #saving each bootstrap iteration
        females_prob.append(extremes_f)
        
    for mode in range(0,25):
        df_m = males_prob[mode]
        df_m.to_csv(f'/home/chloesav/scratch/Miami_bootstrap/apoe_scores/bootstrap_{j}_mode_{mode+1}_males_apoe.csv')
        corrdf_m = man_plot.phenom_correlat(df_m, ukbb_y, y_desc_dict, y_cat_dict)
        corrdf_m.to_csv(f'/home/chloesav/scratch/Miami_bootstrap/bootstrap_{j}_mode_{mode+1}_corrdf_m.csv')
        
        df_f = females_prob[mode]
        df_f.to_csv(f'/home/chloesav/scratch/Miami_bootstrap/apoe_scores/bootstrap_{j}_mode_{mode+1}_females_apoe.csv')
        corrdf_f = man_plot.phenom_correlat(df_f, ukbb_y, y_desc_dict, y_cat_dict)
        corrdf_f.to_csv(f'/home/chloesav/scratch/Miami_bootstrap/bootstrap_{j}_mode_{mode+1}_corrdf_f.csv')
        
