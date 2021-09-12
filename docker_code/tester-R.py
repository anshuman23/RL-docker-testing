#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
from ytdriver.YTDriver import YTDriver, Video
from gym import spaces
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, A2C, DQN
from time import sleep
import random
from stable_baselines3.common.logger import configure
from selenium.webdriver.common.keys import Keys
import pandas as pd
from sqlalchemy import create_engine
import sys
import pickle
import json
import re
from uuid import uuid4

# In[2]:


db = dict(
    host='rostam.idav.ucdavis.edu',
    dbname='youtube',
    user='ytuser',
    passwd='GqBKuUigfQ4F0lyy'
)

def get_engine():
    return create_engine('mysql+pymysql://%s:%s@%s/%s' % (db['user'], db['passwd'], db['host'], db['dbname']))


# In[3]:


def in_range(df, key, mi, mx):
    return df[(df[key] > mi) & (df[key] < mx)]['query']

def get_puppets():
    #slant = pd.read_sql('SELECT * from `slant-scores` WHERE conservative_landmark_follows + liberal_landmark_follows > 12', con=get_engine())
    slant = pd.read_csv('popular-videos.csv')
    
    return {
        'Far Left': in_range(slant, 'slant', -1, -0.6),
        'Left': in_range(slant, 'slant', -0.6, -0.2),
        'Moderate': in_range(slant, 'slant', -0.2, 0.2),
        'Right': in_range(slant, 'slant', 0.2, 0.6),
        'Far Right': in_range(slant, 'slant', 0.6, 1)
    }


# In[4]:


puppetD = get_puppets()


# In[5]:


class YTEnvBasic(gym.Env): 
    metadata = {'render.modes': ['human']}
    
    
    def __init__(self, NUM_TOP_K=15, NUM_FEATURES=1, NUM_ACTIONS=3): 
        self.k = NUM_TOP_K
        self.f = NUM_FEATURES
        self.iter = 0
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(low = 0.0, high = 1.0, shape = (int(self.k*self.f),1), dtype = np.float32)
        self.tracing = False
        
    
    def reset(self):
        self.driver = YTDriver(verbose=True, use_virtual_display=True)
        self.iter = 1
        
        pops = ['Far Left', 'Moderate', 'Far Right']
        ################################################################
        #self.pupp_ideology = np.random.choice(pops, p=[0.45, 0.1, 0.45])
        self.pupp_ideology = 'Far Right'
        ################################################################
        seed_v_ids = puppetD[self.pupp_ideology].sample(60).to_list()
        seed_videos = ['https://www.youtube.com/watch?v=' + v_id for v_id in seed_v_ids]

        print("Starting Ideology: " + self.pupp_ideology)
        
        for vid_url in seed_videos:
            self.driver.driver.get(vid_url)
            try:
                self.driver.exposed_ad_handler()
            except:
                print("Error handler didn't work...")
            sleep(15)

        
        try:
            self.videos = self.driver.get_homepage()
        except: 
            self.driver.close()
            print("Error in beginning only!!!")
            
        scores, state = self._get_data_list(self.videos, 'homepage')
        
        self.curr_reward = self._compile_scores(scores)
        
        print("Reward Obtained: " + str(self.curr_reward))
        
        try:
            curr_state = np.array(state).reshape((int(self.k*self.f),1)).astype(np.float32)
        except:
            print("State Problem!")
            curr_state = np.ones((int(self.k*self.f),1)).astype(np.float32)
        
        curr_state = self._scale(curr_state, 0.0, 1.0)
        
        return curr_state
        
        
    def step(self, action):
        sleep(0.5)
        
        err_flag = False
        
        self.iter += 1
        
        sleep(0.5)
        
        if action >= 0:
            print("Choosing seeding action")
            
            if action == 0:
                print("Far Left seeding")
                self._seed_it_specific('Far Left')
                
            elif action == 1:
                print("Moderate seeding")
                self._seed_it_specific('Moderate')
                
            elif action == 2:
                print("Far Right seeding")
                self._seed_it_specific('Far Right')
                
            elif action == 3:
                print("Left seeding")
                self._seed_it_specific('Left')
                
            elif action == 4:
                print("Right seeding")
                self._seed_it_specific('Right')
                
        sleep(0.5)

        try:
            self.videos = self.driver.get_homepage()
        except: #Hmm, messed up, going to end episode!
            self.videos = []
            err_flag = True
        
        scores, state = self._get_data_list(self.videos, 'homepage')
        
        if scores == [] or scores is None: 
            self.curr_reward = -1.0
            self.mean = -1.0
            self.std_dev = -1.0
        else:
            self.curr_reward = self._compile_scores(scores)
            self.mean = np.mean(scores)
            self.std_dev = np.std(scores)
            
        print("Reward Obtained: " + str(self.curr_reward))
        
        if self.iter > 30: #Hmm, reward has been too high, going to reset!
            self.driver.close()
            done = True
        elif err_flag: #Hmm, in reference to an error encountered before, we have to reset!
            self.driver.close()
            done = True
        else: #All good so far, continue the current episode!
            done = False
        info = {}
        
        try:
            curr_state = np.array(state).reshape((int(self.k*self.f),1)).astype(np.float32)
        except:
            print("State Problem!")
            curr_state = np.ones((int(self.k*self.f),1)).astype(np.float32)
        
        curr_state = self._scale(curr_state, 0.0, 1.0)
        
        return curr_state, self.curr_reward, done, info
        
        
    def render(self, mode='human'):
        print('Reward Obtained: ' + str(self.curr_reward) + ' ; Mean Ideology: ' + str(self.mean) + ' ; Std-Dev Ideology: ' + str(self.std_dev))
        
           
    def _get_data_list(self, videos_list, id_str):
        scores, state = [], []
        self.total_videos, self.none_videos = 0, 0
        homepage_json = {"homepage": []}
        
        for video in videos_list:
            if len(scores) == self.k:
                break
                
            try:
                video.get_metadata()
                score = video.score
                
                self.total_videos += 1
                
                if score is None:
                    
                    ##############################
                    video.get_mean_channel_slant()
                    score = video.score
                    ##############################
                    
                    if score is None:                    
                        self.none_videos += 1
                        continue

                if True:
                    scores.append(score)
                    state.append(score)
                    ###############################################
                    homepage_json['homepage'].append(video.videoId)
                    ###############################################
                    
            except:
                continue
        
        if self.tracing:
            with open('../YT-Visualizer/data/' + self.pupp_ideology + '-' + str(self.iter) + '.json', 'w') as f:
                json.dump(homepage_json, f)
            
        return scores, state
    
    def _compile_scores(self, scores_list):
        abs_mean_val = abs(np.mean(scores_list))
        std_dev_val = np.std(scores_list)
        print(abs_mean_val, std_dev_val)
        return -(abs_mean_val) #- (0.75)*std_dev_val
    
    def _scale(self, X, x_min, x_max):
        nom = (X-X.min(axis=0))*(x_max-x_min)
        denom = X.max(axis=0) - X.min(axis=0)
        denom[denom==0] = 1
        return x_min + nom/denom
    
    def _seed_it(self):
        vid_url = random.choice(seed_videos_moderate)
        self.driver.driver.get(vid_url)
        try:
            self.driver.exposed_ad_handler()
        except:
            print("Error handler didn't work...")    
        sleep(10)
        
    def _seed_it_specific(self, ideology):
        
        while True:
            try:
                pD = get_puppets()
                break
            except:
                print("Faced issue with DB, retrying...")
        
        s_v_ids = pD[ideology].sample(50).to_list()
        s_videos_moderate = ['https://www.youtube.com/watch?v=' + v_id for v_id in s_v_ids]
        
        vid_url = random.choice(s_videos_moderate)
        self.driver.driver.get(vid_url)
        try:
            self.driver.exposed_ad_handler()
        except:
            print("Error handler didn't work...")   
        
        if self.tracing:
            vid_id_action = re.search(r'\?v=(.*)?$', vid_url).group(1)
            with open('../YT-Visualizer/data/ACTION-' + self.pupp_ideology + '-' + str(self.iter) + '.json', 'w') as f:
                json.dump({"homepage": [vid_id_action]}, f)
                
        sleep(10)
        
        
    def expose_mean_stddev(self):
        return self.mean, self.std_dev
    
    def expose_missrate_stats(self):
        return self.total_videos, self.none_videos
        
    
        


# In[6]:


env = YTEnvBasic()
#check_env(env, warn=True)


# In[7]:


m = DQN.load('SAVED_MODELS/DQN_MODEL_0-1-R', env=env)
m.load_replay_buffer("SAVED_MODELS/DQN_MODEL_0-1-R_rb")

uid_val = uuid4()

# In[9]:


#FAR RIGHT TEST

for j in range(30): 
    print("Run #" + str(j))

    #TESTING THE RL MODEL
    env = YTEnvBasic()
    obs = env.reset()
    
    rewards_list = []
    mean_list = []
    std_dev_list = []

    while True:
        action, _states = m.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        
        mean, std_dev = env.expose_mean_stddev()
        rewards_list.append(rewards)
        mean_list.append(mean)
        std_dev_list.append(std_dev)
        
        env.render()

        if done:
            break
            
    try:
        env.driver.close()
    except:
        print("No point closing, window already closed..")
        
    np.save('/output/SAVED_LISTS/REWARD_DQN_'+ '_' + str(uid_val), np.array(rewards_list))
    np.save('/output/SAVED_LISTS/MEAN_DQN_'+ '_' + str(uid_val), np.array(mean_list))
    np.save('/output/SAVED_LISTS/STDDEV_DQN_'+ '_' + str(uid_val), np.array(std_dev_list))
    
    
    
    
    ###############################################################################


    #TESTING THE RANDOM AGENT
    env = YTEnvBasic()
    obs = env.reset()
    
    rewards_list = []
    mean_list = []
    std_dev_list = []
    
    
    while True:
        action = random.randint(0, 2)
        obs, rewards, done, info = env.step(action)
        
        mean, std_dev = env.expose_mean_stddev()
        rewards_list.append(rewards)
        mean_list.append(mean)
        std_dev_list.append(std_dev)
        
        env.render()

        if done:
            break
            
    
    try:
        env.driver.close()
    except:
        print("No point closing, window already closed..")

    np.save('/output/SAVED_LISTS/REWARD_RANDOM_'+ '_' + str(uid_val), np.array(rewards_list))
    np.save('/output/SAVED_LISTS/MEAN_RANDOM_' + '_' + str(uid_val), np.array(mean_list))
    np.save('/output/SAVED_LISTS/STDDEV_RANDOM_' + '_' + str(uid_val), np.array(std_dev_list))
    
    
    
    ###############################################################################


    #TESTING THE LEFT AGENT
    env = YTEnvBasic()
    obs = env.reset()
    
    rewards_list = []
    mean_list = []
    std_dev_list = []
    
    
    while True:
        action = 0
        obs, rewards, done, info = env.step(action)
        
        mean, std_dev = env.expose_mean_stddev()
        rewards_list.append(rewards)
        mean_list.append(mean)
        std_dev_list.append(std_dev)
        
        env.render()

        if done:
            break
            
    
    try:
        env.driver.close()
    except:
        print("No point closing, window already closed..")

    np.save('/output/SAVED_LISTS/REWARD_LEFT_'+ '_' + str(uid_val), np.array(rewards_list))
    np.save('/output/SAVED_LISTS/MEAN_LEFT_' + '_' + str(uid_val), np.array(mean_list))
    np.save('/output/SAVED_LISTS/STDDEV_LEFT_' + '_' + str(uid_val), np.array(std_dev_list))
    
    
    
    
    ###############################################################################


    #TESTING THE MODERATE AGENT
    env = YTEnvBasic()
    obs = env.reset()
    
    rewards_list = []
    mean_list = []
    std_dev_list = []
    
    
    while True:
        action = 1
        obs, rewards, done, info = env.step(action)
        
        mean, std_dev = env.expose_mean_stddev()
        rewards_list.append(rewards)
        mean_list.append(mean)
        std_dev_list.append(std_dev)
        
        env.render()

        if done:
            break
            
    
    try:
        env.driver.close()
    except:
        print("No point closing, window already closed..")

    np.save('/output/SAVED_LISTS/REWARD_MODERATE_'+ '_' + str(uid_val), np.array(rewards_list))
    np.save('/output/SAVED_LISTS/MEAN_MODERATE_' + '_' + str(uid_val), np.array(mean_list))
    np.save('/output/SAVED_LISTS/STDDEV_MODERATE_' + '_' + str(uid_val), np.array(std_dev_list))


# In[ ]:




