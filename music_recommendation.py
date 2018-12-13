# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 12:05:48 2018

@author: Bhavesh
"""
#------Datset_source_link: https://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-360K.html
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

# display results to 3 decimal points, not in scientific notation
pd.set_option('display.float_format', lambda x: '%.3f' % x)

import warnings
warnings.filterwarnings("ignore")


user_data=pd.read_table('C:/Users/Bhavesh/Pictures/Music_recommendation/lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv',
                        header=None ,nrows=2e7,
                        names=['user','musicbrainz-artist-id','artist-name','plays'])

user_profiles=pd.read_table('file:///C:/Users/Bhavesh/Pictures/Music_recommendation/lastfm-dataset-360K/usersha1-profile.tsv',
                            header=None,nrows=2e7,
                            names=['users','gender','age','country','signup'],
                            usecols=['users','country'])

if user_data['artist-name'].isnull().sum()>0:
    user_data=user_data.dropna(axis=0,subset=['artist-name'])
    
artist_plays=(user_data.groupby(by=['artist-name'])['plays'].sum().reset_index().rename(columns={'plays': 'total_artist_plays'})[['artist-name','total_artist_plays']]) 
#print(artist_plays.head())

user_data_with_artist_plays=user_data.merge(artist_plays,left_on='artist-name',right_on='artist-name',how='left')
#print(user_data_with_artist_plays.head())

popularity_threshold=40000
user_data_popular_artists=user_data_with_artist_plays.iloc[0:popularity_threshold,:]
user_data_popular_artists['users']=user_data_popular_artists['user']
user_data_popular_artists=user_data_popular_artists.drop('user',axis=1)

combined=user_data_popular_artists.merge(user_profiles,left_on='users',right_on='users',how='left')
#usa_data=combined.query('country ==\'United_States\'')
usa_data=combined[combined['country']=='United States']
#print(usa_data.head())

if not usa_data[usa_data.duplicated(['users','artist-name'])].empty:
    initial_rows=usa_data.shape[0]
    
    print('Initial_dataframe_shape{0}'.format(usa_data.shape))
    usa_data=usa_data.drop_duplicates(['users','artist-name'])
    current_rows=usa_data.shape[0]
    print('New dataframe shape{0}'.format(usa_data.shape))
    print('Removed{0} rows'.format(initial_rows-current_rows))
    print('Removed {0} rows'.format(initial_rows - current_rows))
    
    
wide_artist_data_pivot=usa_data.pivot(index='artist-name',columns='users',values='plays').fillna(0)

wide_artist_data_sparse = csr_matrix(wide_artist_data_pivot.values)

from sklearn.neighbors import NearestNeighbors
model_knn=NearestNeighbors(metric='cosine',algorithm='brute')
model_knn.fit(wide_artist_data_sparse)

query_index = np.random.choice(wide_artist_data_pivot.shape[0])
distances, indices = model_knn.kneighbors(wide_artist_data_pivot.iloc[query_index, :].reshape(1, -1), n_neighbors = 6)


for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:\n'.format(wide_artist_data_pivot.index[query_index]))
    else:
        print('{0}: {1}, with distance of {2}:'.format(i, wide_artist_data_pivot.index[indices.flatten()[i]], distances.flatten()[i]))
