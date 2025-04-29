import pandas as pd
from datetime import datetime, timedelta

#bring in CSV created from import json script
df = pd.read_csv('/Users/mason/Documents/Data Science/spotify audio json files/ms_spotify_streaming_history.csv')

#drop initial unnecessary columns
df = df.drop(columns=['offline','shuffle','audiobook_chapter_title','spotify_track_uri','incognito_mode',
                       'audiobook_uri','episode_show_name','audiobook_chapter_uri', 'conn_country',
                       'platform','episode_name','spotify_episode_uri','audiobook_title','ip_addr',
                       'offline_timestamp'])


#convert the timestamp to datetime               
df['ts'] = pd.to_datetime(df['ts'])
df['date'] = df['ts'].dt.date

df['seconds played'] = (df['ms_played'] / 1000).round()

#reorder the fields for visual purposes
df = df.reindex(columns=['date','ts','master_metadata_album_artist_name','master_metadata_track_name',
                        'master_metadata_album_album_name','reason_start','reason_end','skipped',
                        'seconds played','ms_played'])


#drop the original timestamp column now that the date is extracted
df= df.drop(columns=['ts','ms_played'])


#drop NA values from fields containing NAs (due to removed fields such as audiobook titles)
#for this analysis, I am only focused on music
df = df.dropna(subset=['master_metadata_album_artist_name','reason_start','reason_end'])

print(df.isna().sum())

#rename my fields to my liking
df = df.rename(columns={'master_metadata_album_artist_name': 'artist', 
                        'master_metadata_track_name': 'track title',
                        'master_metadata_album_album_name': 'album title'})

print(df.info())
print(df.head())

df['date'] = pd.to_datetime(df['date'])

start_date = pd.to_datetime('2023-01-01')
end_date = pd.to_datetime('2025-01-08')


filtered_df = df.loc[(df['date'] >= start_date) & (df['date'] <= end_date)]


#export to csv so I can view in excel if desired
df.to_csv('cleaned_data.csv',index=False)

filtered_df.to_csv('cleaned_data_01012023_01082025.csv', index=False)
