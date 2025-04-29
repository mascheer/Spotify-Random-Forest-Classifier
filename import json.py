import json
import csv
import glob

#path to spotify json files
json_files = glob.glob('/Users/mason/Documents/Data Science/spotify audio json files/*.json')

#initialize index for all data from path
all_spotify_data = []

#read all json files in path
for file in json_files:
    with open(file, 'r', encoding = 'utf-8') as f:
        data = json.load(f)
        all_spotify_data.extend(data)

#get unique keys
keys = set()
for entry in all_spotify_data:
    keys.update(entry.keys())
keys = list(keys)

#write data to csv
with open('ms_spotify_streaming_histor_cleaned.csv','w', newline='', encoding = 'utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=keys)
    writer.writeheader()
    writer.writerows(all_spotify_data)
   

print("CSV File Created Successfully")