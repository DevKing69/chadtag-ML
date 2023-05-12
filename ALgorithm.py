import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Define search terms and YouTube API key
search_terms = "machine learning, data science, programming"
api_key = "your_api_key_here"

# Build YouTube API client
youtube = build('youtube', 'v3', developerKey=api_key)

# Search for top videos related to search terms
search_response = youtube.search().list(q=search_terms, type='video', part='id,snippet', maxResults=50).execute()

# Store video titles and descriptions in dataframe
videos_df = pd.DataFrame(columns=['title', 'description'])
for search_result in search_response.get('items', []):
    video_title = search_result['snippet']['title']
    video_desc = search_result['snippet']['description']
    videos_df = videos_df.append({'title': video_title, 'description': video_desc}, ignore_index=True)

# Compute cosine similarity matrix for all video descriptions
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(videos_df['description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Identify top related videos and return URLs
top_n = 5
indices = pd.Series(videos_df.index)
related_videos = []
for idx in range(len(cosine_sim)):
    similar_videos = list(enumerate(cosine_sim[idx]))
    similar_videos = sorted(similar_videos, key=lambda x: x[1], reverse=True)
    similar_videos = similar_videos[1:top_n+1]
    video_indices = [i[0] for i in similar_videos]
    video_urls = ['https://www.youtube.com/watch?v=' + search_response['items'][i]['id']['videoId'] for i in video_indices]
    related_videos.append(video_urls)

print("Top related videos:")
print(related_videos)
