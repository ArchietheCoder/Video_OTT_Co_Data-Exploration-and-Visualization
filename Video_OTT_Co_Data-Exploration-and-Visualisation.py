#!/usr/bin/env python
# coding: utf-8

# # Video OTT Company: Data Exploration and Visualisation

# In[219]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('D:\\Scaler\\Scaler\\Python\\Dataset\\netflix.csv')


# In[3]:


df.head()


# # 1. Understanding/Inspecting the data-set

# In[4]:


df.isnull().sum()


# In[5]:


df.info()


# In[6]:


df.head()


# In[7]:


df.tail()


# In[8]:


df.shape


# In[9]:


df.columns


# In[10]:


df.describe()


# In[11]:


df.isnull().sum().sort_values(ascending=False)


# In[12]:


round(df.isnull().sum()/df.shape[0]*100,2).sort_values(ascending=False)


# # 2. Data Cleaning
# 

# ## Handling Missing Values
# ### 2.a As 'Director' Column contains missing (NaaN) values, the same is replaced by "Director not specified"

# In[13]:


df['director']=df['director'].fillna('Director not specified')


# In[14]:


df.head()


# In[15]:


round(df.isnull().sum()/df.shape[0]*100,2).sort_values(ascending=False)


# ### 2.b As 'Cast' column contains missing values, the same is replaced by 'Cast not specified'

# In[16]:


df['cast']=df['cast'].fillna('Cast not specified')


# In[17]:


df.head()


# In[18]:


round(df.isnull().sum()/df.shape[0]*100,2).sort_values(ascending=False)


# ### 2.c As Rating, duration and date_added columns contain missing values and their number is negligible, those columns are being dropped

# In[19]:


df.dropna(subset=['rating', 'duration', 'date_added'], axis=0, inplace=True)


# In[20]:


df.shape


# In[21]:


round(df.isnull().sum()/df.shape[0]*100,2).sort_values(ascending=False)


# In[24]:


country_counts = (df['country'].value_counts() / df.shape[0] * 100).round(2)
country_counts.head()


# ### 2.d As country contains a good number of missing values and we can see that United States contributes 32%, we can replace the missing values with United states

# In[25]:


df['country'].fillna(df['country'].mode()[0], inplace=True)


# In[26]:


df.head()


# In[27]:


country_counts = (df['country'].value_counts() / df.shape[0] * 100).round(2)
country_counts.head()


# # All the missing values handled so far 

# In[28]:


round(df.isnull().sum()/df.shape[0]*100,2).sort_values(ascending=False)


# # 3. Data Preparation

# ## 3.1 Managing 'date_added' column

# In[70]:


df.dtypes


# ## Type of 'date_added' column needs to be converted into date type

# In[29]:


df['date_added'] = pd.to_datetime(df['date_added'])



# In[30]:


df.dtypes


# ### Additional columns named 'month', 'year' and 'week' are added which might help in data analysis 

# In[32]:


df['month'] = df['date_added'].dt.month
df['year'] = df['date_added'].dt.year


# ## 3.2 Managing 'type' column - Divided them into 'Movie' and 'TV Show' types so that we can do the 'duration' analysis separately in terms of minutes and seasons respectively 

# In[34]:


movies_df=df.loc[(df['type']=='Movie')]
movies_df.head(2)


# In[35]:


tvshow_df=df.loc[(df['type']=='TV Show')]
tvshow_df.head(2)


# ## 3.3 Managing 'duration' column

# ### Data type of 'duration' column needs to be converted into numerical type to do the required data analysis 

# In[36]:


movies_df = movies_df.copy()
movies_df['duration'] = movies_df['duration'].apply(lambda x: x.replace(' min', '') if 'min' in x else x)
movies_df.duration


# In[37]:


movies_df.info()


# In[38]:


movies_df['duration'] = movies_df['duration'].apply(lambda x: int(x) if isinstance(x, str) else x)

movies_df.describe()


# In[39]:


movies_df.dtypes


# In[40]:


tvshow_df = tvshow_df.copy()
tvshow_df['duration'] = tvshow_df['duration'].apply(lambda x: x.replace(' Season', '') if 'Season' in x else x)
tvshow_df['duration'] = tvshow_df['duration'].apply(lambda x: x.replace('s', '') if 's' in x else x)
tvshow_df.duration


# In[41]:


tvshow_df['duration'] = tvshow_df['duration'].apply(lambda x: int(x) if isinstance(x, str) else x)

tvshow_df.describe()


# In[42]:


tvshow_df.dtypes


# ## 3.4 Managing Multiple values in a single record/cell 
# ### a. 'cast' column

# In[98]:


constraint=df['cast'].apply(lambda x: str(x).split(', ')).tolist()


# In[99]:


cast_df=pd.DataFrame(constraint,index=df['title'])


# In[100]:


cast_df=cast_df.stack()
cast_df=pd.DataFrame(cast_df)
cast_df.reset_index(inplace=True)
cast_df=cast_df[['title',0]]
cast_df.columns=['title','cast']


# In[101]:


cast_df.head(10)


# 

# ## 3.4 Managing Multiple values in a single record/cell - 
# ### b. 'director' column

# In[47]:


Cons=df['director'].apply(lambda x: str(x).split(', ')).tolist()


# In[48]:


director_df=pd.DataFrame(Cons,index=df['title'])


# In[49]:


director_df=director_df.stack()
director_df=pd.DataFrame(director_df)
director_df.reset_index(inplace=True)
director_df=director_df[['title',0]]
director_df.columns=['title','director']


# In[50]:


director_df.head()


# ## 3.4 Managing Multiple values in a single record/cell - 
# ### c. 'listed_in' column (Renamed as genre_df dataframe)
# 

# In[57]:


Const=df['listed_in'].apply(lambda x: str(x).split(', ')).tolist()


# In[58]:


genre_df=pd.DataFrame(Const,index=df['title'])


# In[59]:


genre_df=genre_df.stack()
#genre_df=genre_df.stack()
genre_df=pd.DataFrame(genre_df)
genre_df.reset_index(inplace=True)
genre_df=genre_df[['title',0]]
genre_df.columns=['title','genre']


# In[60]:


genre_df.head()


# # 4. Data Analysis and Insights

# ## 4.1 Top 10 most popular Directors

# In[64]:


director_count = director_count[director_count.director != 'Director not specified']
director_count = director_count.sort_values(by=['count'], ascending = False)
director_count.head(10)


# In[122]:


import matplotlib.pyplot as plt

# Filter out 'Director not specified' rows
director_count = director_count[director_count.director != 'Director not specified']

# Sort the values in descending order
director_count = director_count.sort_values(by='count', ascending=True)

# Select the top 10 directors
top_directors = director_count.tail(10)

# Create the bar chart
plt.figure(figsize=(6, 4))
plt.barh(top_directors['director'], top_directors['count'])
plt.xlabel('Count')
plt.ylabel('Director')
plt.title('Top 10 Directors by Count')
plt.show()


# ### Insights: 
# Rajiv Chilaka followed by Jan Suter are the most popular directors amongst the viewers

# ## 4.2 Top 10 Actors/Cast based on the number of Titles

# In[109]:


# Filter out 'Cast not specified' rows
cast_df = cast_df[cast_df['cast'] != 'Cast not specified']

# Count the occurrences of each actor
cast_count = cast_df['cast'].value_counts()

# Filter out 'Cast not specified' rows
cast_count = cast_count[cast_count.index != 'Cast not specified']

# Select the top 10 actors in descending order
top_actors = cast_count.head(10)[::-1]
top_actors_table = top_actors_table.sort_values(by='Count', ascending=False)


top_actors_table



# In[123]:


# Create the bar chart
plt.figure(figsize=(6, 4))
plt.barh(top_actors.index, top_actors.values)
plt.xlabel('Count')
plt.ylabel('Actor')
plt.title('Top 10 Actors by Count')
plt.show()



# ## Insights:
# 1. Anupam Kher: Anupam Kher is the top actor with a count of 43. He has appeared in a significant number of productions within the dataset, making him the most frequent actor.
# 
# 2.Shah Rukh Khan: Shah Rukh Khan follows closely with a count of 35. As a renowned Bollywood actor, he has a substantial presence in the dataset and is widely recognized for his performances.
# 
# 3.Julie Tejwani: Julie Tejwani holds the third position with a count of 33. While not as widely known as the top two actors, she has still made notable appearances in several productions.
# 
# 4.Takahiro Sakurai: Takahiro Sakurai and Naseeruddin Shah share the fifth and sixth positions, both with a count of 32. Takahiro Sakurai is a popular voice actor, particularly in Japanese anime, while Naseeruddin Shah is a highly regarded Indian actor known for his diverse roles.
# 
# These insights provide an overview of the top actors based on their frequency of appearance in the dataset. It indicates the actors who have been featured in a significant number of productions, highlighting their prominence and popularity within the industry.

# ## 4.3 TV Shows and Movies Break-Up

# In[70]:


df.type.value_counts()


# In[130]:


import matplotlib.pyplot as plt

# Count the number of TV shows and movies
tv_show_count = tvshow_df.shape[0]
movie_count = movies_df.shape[0]

# Create the data for the pie chart
sizes = [tv_show_count, movie_count]
labels = ['TV Shows', 'Movies']
colors = ['#FF7F50', '#6495ED']
explode = (0.1, 0)

# Create the donut pie chart
plt.figure(figsize=(4, 4))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'white'})
plt.title('TV Shows and Movies Split')
plt.legend(title='Type', loc='upper right')
plt.gca().add_artist(plt.Circle((0,0),0.5,fc='white'))
plt.axis('equal')

# Display the chart
plt.show()


# ### Insight: 
# Movies are way more popular than TV Shows in Netflix

# ## 4.4 Types of ratings present in Netflix

# In[74]:


df.rating.value_counts()


# In[131]:


# Get the value counts of ratings and sort in descending order
rating_counts = df['rating'].value_counts().sort_values(ascending=True)

# Create the descending horizontal bar chart
plt.figure(figsize=(6, 4))
rating_counts.plot(kind='barh')
plt.xlabel('Count')
plt.ylabel('Rating')
plt.title('Value Counts of Ratings (Descending)')
plt.show()



# ### Insights: 
# 1. TV-MA (Mature Audience): With a count of 3205, TV-MA is the most frequently assigned rating in the dataset. TV-MA content is intended for mature audiences only and may include explicit language, graphic violence, sexual content, or other adult-oriented themes.
# 
# 2. TV-14 (Parents Strongly Cautioned): The second most common rating is TV-14, with a count of 2157. TV-14 content may contain intense violence, stronger language, or more suggestive themes. Parents are advised to exercise caution and determine if the content is suitable for viewers aged 14 and above.

# ## 4.5 Top 10 Genres/Listed_in by Count

# In[142]:


import matplotlib.pyplot as plt

# Split the 'listed_in' column and create a DataFrame for genres
genre_df = df['listed_in'].apply(lambda x: str(x).split(', ')).explode().reset_index()
genre_df.columns = ['index', 'genre']
genre_counts = genre_df['genre'].value_counts().head(10).sort_values(ascending=False)
genre_counts


# In[149]:


# Create a horizontal bar chart
genre_counts = genre_df['genre'].value_counts().head(10).sort_values(ascending=True)
plt.figure(figsize=(6, 4))
plt.barh(genre_counts.index, genre_counts.values, color='green')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.title('Top 10 Genres/Listed_in by Count')
plt.show()


# # Insights:
# 1. International Movies: The genre "International Movies" has the highest count of occurrences with 2752. This genre likely includes films from various countries and showcases the diversity of global cinema.
# 
# 2. Dramas: "Dramas" follow closely with a count of 2426. This genre encompasses a wide range of narrative-driven films that explore various emotional and character-driven themes.
# 
# 3. Comedies: "Comedies" rank third with 1674 occurrences. Comedic films are known for their light-hearted and humorous content, providing entertainment and laughter to viewers.
# 
# 4. International TV Shows: The genre "International TV Shows" has a count of 1349, indicating the popularity of TV shows from different countries and regions.
# 
# 5. Documentaries: "Documentaries" are also prevalent with 869 occurrences. This genre focuses on presenting factual information, real-life events, and non-fictional subjects, offering educational and informative content.
# 
# The above-mentioned insights provide an overview of the top genres and their respective counts, highlighting the diverse range of content available in terms of international films, dramas, comedies, TV shows, documentaries, action & adventure, and more.

# # 4.6 Region-wise trend 

# In[83]:


df.country.value_counts().head(10)


# In[132]:


import seaborn as sns
import matplotlib.pyplot as plt

# Get the value counts of countries
country_counts = df['country'].value_counts()

# Filter out missing values
country_counts = country_counts.dropna()

# Select the top 10 countries
top_countries = country_counts.head(10)

# Create the bar chart using seaborn
plt.figure(figsize=(6, 4))
sns.barplot(x=top_countries.values, y=top_countries.index, palette='viridis')
plt.xlabel('Count')
plt.ylabel('Country')
plt.title('Top 10 Countries by Count')

# Display the chart
plt.show()



# ## Insights:
# 1. United States: With a significant viewership, the United States is one of the largest audiences for Netflix. This indicates a strong presence and popularity of Netflix among viewers in the United States.
# 
# 2. India: India has emerged as a significant market for Netflix, with a substantial viewership. The platform's investment in Indian content and the growing popularity of streaming services have contributed to its success among Indian viewers.
# 
# 3. United Kingdom: The United Kingdom has a substantial Netflix viewership, reflecting its popularity among British viewers. British audiences enjoy a wide range of content available on the platform, including both local and international shows.
# 
# 4. Japan: Netflix has gained popularity among Japanese viewers, and the platform offers a diverse selection of Japanese content, including anime, dramas, and movies. This has led to a significant viewership in Japan.
# 
# 5. South Korea: South Korean viewership of Netflix has been on the rise, driven by the platform's focus on Korean dramas (K-dramas) and other South Korean content. The popularity of K-dramas globally has contributed to the increased viewership from South Korea.
# 
# 6. Canada: Canadian viewership of Netflix remains strong, with a dedicated audience enjoying a variety of content on the platform. Canadian viewers have access to both local and international shows and movies.
# 
# 7. Spain, France, Mexico, and Egypt: These countries have significant Netflix viewership, indicating a strong audience base for the platform. The availability of localized content and international offerings has contributed to the popularity of Netflix among viewers in these countries.
# 
# These insights provide an understanding of the popularity and viewership of Netflix among audiences in different countries. The specific viewership data may vary and is subject to changes over time based on the platform's growth and regional preferences.

# # 4.7 Year-wise trend

# In[119]:


year_counts = df['release_year'].value_counts().sort_values(ascending=False).head(21)
year_counts = year_counts.sort_index(ascending=True)
year_counts


# In[134]:


# Create a trend graph
plt.figure(figsize=(7, 4))
plt.plot(year_counts.index, year_counts.values, marker='o')
plt.xlabel('Year')
plt.ylabel('Count')
plt.title('Trend of Releases over Years')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# # Insights:
# 1. Increasing Trend: The count of releases shows a generally increasing trend from 2001 to 2018. The number of releases gradually rises with some fluctuations, indicating the growth of the film industry during this period.
# 
# 2. Peak in 2017: The year 2017 witnessed the highest count of releases with 1030 films. This could be attributed to various factors such as increased production, market demand, or industry trends.
# 
# 3. Slight Decline: After reaching its peak in 2017, the count of releases shows a slight decline in subsequent years. However, it is important to note that the counts in 2018, 2019, and 2020 are still relatively high compared to earlier years.
# 
# 4. Impact of Recent Years: The years 2015 to 2018 mark a significant period of growth and high activity in the film industry. These years have seen a substantial number of releases, indicating a vibrant and dynamic industry during this timeframe.
# 
# 5. Resilience during COVID-19: Despite the challenges posed by the COVID-19 pandemic, the film industry remained active in 2020 and 2021, with a considerable number of releases. This suggests the adaptability and resilience of the industry in navigating difficult circumstances.
# 
# These insights provide an overview of the trends and patterns in the count of releases over the years. They highlight the growth and fluctuations in the film industry, with specific attention to peak years and notable changes in recent times.

# # 4.8 Insights on the duration of movies

# In[164]:


# Sort the DataFrame by duration in descending order
movies_df_copy = movies_df.copy()
movies_df_copy['duration'] = movies_df_copy['duration'].apply(lambda x: str(x).replace(' min', '') if isinstance(x, str) else x)
movies_df_copy['duration'] = pd.to_numeric(movies_df_copy['duration'], errors='coerce')
movies_df_copy = movies_df_copy.sort_values(by='duration', ascending=False)
top_5_movies = movies_df_copy.head(5)[['show_id', 'title', 'duration']]
top_5_movies


# In[165]:


bottom_4_movies = movies_df_copy.tail(4)[['show_id', 'title', 'duration']]
bottom_4_movies


# In[157]:


movies_df_copy = movies_df_copy.sort_values(by='duration', ascending=True)

# Extract the top 5 movies with the highest durations
top_5_movies = movies_df_copy.head(5)

# Extract the bottom 4 movies with the lowest durations
bottom_4_movies = movies_df_copy.tail(4)

# Create a bar chart for the top 5 movies
plt.figure(figsize=(4, 3))
plt.barh(top_5_movies['title'], top_5_movies['duration'], color='blue')
plt.xlabel('Duration (minutes)')
plt.ylabel('Movie')
plt.title('Top 5 Movies with Highest Durations')
plt.show()


# In[162]:


movies_df_copy = movies_df_copy.sort_values(by='duration', ascending=True)
# Create a bar chart for the bottom 4 movies
plt.figure(figsize=(4, 3))
plt.barh(bottom_4_movies['title'], bottom_4_movies['duration'], color='red')
plt.xlabel('Duration (minutes)')
plt.ylabel('Movie')
plt.title('Bottom 4 Movies with Lowest Durations')
plt.show()


# # Insights:
# 1. "Black Mirror: Bandersnatch" has the highest duration among the top 5 movies, with a duration of 312 minutes.
# 2. "Headspace: Unwind Your Mind" is the second movie with a high duration of 273 minutes.
# whereas, 
# 
# 1. "Canvas" has the highest duration among the lowest duration movies, with a duration of 9 minutes.
# 2. "Cops and Robbers" is the second movie with a duration of 8 minutes.

# # 4.9 Insights on the duration of TV Shows

# In[175]:


tvshow_df.duration.value_counts().tail(10) 


# In[179]:


#longest_tvshows = tvshow_df.loc[(tvshow_df['duration']>=13)]
#longest_tvshows[['show_id', 'title', 'duration']]
longest_tvshows = tvshow_df.loc[tvshow_df['duration'] >= 13]
longest_tvshows = longest_tvshows.sort_values(by='duration', ascending=False)
longest_tvshows[['show_id', 'title', 'duration']]


# In[177]:


longest_tvshows.rating.value_counts()


# # Insights: 
# 1. Grey's Anatomy: Grey's Anatomy has the highest number of seasons with a duration of 17. It has been running for a significant amount of time, making it one of the longest-running shows on the list.
# 2. Supernatural: Supernatural follows closely with a duration of 15 seasons. The show gained immense popularity for its supernatural-themed plot and engaging characters.
# 3. NCIS: NCIS stands for Naval Criminal Investigative Service and has a duration of 15 seasons. It is a popular crime procedural series known for its intriguing investigations and compelling characters.
# 4. TV-14: Among the longest TV shows, four of them have a TV-14 rating. TV-14 indicates that the content may be unsuitable for children under 14 years of age and may contain intense violence, strong language, or suggestive themes. These shows likely have mature themes and are intended for a more mature audience.

# # 4.10 Month vs. Netflix Content Update

# In[207]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[208]:


df=pd.read_csv('D:\\Scaler\\Scaler\\Python\\Project\\Netflix\\netflix_copy.csv')


# In[210]:


netflix_date = df[['date_added']].dropna()
netflix_date['year'] = netflix_date['date_added'].apply(lambda x: x.split(', ')[-1])
netflix_date['month'] = netflix_date['date_added'].apply(lambda x: x.lstrip().split(' ')[0])

#month_order = ['January', 'February', 'March', 'April', 'May', 'Jun', 'July', 'August', 'September', 'October', 'November', 'December']
#new_df = netflix_date.groupby('year')['month'].value_counts().unstack().fillna(0)[month_order].T
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
new_df = netflix_date.groupby('year')['month'].value_counts().unstack().fillna(0)[month_order].T

# Plot the heatmap
plt.figure(figsize=(10, 7), dpi=200)
plt.pcolor(new_df, cmap = 'Greens', edgecolors = 'white', linewidth=2) #heatmap
plt.xticks(np.arange(0.5, len(new_df.columns), 1), new_df.columns, fontsize=7, fontfamily='Calibri')
plt.yticks(np.arange(0.5, len(new_df.index), 1), new_df.index, fontsize=7, fontfamily='Calibri')

plt.title('Month vs. Netflix Content Update', fontsize=12, fontfamily='Calibri', fontweight='bold', position=(0.20,1.0+0.02))
cbar = plt.colorbar()

cbar.ax.tick_params(labelsize=8)
cbar.ax.minorticks_on()
plt.show()


# In[217]:


# Convert 'date_added' column to datetime
df['date_added'] = pd.to_datetime(df['date_added'])

# Extract year and month from the 'date_added' column
df['year'] = df['date_added'].dt.year
df['month'] = df['date_added'].dt.month_name()

# Group by year and month and count the number of releases
monthly_releases = df.groupby(['year', 'month']).size()

# Sort the values in descending order
monthly_releases = monthly_releases.sort_values(ascending=False)

# Get the top 10 months with years
top_10_months = monthly_releases.head(10)

# Print the results
print("Top 10 Months with Year (Descending Order):")
print(top_10_months)


# # Insights:
# 1. July 2021 had the highest number of releases, with 257 new additions.
# 2. November 2019 followed closely with 255 releases, making it the second highest month.
# 3. December 2019 had 215 new releases, ranking as the third highest month.
# 4. June 2021 had 207 additions, placing it in the fourth position.
# 5. January 2020 had 205 new releases, making it the fifth highest month.

# # *********************************************************************************************

# # Conclusive Insights & Actionable Recommendations:

# Based on the above-mentioned insights, we can draw the following conclusions and provide recommendations:
# 
# ### 1. Content Availability in Different Countries:
# 
# The dataset indicates a significant viewership in the United States, India, and the United Kingdom.
# Netflix has a strong presence and popularity among viewers in these countries.
# 
# #### Recommendations:
#    1. Continue investing in content that caters to the preferences and interests of viewers in these countries.
#    2. Explore partnerships and collaborations with local production houses to create region-specific content.
#    3. Conduct market research to understand the demand for specific genres and themes in each country and tailor the content accordingly.
# 
# ### 2. Shift in Focus: TV Shows vs. Movies:
# 
# The dataset suggests that movies are more popular than TV shows on Netflix.
# 
# #### Recommendations:
#     1. Analyze the viewership trends and preferences to understand the factors contributing to the popularity of movies.
#     2. Explore strategies to enhance the TV show offerings and attract a larger audience, such as introducing compelling original series, diversifying genres, and leveraging popular actors and directors.
# 
# ### 3.Release Strategy for TV Shows:
# 
# The analysis of the top months for releases can provide insights into the best time to launch a TV show.
# 
# #### Recommendations:
#      1. Consider launching TV shows during months like July, November, and December when the viewership tends to be higher based on historical data.
#      2. Strategically plan the marketing and promotion of new TV shows to generate buzz and attract viewers during these peak months.
# 
# ### 4. Analysis of Actors and Directors:
# 
#      * Rajiv Chilaka and Jan Suter emerged as the most popular directors.
#       * Anupam Kher, Shah Rukh Khan, and Julie Tejwani are the top actors with significant presence and count of appearances.
# #### Recommendations:
#      1. Collaborate with popular directors like Rajiv Chilaka and Jan Suter to create engaging and appealing content.
#      2. Consider casting actors like Anupam Kher, Shah Rukh Khan, and Julie Tejwani in upcoming projects to leverage their popularity and attract viewers.
# 
# 
# ### 5. Content Localization and International Market:
# 
# The dataset highlights the genre "International Movies" with the highest count of occurrences.
# 
# #### Recommendations:
#       1. Expand the collection of international movies to cater to the diverse tastes and preferences of viewers from different countries.
#       2. Explore partnerships and acquisitions of international content to provide a wide range of options to global audiences.
# Invest in localization efforts, such as dubbing and subtitling, to make international content more accessible to viewers worldwide.
# 
# #### These insights and recommendations provide a starting point for decision-making and strategy development. Further analysis and data exploration can help in refining these recommendations and tailoring them to the specific goals and objectives of the business executives.
# 
# 

#    # ********************************* END OF PROJECT **********************************
