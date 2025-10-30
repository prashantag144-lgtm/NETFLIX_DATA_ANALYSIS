import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_netflix_data(file_name):
   
    
    print("Loading data...")
    try:
        data = pd.read_csv(file_name, na_values=['Not Given'])
    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return

    print("Data loaded successfully.")
    
    
    print("Cleaning data...")
    
    
    duplicates_before = data.duplicated(subset=['title']).sum()
    data = data.drop_duplicates(subset=['title'], keep='first')
    data.reset_index(drop=True, inplace=True)
    print(f"Dropped {duplicates_before} duplicate titles.")

    
    data['director'] = data['director'].fillna('Unknown')
    data['country'] = data['country'].fillna('Unknown')
    print("Filled missing 'director' and 'country' values with 'Unknown'.")
    
    
    data['date_added'] = data['date_added'].str.strip()
    data['date_added'] = pd.to_datetime(data['date_added'], format='%m/%d/%Y', errors='coerce')
    
    
    data.dropna(subset=['date_added'], inplace=True)

    
    data['year_added'] = data['date_added'].dt.year.astype(int)
    data['month_added'] = data['date_added'].dt.month.astype(int)
    
    
    cleaned_file_name = 'netflix_cleaned.csv'
    data.to_csv(cleaned_file_name, index=False)
    print(f"Cleaned data saved to {cleaned_file_name}")

    
    print("Starting analysis and visualization...")
    sns.set(style="darkgrid")
    
    
    plt.figure(figsize=(8, 6))
    type_counts = data['type'].value_counts()
    sns.barplot(x=type_counts.index, y=type_counts.values, palette="muted")
    plt.title('Netflix Content Distribution: Movie vs. TV Show', fontsize=16)
    plt.xlabel('Content Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    for i, count in enumerate(type_counts):
        plt.text(i, count + 50, str(count), ha='center', fontweight='bold')
    plt.savefig('plot1_content_type_distribution.png', bbox_inches='tight')
    print("Generated plot 1: Content Type Distribution")

    
    yearly_counts = data['year_added'].value_counts().sort_index()
    yearly_counts = yearly_counts[yearly_counts.index >= 2010] # Focus on recent years
    plt.figure(figsize=(12, 7))
    sns.lineplot(x=yearly_counts.index, y=yearly_counts.values, marker='o', color='royalblue')
    plt.title('Content Added on Netflix Over Time (Yearly)', fontsize=16)
    plt.xlabel('Year Added', fontsize=12)
    plt.ylabel('Number of Titles Added', fontsize=12)
    plt.xticks(yearly_counts.index.astype(int), rotation=45)
    plt.savefig('plot2_content_added_over_time.png', bbox_inches='tight')
    print("Generated plot 2: Content Added Over Time")

    
    top_directors = data['director'].value_counts()
    top_15_directors = top_directors[top_directors.index != 'Unknown'].head(15)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_15_directors.values, y=top_15_directors.index, palette="viridis")
    plt.title('Top 15 Directors on Netflix (Excluding "Unknown")', fontsize=16)
    plt.xlabel('Number of Titles', fontsize=12)
    plt.ylabel('Director', fontsize=12)
    plt.tight_layout()
    plt.savefig('plot3_top_15_directors.png', bbox_inches='tight')
    print("Generated plot 3: Top 15 Directors")

    
    top_countries = data['country'].value_counts()
    top_15_countries = top_countries[top_countries.index != 'Unknown'].head(15)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_15_countries.values, y=top_15_countries.index, palette="plasma")
    plt.title('Top 15 Content-Producing Countries (Excluding "Unknown")', fontsize=16)
    plt.xlabel('Number of Titles', fontsize=12)
    plt.ylabel('Country', fontsize=12)
    plt.tight_layout()
    plt.savefig('plot4_top_15_countries.png', bbox_inches='tight')
    print("Generated plot 4: Top 15 Countries")

    
    genres = data['listed_in'].str.split(', ').explode()
    top_15_genres = genres.value_counts().head(15)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_15_genres.values, y=top_15_genres.index, palette="rocket")
    plt.title('Top 15 Genres on Netflix', fontsize=16)
    plt.xlabel('Number of Titles', fontsize=12)
    plt.ylabel('Genre', fontsize=12)
    plt.tight_layout()
    plt.savefig('plot5_top_15_genres.png', bbox_inches='tight')
    print("Generated plot 5: Top 15 Genres")

    
    plt.figure(figsize=(14, 7))
    rating_order = data['rating'].value_counts().index
    sns.countplot(x='rating', data=data, order=rating_order, palette="coolwarm")
    plt.title('Distribution of Content Ratings on Netflix', fontsize=16)
    plt.xlabel('Rating', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.savefig('plot6_rating_distribution.png', bbox_inches='tight')
    print("Generated plot 6: Rating Distribution")

   
    movies_df = data[data['type'] == 'Movie'].copy()
    tv_shows_df = data[data['type'] == 'TV Show'].copy()

    
    movies_df['duration'] = movies_df['duration'].str.replace(' min', '').astype(int)
    plt.figure(figsize=(12, 7))
    sns.histplot(movies_df['duration'], bins=50, kde=True, color='teal')
    plt.title('Distribution of Movie Durations (in minutes)', fontsize=16)
    plt.xlabel('Duration (minutes)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    mean_duration = movies_df['duration'].mean()
    plt.axvline(mean_duration, color='red', linestyle='--', label=f"Mean: {mean_duration:.2f} min")
    plt.legend()
    plt.savefig('plot7_movie_duration_histogram.png', bbox_inches='tight')
    print("Generated plot 7: Movie Duration Histogram")

    # Clean and analyze TV show seasons
    tv_shows_df['duration'] = tv_shows_df['duration'].str.replace(' Seasons', '').str.replace(' Season', '').astype(int)
    season_counts = tv_shows_df['duration'].value_counts().sort_index()
    plt.figure(figsize=(12, 7))
    sns.barplot(x=season_counts.index, y=season_counts.values, palette="PRGn")
    plt.title('Distribution of TV Show Seasons', fontsize=16)
    plt.xlabel('Number of Seasons', fontsize=12)
    plt.ylabel('Number of TV Shows', fontsize=12)
    plt.savefig('plot8_tv_show_seasons_distribution.png', bbox_inches='tight')
    print("Generated plot 8: TV Show Seasons Distribution")
    
    print("\n--- Analysis Complete ---")

if __name__ == "__main__":
    analyze_netflix_data('netflix1.csv')