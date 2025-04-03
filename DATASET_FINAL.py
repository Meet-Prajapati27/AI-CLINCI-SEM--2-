import os
import json
import time
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup
import re


class LyricsDatasetExpander:
    """Expands the existing song_dataset by adding more songs without duplicates."""

    def __init__(self, genius_access_token=None):
        """
        Initialize the dataset expander with Genius API token.

        Args:
            genius_access_token: Genius API access token
        """
        self.genius_token = genius_access_token

        if not self.genius_token:
            raise ValueError("Genius access token is required.")

        # Initialize Genius API settings
        self.genius_headers = {'Authorization': f'Bearer {self.genius_token}'}
        self.genius_base_url = "https://api.genius.com"

        # Load existing dataset
        self.existing_data = self.load_existing_dataset()
        print(f"Loaded {len(self.existing_data)} existing songs from song_dataset files")

        # Create sets for fast duplicate checking
        self.existing_titles = set()
        self.existing_ids = set()
        for song in self.existing_data:
            title_artist = f"{song.get('title', '').lower()}|{song.get('artist', '').lower()}"
            self.existing_titles.add(title_artist)

            if 'genius_id' in song:
                self.existing_ids.add(str(song['genius_id']))
            elif 'spotify_id' in song:
                self.existing_ids.add(str(song['spotify_id']))

        # Dataset storage for new songs
        self.new_dataset = []

        # Rate limiting settings
        self.genius_delay = 1.0  # Time between Genius requests

        # Track progress
        self.target_total = 500  # Target total songs (existing + new)
        self.needed_songs = max(0, self.target_total - len(self.existing_data))
        print(f"Need to add {self.needed_songs} new songs to reach target of {self.target_total}")

        # Artist and genre lists
        self.artists_by_genre = self.load_expanded_artists_by_genre()

    def load_existing_dataset(self):
        """Load existing song_dataset files (both CSV and JSON if available)."""
        data = []

        # Try loading JSON first
        if os.path.exists("song_dataset.json"):
            try:
                with open("song_dataset.json", 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"Loaded {len(data)} songs from song_dataset.json")
            except Exception as e:
                print(f"Error loading song_dataset.json: {e}")

        # If JSON failed or doesn't exist, try CSV
        if not data and os.path.exists("song_dataset.csv"):
            try:
                df = pd.read_csv("song_dataset.csv")
                data = df.to_dict('records')
                print(f"Loaded {len(data)} songs from song_dataset.csv")
            except Exception as e:
                print(f"Error loading song_dataset.csv: {e}")

        return data

    def load_expanded_artists_by_genre(self):
        """Load an expanded list of artists by genre to reach 500+ songs."""
        return {
            "Pop": [
                "Taylor Swift", "Ed Sheeran", "Ariana Grande", "Justin Bieber", "Billie Eilish",
                "Dua Lipa", "Katy Perry", "Bruno Mars", "Adele", "Shawn Mendes",
                "Lady Gaga", "The Weeknd", "Post Malone", "Selena Gomez", "Maroon 5",
                "Harry Styles", "Rihanna", "Camila Cabello", "Charlie Puth", "Imagine Dragons",
                "Justin Timberlake", "Miley Cyrus", "Sia", "Halsey", "Olivia Rodrigo",
                "Sam Smith", "Demi Lovato", "Ellie Goulding", "P!nk", "One Direction"
            ],
            "Rock": [
                "Queen", "The Beatles", "Led Zeppelin", "AC/DC", "Pink Floyd",
                "Nirvana", "Guns N' Roses", "Metallica", "Coldplay", "Linkin Park",
                "Red Hot Chili Peppers", "Green Day", "U2", "Radiohead", "Arctic Monkeys",
                "The Rolling Stones", "Twenty One Pilots", "Foo Fighters", "Aerosmith", "The Killers",
                "Muse", "Pearl Jam", "Bon Jovi", "Fall Out Boy", "The White Stripes",
                "Oasis", "The Black Keys", "Paramore", "Kings of Leon", "My Chemical Romance"
            ],
            "Hip-Hop": [
                "Drake", "Kendrick Lamar", "Eminem", "Jay-Z", "Kanye West",
                "Travis Scott", "J. Cole", "Cardi B", "Megan Thee Stallion", "Lil Nas X",
                "Post Malone", "Nicki Minaj", "Juice WRLD", "DaBaby", "Tyler, The Creator",
                "50 Cent", "Snoop Dogg", "XXXTentacion", "A$AP Rocky", "Lil Wayne",
                "Tupac Shakur", "The Notorious B.I.G.", "Chance the Rapper", "Logic", "Childish Gambino",
                "Kid Cudi", "Future", "Young Thug", "21 Savage", "Migos"
            ],
            "R&B": [
                "Beyoncé", "The Weeknd", "Alicia Keys", "John Legend", "Usher",
                "SZA", "H.E.R.", "Daniel Caesar", "Frank Ocean", "Khalid",
                "Jhené Aiko", "Chris Brown", "Toni Braxton", "Ella Mai", "Summer Walker",
                "Janelle Monáe", "Kehlani", "Anderson .Paak", "Bryson Tiller", "Miguel",
                "Mary J. Blige", "D'Angelo", "Erykah Badu", "Maxwell", "Sade",
                "Ne-Yo", "Ciara", "Tinashe", "Solange", "Teyana Taylor"
            ],
            "Country": [
                "Luke Combs", "Morgan Wallen", "Carrie Underwood", "Thomas Rhett", "Luke Bryan",
                "Kacey Musgraves", "Dan + Shay", "Miranda Lambert", "Sam Hunt", "Keith Urban",
                "Chris Stapleton", "Florida Georgia Line", "Blake Shelton", "Maren Morris", "Kenny Chesney",
                "Dierks Bentley", "Kane Brown", "Tim McGraw", "Jason Aldean", "Zac Brown Band",
                "Johnny Cash", "Dolly Parton", "George Strait", "Shania Twain", "Reba McEntire",
                "Brad Paisley", "Garth Brooks", "Alan Jackson", "Willie Nelson", "Eric Church"
            ],
            "Indie": [
                "Tame Impala", "Arctic Monkeys", "Lana Del Rey", "Vampire Weekend", "The 1975",
                "Mac DeMarco", "Beach House", "Clairo", "King Krule", "The Smiths",
                "Belle and Sebastian", "Alvvays", "Mitski", "Car Seat Headrest", "The National",
                "Wilco", "Neutral Milk Hotel", "Phoebe Bridgers", "Japanese Breakfast", "Soccer Mommy",
                "Wolf Alice", "Cocteau Twins", "Perfume Genius", "Yo La Tengo", "Big Thief",
                "Fleet Foxes", "Bon Iver", "The Strokes", "Arcade Fire", "Sufjan Stevens"
            ],
            "Electronic": [
                "Daft Punk", "Calvin Harris", "Avicii", "Skrillex", "Marshmello",
                "Zedd", "Diplo", "Deadmau5", "Martin Garrix", "The Chainsmokers",
                "David Guetta", "Kygo", "Steve Aoki", "Flume", "Major Lazer",
                "Disclosure", "Porter Robinson", "Illenium", "Tiësto", "Galantis",
                "Justice", "Aphex Twin", "Four Tet", "Jamie xx", "Bonobo",
                "Kaytranada", "Floating Points", "Jon Hopkins", "Bicep", "RÜFÜS DU SOL"
            ],
            "Jazz": [
                "Miles Davis", "John Coltrane", "Duke Ellington", "Louis Armstrong", "Thelonious Monk",
                "Charlie Parker", "Billie Holiday", "Ella Fitzgerald", "Dave Brubeck", "Nina Simone",
                "Herbie Hancock", "Charles Mingus", "Chet Baker", "Dizzy Gillespie", "Bill Evans",
                "Stan Getz", "Wes Montgomery", "Oscar Peterson", "Art Blakey", "Sonny Rollins",
                "Kamasi Washington", "Robert Glasper", "Christian Scott", "Esperanza Spalding", "GoGo Penguin",
                "Norah Jones", "Diana Krall", "Gregory Porter", "Cécile McLorin Salvant", "Snarky Puppy"
            ]
        }

    def is_duplicate(self, song_id, title, artist):
        """Check if a song is already in the existing dataset."""
        # Check by ID
        if song_id and str(song_id) in self.existing_ids:
            return True

        # Check by title and artist
        title_artist = f"{title.lower()}|{artist.lower()}"
        return title_artist in self.existing_titles

    def search_for_popular_songs(self, artist_name, genre, limit=5):
        """Search for popular songs for a given artist."""
        search_url = f"{self.genius_base_url}/search"
        params = {'q': artist_name}

        try:
            time.sleep(self.genius_delay)  # Rate limiting
            response = requests.get(search_url, headers=self.genius_headers, params=params)
            response.raise_for_status()

            data = response.json()
            hits = data.get('response', {}).get('hits', [])

            songs = []
            for hit in hits[:limit * 3]:  # Get more than needed to filter
                result = hit['result']
                # Check if this is actually by the requested artist
                if result['primary_artist']['name'].lower() == artist_name.lower():
                    # Check if it's a duplicate
                    if not self.is_duplicate(result['id'], result['title'], artist_name):
                        songs.append({
                            'genius_id': result['id'],
                            'title': result['title'],
                            'artist': artist_name,
                            'genre': genre
                        })

                        if len(songs) >= limit:
                            break

            print(f"Found {len(songs)} new songs for {artist_name}")
            return songs
        except Exception as e:
            print(f"Error searching for songs by {artist_name}: {e}")
            time.sleep(5)  # Longer delay on error
            return []

    def get_song_details(self, song_id, title, artist, genre):
        """Get detailed song information and lyrics from Genius."""
        song_url = f"{self.genius_base_url}/songs/{song_id}"

        try:
            time.sleep(self.genius_delay)  # Rate limiting
            response = requests.get(song_url, headers=self.genius_headers)
            response.raise_for_status()

            song_data = response.json()['response']['song']
            lyrics_url = song_data['url']

            # Get lyrics and analyze them
            lyrics_summary = self.extract_lyrics_metrics(lyrics_url, title, artist)

            if not lyrics_summary:
                return None

            # Create song entry with all needed data
            song_entry = {
                'spotify_id': f"genius_{song_id}",  # Placeholder since we're not using Spotify
                'title': title,
                'artist': artist,
                'genre': genre,
                'release_date': song_data.get('release_date', ''),
                'popularity': 50,  # Default popularity score
                'genius_id': song_id,
                'genius_url': lyrics_url,

                # Add lyrics metrics directly
                'lyrics_total_words': lyrics_summary['total_words'],
                'lyrics_unique_words': lyrics_summary['unique_words'],
                'lyrics_vocabulary_diversity': lyrics_summary['vocabulary_diversity'],
                'lyrics_avg_word_length': lyrics_summary['avg_word_length'],
                'lyrics_repetition_score': lyrics_summary['repetition_score'],
                'lyrics_rhyme_density': lyrics_summary['rhyme_density'],

                # Add derived features
                'feature_word_diversity': lyrics_summary['vocabulary_diversity'],
                'feature_complexity_score': lyrics_summary['complexity_score'],
                'feature_memorability_score': lyrics_summary['memorability_score'],
                'feature_learning_difficulty': lyrics_summary['learning_difficulty']
            }

            return song_entry
        except Exception as e:
            print(f"Error getting details for song {title} by {artist}: {e}")
            time.sleep(5)  # Longer delay on error
            return None

    def extract_lyrics_metrics(self, lyrics_url, title, artist):
        """Extract and analyze lyrics to compute metrics."""
        try:
            time.sleep(self.genius_delay)  # Rate limiting
            response = requests.get(lyrics_url)
            response.raise_for_status()

            html = response.text
            soup = BeautifulSoup(html, 'html.parser')

            # Find lyrics div - this might need adaptation as Genius updates their site
            lyrics_containers = soup.find_all('div', class_=lambda c: c and 'Lyrics__Container' in c)

            if not lyrics_containers:
                print(f"Could not find lyrics container for {title} by {artist}")
                return None

            # Combine lyrics from all containers
            lyrics_text = ''
            for container in lyrics_containers:
                # Remove script and other non-text elements
                for script in container.find_all('script'):
                    script.decompose()

                # Get text and clean up
                container_text = container.get_text().strip()
                container_text = re.sub(r'\[.*?\]', '', container_text)  # Remove [Verse], [Chorus] labels
                lyrics_text += container_text + '\n'

            # Clean up lyrics text - remove extra whitespace, etc.
            lyrics_text = re.sub(r'\n+', '\n', lyrics_text)
            lyrics_text = re.sub(r'\s+', ' ', lyrics_text)
            lyrics_text = lyrics_text.strip()

            # Basic analysis
            words = lyrics_text.split()
            word_count = len(words)

            if word_count < 50:  # Minimum 50 words for meaningful analysis
                print(f"Lyrics too short for {title} by {artist}: {word_count} words")
                return None

            unique_words = len(set([w.lower() for w in words]))

            # Calculate repetition score
            lines = lyrics_text.split('\n')
            unique_lines = set(lines)
            repetition_score = 1 - (len(unique_lines) / max(len(lines), 1))

            # Calculate average word length
            avg_word_length = sum(len(word) for word in words) / max(len(words), 1)

            # Count rhymes (simplistic approach)
            rhyme_count = 0
            line_endings = [line.split()[-1].lower() if line.split() else '' for line in lines]
            for i in range(len(line_endings) - 1):
                for j in range(i + 1, min(i + 4, len(line_endings))):
                    if line_endings[i] and line_endings[j] and line_endings[i][-2:] == line_endings[j][-2:]:
                        rhyme_count += 1

            rhyme_density = rhyme_count / max(len(lines), 1)

            # Calculate complexity score
            complexity_score = (
                                       avg_word_length * 0.4 +
                                       (1 - repetition_score) * 0.4 +
                                       (unique_words / max(word_count, 1)) * 0.2
                               ) * 10  # Scale to approx. 1-10

            # Estimate learning difficulty
            complexity_factors = [
                unique_words / max(word_count, 1) * 3,  # Vocabulary diversity factor
                avg_word_length / 5 * 2,  # Word length factor
                (1 - repetition_score) * 3,  # Lack of repetition factor
                min(word_count / 300, 2)  # Length factor
            ]

            learning_difficulty = sum(complexity_factors) + 2  # Base difficulty of 2
            learning_difficulty = min(max(learning_difficulty, 1), 10)  # Clamp to 1-10 scale

            # Calculate memorability (inverse of some complexity factors)
            memorability_factors = [
                repetition_score * 5,  # Repetition helps memorability
                min(rhyme_density * 10, 3),  # Rhymes help memorability
                (1 - unique_words / max(word_count, 1)) * 2  # Less vocabulary diversity is easier to memorize
            ]

            memorability_score = sum(memorability_factors) + 2  # Base memorability of 2
            memorability_score = min(max(memorability_score, 1), 10)  # Clamp to 1-10 scale

            return {
                'total_words': word_count,
                'unique_words': unique_words,
                'vocabulary_diversity': unique_words / max(word_count, 1),
                'avg_word_length': avg_word_length,
                'repetition_score': repetition_score,
                'rhyme_density': rhyme_density,
                'complexity_score': complexity_score,
                'memorability_score': memorability_score,
                'learning_difficulty': learning_difficulty
            }
        except Exception as e:
            print(f"Error extracting lyrics metrics for {title} by {artist}: {e}")
            return None

    def build_expanded_dataset(self, songs_per_artist=3, max_songs_per_genre=None):
        """Build an expanded dataset by adding new songs to existing dataset."""
        print(f"\nBuilding expanded dataset to reach {self.target_total} total songs")

        # Initialize progress tracking
        songs_added = 0
        songs_needed = self.needed_songs

        # Process genres in random order for variety
        genres = list(self.artists_by_genre.keys())
        np.random.shuffle(genres)

        for genre in genres:
            if songs_added >= songs_needed:
                break

            print(f"\n=== Processing genre: {genre} ===")

            # Calculate songs needed for this genre
            if max_songs_per_genre:
                genre_limit = min(max_songs_per_genre, songs_needed - songs_added)
            else:
                genre_limit = songs_needed - songs_added

            # Process artists in random order for variety
            artists = self.artists_by_genre[genre].copy()
            np.random.shuffle(artists)

            genre_songs_added = 0
            for artist in artists:
                if genre_songs_added >= genre_limit or songs_added >= songs_needed:
                    break

                print(f"Processing artist: {artist}")

                # Get popular songs for this artist
                songs = self.search_for_popular_songs(artist, genre, limit=songs_per_artist)

                # Process each song
                for song in songs:
                    if genre_songs_added >= genre_limit or songs_added >= songs_needed:
                        break

                    # Get detailed song info and lyrics metrics
                    song_entry = self.get_song_details(
                        song['genius_id'],
                        song['title'],
                        song['artist'],
                        song['genre']
                    )

                    if song_entry:
                        # Add to new dataset
                        self.new_dataset.append(song_entry)

                        # Add to existing titles and IDs for duplicate checking
                        title_artist = f"{song_entry['title'].lower()}|{song_entry['artist'].lower()}"
                        self.existing_titles.add(title_artist)
                        self.existing_ids.add(str(song_entry['genius_id']))

                        # Update counters
                        genre_songs_added += 1
                        songs_added += 1

                        print(
                            f"Added: {song['title']} by {song['artist']} (Difficulty: {song_entry['feature_learning_difficulty']:.1f}) - {songs_added}/{songs_needed}")

                # Take a break between artists
                time.sleep(2)

            print(f"Added {genre_songs_added} songs for genre {genre}")
            # Take a longer break between genres
            time.sleep(5)

        print(f"\nAdded {songs_added} new songs to reach a total of {len(self.existing_data) + songs_added} songs")
        return self.new_dataset

    def save_expanded_dataset(self):
        """Save the expanded dataset by appending new songs to existing files."""
        if not self.new_dataset:
            print("No new data to save")
            return False

        # Combine existing and new data
        combined_data = self.existing_data + self.new_dataset
        print(f"Combined dataset has {len(combined_data)} songs")

        # Save to JSON
        with open("song_dataset.json", 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)

        print(f"Updated song_dataset.json with {len(combined_data)} songs")

        # Save to CSV
        df = pd.DataFrame(combined_data)
        df.to_csv("song_dataset.csv", index=False)

        print(f"Updated song_dataset.csv with {len(combined_data)} songs")
        return True

    def analyze_dataset(self):
        """Provide analysis of the combined dataset."""
        combined_data = self.existing_data + self.new_dataset

        analysis = {
            "total_songs": len(combined_data),
            "existing_songs": len(self.existing_data),
            "new_songs": len(self.new_dataset),
            "genres": {},
            "difficulty_distribution": {
                "easy (1-3)": 0,
                "medium (4-7)": 0,
                "hard (8-10)": 0
            },
            "avg_difficulty": 0,
            "avg_word_count": 0,
            "avg_memorability": 0
        }

        # Calculate genre distribution
        for item in combined_data:
            genre = item.get('genre', 'Unknown')
            if genre in analysis["genres"]:
                analysis["genres"][genre] += 1
            else:
                analysis["genres"][genre] = 1

            # Calculate difficulty distribution
            difficulty = item.get('feature_learning_difficulty', 5)
            if difficulty <= 3:
                analysis["difficulty_distribution"]["easy (1-3)"] += 1
            elif difficulty <= 7:
                analysis["difficulty_distribution"]["medium (4-7)"] += 1
            else:
                analysis["difficulty_distribution"]["hard (8-10)"] += 1

            # Accumulate for averages
            analysis["avg_difficulty"] += difficulty
            analysis["avg_word_count"] += item.get('lyrics_total_words', 0)
            analysis["avg_memorability"] += item.get('feature_memorability_score', 0)

        # Calculate averages
        if len(combined_data) > 0:
            analysis["avg_difficulty"] /= len(combined_data)
            analysis["avg_word_count"] /= len(combined_data)
            analysis["avg_memorability"] /= len(combined_data)

        return analysis


# Example usage
if __name__ == "__main__":
    # Replace with your actual Genius access token
    GENIUS_ACCESS_TOKEN = "nXWe6HMDb4JaE-g9hh36K8hqwgFS8EomY8GZfGrNYaYas2uf95-pGxtfLKCKF0HX"

    try:
        # Create dataset expander
        expander = LyricsDatasetExpander(
            genius_access_token=GENIUS_ACCESS_TOKEN
        )

        # Build expanded dataset - aim for 500 total songs
        expander.build_expanded_dataset(
            songs_per_artist=3,
            max_songs_per_genre=75  # Limit to 75 songs per genre for balance
        )

        # Save the expanded dataset
        expander.save_expanded_dataset()

        # Analyze the dataset
        analysis = expander.analyze_dataset()
        print("\nFinal Dataset Analysis:")
        print(json.dumps(analysis, indent=2))

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()