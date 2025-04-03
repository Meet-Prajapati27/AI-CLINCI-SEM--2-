# Import all required libraries
import pandas as pd
import numpy as np
import os
import json
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# Add debug print function
def debug_print(message):
    """Print debug messages with timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] DEBUG: {message}")


class LyricsLearningRecommender:
    """AI-based recommendation system for lyrics learning progression with user feedback."""

    def __init__(self, dataset_path='song_dataset.csv', model_path='lyrics_learning_model.pkl',
                 feedback_path='user_feedback.csv'):
        """Initialize the recommendation system."""
        debug_print("Initializing recommender")
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.feedback_path = feedback_path
        self.dataset = None
        self.model_data = None
        self.scaler = None
        self.features = None
        self.feedback_data = None
        self.rf_model = None  # Random Forest model for learning from feedback

    def load_dataset(self):
        """Load the song dataset from CSV."""
        debug_print(f"Attempting to load dataset from {self.dataset_path}")
        try:
            # First check if file exists
            if not os.path.exists(self.dataset_path):
                print(f"Error: Dataset file not found at {self.dataset_path}")
                return False

            df = pd.read_csv(self.dataset_path)
            print(f"Successfully loaded dataset with {len(df)} songs")
            self.dataset = df
            return True
        except FileNotFoundError:
            print(f"Error: Could not find dataset at {self.dataset_path}")
            return False
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False

    def load_feedback(self):
        """Load user feedback if available."""
        debug_print("Loading feedback data")
        if not os.path.exists(self.feedback_path):
            # Initialize empty feedback dataframe
            self.feedback_data = pd.DataFrame(columns=[
                'user_id', 'song_id', 'learning_time_minutes',
                'difficulty_rating', 'memorability_rating', 'timestamp'
            ])
            print("No existing feedback data. Starting fresh.")
            return True

        try:
            self.feedback_data = pd.read_csv(self.feedback_path)
            print(f"Loaded {len(self.feedback_data)} feedback entries")
            return True
        except Exception as e:
            print(f"Error loading feedback data: {e}")
            self.feedback_data = pd.DataFrame(columns=[
                'user_id', 'song_id', 'learning_time_minutes',
                'difficulty_rating', 'memorability_rating', 'timestamp'
            ])
            return False

    def save_feedback(self):
        """Save user feedback to CSV."""
        if self.feedback_data is not None:
            try:
                self.feedback_data.to_csv(self.feedback_path, index=False)
                print(f"Saved {len(self.feedback_data)} feedback entries")
                return True
            except Exception as e:
                print(f"Error saving feedback: {e}")
                return False
        return False

    def add_feedback(self, user_id, song_id, learning_time, difficulty, memorability):
        """Add user feedback for a song learning experience."""
        debug_print(f"Adding feedback for user {user_id}, song {song_id}")
        if self.feedback_data is None:
            self.load_feedback()

        # Create new feedback entry
        new_feedback = {
            'user_id': user_id,
            'song_id': song_id,
            'learning_time_minutes': learning_time,
            'difficulty_rating': difficulty,
            'memorability_rating': memorability,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Add to feedback dataframe
        self.feedback_data = pd.concat([self.feedback_data, pd.DataFrame([new_feedback])], ignore_index=True)

        # Save updated feedback
        self.save_feedback()

        print(f"Added feedback for song ID {song_id}")

        # Update model if we have enough feedback
        if len(self.feedback_data) % 5 == 0:  # Update every 5 feedback entries
            self.update_model_with_feedback()

        return True

    def extract_learning_features(self):
        """Extract learning-relevant features from the dataset."""
        debug_print("Extracting learning features")
        if self.dataset is None:
            print("Error: No dataset loaded")
            return False

        # Make a copy to avoid modifying the original
        df = self.dataset.copy()

        # List to store the features we'll use
        learning_features = []

        # Check for each potential feature in our dataset
        feature_candidates = [
            'lyrics_total_words', 'lyrics_unique_words', 'lyrics_vocabulary_diversity',
            'lyrics_avg_word_length', 'lyrics_repetition_score', 'lyrics_rhyme_density',
            'feature_word_diversity', 'feature_complexity_score',
            'feature_memorability_score', 'feature_learning_difficulty'
        ]

        # Add only features that exist in our dataset
        for feature in feature_candidates:
            if feature in df.columns:
                learning_features.append(feature)

        if not learning_features:
            print("Warning: No learning-specific features found in dataset")
            # Use any numeric features as fallback
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                print(f"Using {len(numeric_cols)} numeric features instead")
                learning_features = numeric_cols

        print(f"Using features: {', '.join(learning_features)}")
        self.features = learning_features
        return True

    def train_model(self):
        """Train the recommendation model using the dataset."""
        debug_print("Training recommendation model")
        if self.dataset is None:
            print("Error: No dataset loaded")
            return False

        if not self.features:
            print("Error: No features extracted")
            return False

        print("Training recommendation model...")

        # Clean the dataset by filling missing values
        for feature in self.features:
            if self.dataset[feature].isnull().any():
                print(f"Filling missing values in feature: {feature}")
                # For each feature, fill NaN with the mean of that feature
                self.dataset[feature] = self.dataset[feature].fillna(self.dataset[feature].mean())

        # Prepare feature matrix
        X = self.dataset[self.features].values

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Create similarity matrix
        similarity_matrix = cosine_similarity(X_scaled)

        # Store song indices for lookups
        song_indices = {idx: i for i, idx in enumerate(self.dataset.index)}

        # Store model data
        self.model_data = {
            'similarity_matrix': similarity_matrix,
            'song_indices': song_indices,
            'feature_names': self.features
        }

        # If we have feedback data, train a Random Forest model
        if self.feedback_data is not None and len(self.feedback_data) >= 5:
            self.train_feedback_model()

        print(f"Model trained successfully with {len(self.features)} features")
        return True

    def train_feedback_model(self):
        """Train a Random Forest model using user feedback."""
        debug_print("Training feedback model")
        if self.feedback_data is None or len(self.feedback_data) < 5:
            print("Not enough feedback data to train model")
            return False

        if self.dataset is None:
            print("No dataset available")
            return False

        print("Training feedback-based model...")

        # Merge feedback with song features
        merged_data = []

        for _, feedback in self.feedback_data.iterrows():
            song_id = feedback['song_id']
            if song_id in self.dataset.index:
                # Get song features
                song_features = self.dataset.loc[song_id][self.features].to_dict()

                # Add feedback data
                song_features.update({
                    'user_id': feedback['user_id'],
                    'learning_time': feedback['learning_time_minutes'],
                    'user_difficulty': feedback['difficulty_rating'],
                    'user_memorability': feedback['memorability_rating']
                })

                merged_data.append(song_features)

        if not merged_data:
            print("No valid feedback data to train model")
            return False

        # Create DataFrame from merged data
        train_df = pd.DataFrame(merged_data)

        # Prepare features and targets
        X = train_df[self.features].values
        y_difficulty = train_df['user_difficulty'].values

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Train Random Forest model for difficulty prediction
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_model.fit(X_scaled, y_difficulty)

        print(f"Feedback model trained with {len(train_df)} examples")
        return True

    def update_model_with_feedback(self):
        """Update the model based on accumulated feedback."""
        debug_print("Updating model with feedback")
        if self.feedback_data is None or len(self.feedback_data) < 5:
            print("Not enough feedback data to update model")
            return False

        # Train the feedback model
        success = self.train_feedback_model()

        if success and self.rf_model is not None:
            # Get feature importances
            importances = self.rf_model.feature_importances_

            # Create a dictionary of feature importances
            feature_importance = dict(zip(self.features, importances))

            print("Updated feature importances based on feedback:")
            for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
                print(f"  {feature}: {importance:.4f}")

            # Store feature importances in model data
            self.model_data['feature_importances'] = feature_importance

            # Save the updated model
            self.save_model()

            return True

        return False

    def save_model(self):
        """Save the trained model to a file."""
        debug_print(f"Saving model to {self.model_path}")
        if self.model_data is None or self.scaler is None or self.features is None:
            print("Error: No trained model to save")
            return False

        model_package = {
            'model_data': self.model_data,
            'scaler': self.scaler,
            'features': self.features,
            'rf_model': self.rf_model
        }

        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_package, f)

            print(f"Model saved to {self.model_path}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    def load_model(self):
        """Load a trained model from a file."""
        debug_print(f"Loading model from {self.model_path}")
        try:
            with open(self.model_path, 'rb') as f:
                model_package = pickle.load(f)

            self.model_data = model_package['model_data']
            self.scaler = model_package['scaler']
            self.features = model_package['features']

            # Load Random Forest model if available
            if 'rf_model' in model_package:
                self.rf_model = model_package['rf_model']
                print("Loaded feedback-based model")

            print(f"Model loaded from {self.model_path}")
            return True
        except FileNotFoundError:
            print(f"No existing model found at {self.model_path}")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def setup(self):
        """Set up the recommendation system."""
        debug_print("Setting up recommendation system")
        # Load the dataset
        if not self.load_dataset():
            return False

        # Extract features
        if not self.extract_learning_features():
            return False

        # Load feedback data
        self.load_feedback()

        # Check for existing model
        if os.path.exists(self.model_path):
            if self.load_model():
                return True

        # Train new model if needed
        if self.train_model():
            self.save_model()
            return True

        return False

    def get_recommendations(self, current_song_id, difficulty_increment=0.5, top_n=5, user_id=None):
        """
        Get song recommendations based on the current song.

        Args:
            current_song_id: ID or title of current song
            difficulty_increment: How much to increase difficulty (0-1)
            top_n: Number of recommendations to return
            user_id: Optional user ID for personalized recommendations

        Returns:
            DataFrame with recommended songs
        """
        debug_print(f"Getting recommendations for song {current_song_id}")
        if self.model_data is None or self.dataset is None:
            print("Error: Model not set up. Call setup() first.")
            return None

        # Get current song
        current_song = None
        current_idx = None

        # Try to find by ID/index
        if isinstance(current_song_id, int):
            if current_song_id in self.dataset.index:
                current_song = self.dataset.loc[current_song_id]
                current_idx = current_song_id
            else:
                # Try as a row number
                try:
                    current_song = self.dataset.iloc[current_song_id]
                    current_idx = current_song.name
                except IndexError:
                    pass

        # If not found, try as title
        if current_song is None:
            matches = self.dataset[self.dataset['title'].str.contains(str(current_song_id), case=False)]
            if len(matches) > 0:
                current_song = matches.iloc[0]
                current_idx = current_song.name

        # If still not found, return error
        if current_song is None:
            print(f"Could not find song matching '{current_song_id}'")
            return None

        # Display the current song title
        print(f"\nGetting recommendations based on: {current_song['title']} by {current_song['artist']}")
        if 'feature_learning_difficulty' in current_song:
            print(f"Current song difficulty: {current_song['feature_learning_difficulty']:.1f}/10")

        # Rest of the function remains the same...

        # Get index in similarity matrix
        if current_idx not in self.model_data['song_indices']:
            print(f"Current song index {current_idx} not found in model")
            return None

        matrix_idx = self.model_data['song_indices'][current_idx]

        # Get similarities from matrix
        similarities = self.model_data['similarity_matrix'][matrix_idx]

        # Get difficulty column if available
        difficulty_col = None
        for col in ['feature_learning_difficulty', 'feature_complexity_score']:
            if col in self.dataset.columns:
                difficulty_col = col
                break

        # Current song difficulty
        current_difficulty = float(current_song[difficulty_col]) if difficulty_col else 5.0

        # Target difficulty range
        min_difficulty = current_difficulty + (difficulty_increment * 0.5)
        max_difficulty = current_difficulty + (difficulty_increment * 1.5)

        # Create a dataframe with all songs and their similarity scores
        sim_df = pd.DataFrame({
            'song_id': self.dataset.index,
            'title': self.dataset['title'],
            'artist': self.dataset['artist'],
            'similarity': similarities,
            'genre': self.dataset['genre'] if 'genre' in self.dataset.columns else 'Unknown'
        })

        # Add difficulty if available
        if difficulty_col:
            sim_df['difficulty'] = self.dataset[difficulty_col].values

        # Remove the current song
        sim_df = sim_df[sim_df['song_id'] != current_idx]

        # Apply user personalization if we have feedback data and user_id
        if user_id is not None and self.feedback_data is not None and len(self.feedback_data) > 0:
            user_feedback = self.feedback_data[self.feedback_data['user_id'] == user_id]

            if len(user_feedback) > 0:
                print(
                    f"Personalizing recommendations for user {user_id} based on {len(user_feedback)} feedback entries")

                # Get user's average difficulty rating
                avg_difficulty = user_feedback['difficulty_rating'].mean()

                # Adjust target difficulty based on user feedback
                # If user tends to rate songs as more difficult, lower the target
                difficulty_adjustment = avg_difficulty / 5.0  # Normalize around 5
                adjusted_min = min_difficulty / difficulty_adjustment
                adjusted_max = max_difficulty / difficulty_adjustment

                print(
                    f"Adjusted difficulty range: {adjusted_min:.1f}-{adjusted_max:.1f} (was {min_difficulty:.1f}-{max_difficulty:.1f})")

                min_difficulty = adjusted_min
                max_difficulty = adjusted_max

                # Check if user has previous feedback for songs in our recommendations
                learned_songs = set(user_feedback['song_id'].values)

                # Boost songs similar to those the user found easier
                easy_songs = user_feedback[user_feedback['difficulty_rating'] < avg_difficulty]
                if len(easy_songs) > 0:
                    for _, feedback in easy_songs.iterrows():
                        easy_song_id = feedback['song_id']
                        if easy_song_id in self.model_data['song_indices']:
                            easy_idx = self.model_data['song_indices'][easy_song_id]
                            easy_similarities = self.model_data['similarity_matrix'][easy_idx]

                            # Boost similarity for songs similar to easy songs
                            boost_factor = (avg_difficulty - feedback['difficulty_rating']) / avg_difficulty
                            sim_df['similarity'] += 0.2 * boost_factor * easy_similarities[sim_df['song_id'].map(
                                lambda x: self.model_data['song_indices'].get(x, 0)
                            )]

        # Filter by difficulty range if possible
        if difficulty_col:
            in_range = sim_df[(sim_df['difficulty'] >= min_difficulty) &
                              (sim_df['difficulty'] <= max_difficulty)]

            if len(in_range) >= top_n:
                # If we have enough songs in the target range, sort by similarity
                recommendations = in_range.sort_values('similarity', ascending=False).head(top_n)
            else:
                # Otherwise use all songs but with a penalty for being outside the range
                sim_df['difficulty_distance'] = sim_df['difficulty'].apply(
                    lambda x: min(abs(x - min_difficulty),
                                  abs(x - max_difficulty)) if x < min_difficulty or x > max_difficulty else 0
                )

                # Calculate combined score (70% similarity, 30% difficulty matching)
                sim_df['combined_score'] = (0.7 * sim_df['similarity']) - (0.3 * sim_df['difficulty_distance'])
                recommendations = sim_df.sort_values('combined_score', ascending=False).head(top_n)
        else:
            # If no difficulty info, just use similarity
            recommendations = sim_df.sort_values('similarity', ascending=False).head(top_n)

        # Add explanation for each recommendation
        recommendations['explanation'] = recommendations.apply(
            lambda x: self._explain_recommendation(current_song, self.dataset.loc[x['song_id']]),
            axis=1
        )

        # Add user-specific explanations if available
        if user_id is not None and self.feedback_data is not None:
            user_feedback = self.feedback_data[self.feedback_data['user_id'] == user_id]
            if len(user_feedback) > 0:
                recommendations['user_note'] = recommendations.apply(
                    lambda x: self._get_user_specific_note(x['song_id'], user_id, user_feedback),
                    axis=1
                )

        # Select columns for output
        output_cols = ['song_id', 'title', 'artist', 'similarity', 'explanation']
        if 'genre' in recommendations.columns:
            output_cols.insert(3, 'genre')
        if 'difficulty' in recommendations.columns:
            output_cols.insert(4, 'difficulty')
        if 'user_note' in recommendations.columns:
            output_cols.append('user_note')

        return recommendations[output_cols]

    def _get_user_specific_note(self, song_id, user_id, user_feedback):
        """Generate user-specific notes for recommendations."""
        # Check if user has learned this song
        if song_id in user_feedback['song_id'].values:
            song_feedback = user_feedback[user_feedback['song_id'] == song_id].iloc[0]
            return f"You previously learned this song (rated: {song_feedback['difficulty_rating']}/10 difficulty)"

        # Check if user has learned songs by the same artist
        song_artist = self.dataset.loc[song_id, 'artist']
        artist_songs = self.dataset[self.dataset['artist'] == song_artist]
        artist_song_ids = set(artist_songs.index)

        learned_artist_songs = user_feedback[user_feedback['song_id'].isin(artist_song_ids)]
        if len(learned_artist_songs) > 0:
            avg_difficulty = learned_artist_songs['difficulty_rating'].mean()
            return f"You've learned {len(learned_artist_songs)} song(s) by this artist (avg. {avg_difficulty:.1f}/10 difficulty)"

        # Check if user has learned songs in this genre
        if 'genre' in self.dataset.columns:
            song_genre = self.dataset.loc[song_id, 'genre']
            genre_songs = self.dataset[self.dataset['genre'] == song_genre]
            genre_song_ids = set(genre_songs.index)

            learned_genre_songs = user_feedback[user_feedback['song_id'].isin(genre_song_ids)]
            if len(learned_genre_songs) > 0:
                avg_difficulty = learned_genre_songs['difficulty_rating'].mean()
                return f"You've learned {len(learned_genre_songs)} {song_genre} song(s) (avg. {avg_difficulty:.1f}/10 difficulty)"

        return ""

    def _explain_recommendation(self, current_song, recommended_song):
        """Generate an explanation for a recommendation."""
        explanations = []

        # Compare difficulty
        difficulty_col = None
        for col in ['feature_learning_difficulty', 'feature_complexity_score']:
            if col in current_song.index and col in recommended_song.index:
                difficulty_col = col
                break

        if difficulty_col:
            diff_change = float(recommended_song[difficulty_col]) - float(current_song[difficulty_col])
            if diff_change > 0.5:
                explanations.append(f"Increases difficulty by {diff_change:.1f}")
            elif diff_change < -0.5:
                explanations.append(f"Decreases difficulty by {abs(diff_change):.1f}")
            else:
                explanations.append("Similar difficulty level")

        # Compare vocabulary
        if 'feature_word_diversity' in self.features:
            if float(recommended_song['feature_word_diversity']) > float(current_song['feature_word_diversity']) * 1.2:
                explanations.append("More diverse vocabulary")
            elif float(recommended_song['feature_word_diversity']) < float(
                    current_song['feature_word_diversity']) * 0.8:
                explanations.append("Simpler vocabulary")

        # Compare repetition
        if 'lyrics_repetition_score' in self.features:
            if float(recommended_song['lyrics_repetition_score']) > float(
                    current_song['lyrics_repetition_score']) * 1.2:
                explanations.append("More repetitive patterns")
            elif float(recommended_song['lyrics_repetition_score']) < float(
                    current_song['lyrics_repetition_score']) * 0.8:
                explanations.append("Fewer repetitive elements")

        # Compare memorability
        if 'feature_memorability_score' in self.features:
            if float(recommended_song['feature_memorability_score']) > float(
                    current_song['feature_memorability_score']) * 1.2:
                explanations.append("More memorable")
            elif float(recommended_song['feature_memorability_score']) < float(
                    current_song['feature_memorability_score']) * 0.8:
                explanations.append("Less memorable")

        # Compare genre
        if 'genre' in current_song.index and 'genre' in recommended_song.index:
            if recommended_song['genre'] == current_song['genre']:
                explanations.append(f"Same genre ({recommended_song['genre']})")
            else:
                explanations.append(f"Different genre ({recommended_song['genre']})")

        return "; ".join(explanations)

    def get_user_history(self, user_id):
        """Get learning history for a specific user."""
        debug_print(f"Getting history for user {user_id}")
        if self.feedback_data is None:
            self.load_feedback()

        if self.feedback_data is None or len(self.feedback_data) == 0:
            print(f"No feedback data available for user {user_id}")
            return None

        user_data = self.feedback_data[self.feedback_data['user_id'] == user_id]

        if len(user_data) == 0:
            print(f"No history for user {user_id}")
            return None

        # Join with song data
        history = []
        for _, feedback in user_data.iterrows():
            song_id = feedback['song_id']
            if song_id in self.dataset.index:
                song = self.dataset.loc[song_id]
                history.append({
                    'song_id': song_id,
                    'title': song['title'],
                    'artist': song['artist'],
                    'genre': song['genre'] if 'genre' in song else 'Unknown',
                    'learning_time': feedback['learning_time_minutes'],
                    'difficulty_rating': feedback['difficulty_rating'],
                    'memorability_rating': feedback['memorability_rating'],
                    'timestamp': feedback['timestamp']
                })

        history_df = pd.DataFrame(history)
        return history_df.sort_values('timestamp', ascending=False)

    def get_learning_stats(self, user_id):
        """Get learning statistics for a user."""
        debug_print(f"Getting learning stats for user {user_id}")
        history = self.get_user_history(user_id)

        if history is None or len(history) == 0:
            return None

        stats = {
            'total_songs_learned': len(history),
            'total_learning_time': history['learning_time'].sum(),
            'avg_learning_time': history['learning_time'].mean(),
            'avg_difficulty_rating': history['difficulty_rating'].mean(),
            'avg_memorability_rating': history['memorability_rating'].mean(),
            'genres_learned': history['genre'].value_counts().to_dict(),
            'learning_trend': {}
        }

        # Calculate learning trend (if we have enough data)
        if len(history) >= 3:
            # Sort by timestamp
            history = history.sort_values('timestamp')

            # Split into thirds to see trends
            splits = np.array_split(history, 3)

            stats['learning_trend'] = {
                'early': {
                    'avg_time': splits[0]['learning_time'].mean(),
                    'avg_difficulty': splits[0]['difficulty_rating'].mean()
                },
                'middle': {
                    'avg_time': splits[1]['learning_time'].mean(),
                    'avg_difficulty': splits[1]['difficulty_rating'].mean()
                },
                'recent': {
                    'avg_time': splits[2]['learning_time'].mean(),
                    'avg_difficulty': splits[2]['difficulty_rating'].mean()
                }
            }

            # Calculate improvement percentages
            time_improvement = (stats['learning_trend']['early']['avg_time'] -
                                stats['learning_trend']['recent']['avg_time']) / stats['learning_trend']['early'][
                                   'avg_time'] * 100

            difficulty_increase = (stats['learning_trend']['recent']['avg_difficulty'] -
                                   stats['learning_trend']['early']['avg_difficulty'])

            stats['learning_trend']['time_improvement_percent'] = time_improvement
            stats['learning_trend']['difficulty_increase'] = difficulty_increase

        return stats

    def run_interactive_cli(self):
        """Run an interactive command-line interface."""
        debug_print("Starting interactive CLI")
        print("\n=== Lyrics Learning Recommendation System with Feedback ===\n")

        if not self.setup():
            print("Failed to set up the recommendation system. Exiting.")
            return

        # Initialize user
        user_id = input("Enter your user ID or name: ")
        print(f"\nWelcome, {user_id}!")

        print("\nSystem ready! Using dataset with", len(self.dataset), "songs.")

        while True:
            print("\nOptions:")
            print("1. List available songs")
            print("2. Get song recommendations")
            print("3. Add learning feedback")
            print("4. View your learning history")
            print("5. View your learning stats")
            print("6. Exit")

            choice = input("\nEnter your choice (1-6): ")

            if choice == '1':
                # List songs
                print("\nAvailable Songs:")
                for i, (idx, row) in enumerate(self.dataset[['title', 'artist']].iterrows()):
                    genre = self.dataset.loc[idx, 'genre'] if 'genre' in self.dataset.columns else ''
                    print(f"{idx}. {row['title']} by {row['artist']} {f'({genre})' if genre else ''}")
                    if i > 0 and i % 20 == 19:  # Show 20 songs at a time
                        if input("\nShow more? (y/n): ").lower() != 'y':
                            break


            elif choice == '2':

                # Get recommendations

                song_input = input("\nEnter the ID or title of the song you're currently learning: ")

                # Try to convert to int for ID

                try:

                    song_input = int(song_input)

                except ValueError:

                    pass  # Keep as string for title search

                difficulty = input("How much would you like to increase difficulty (0.1-1.0)? [default: 0.5]: ")

                try:

                    difficulty = float(difficulty)

                    if difficulty < 0.1:

                        difficulty = 0.1

                    elif difficulty > 1.0:

                        difficulty = 1.0

                except ValueError:

                    difficulty = 0.5

                use_personalization = input(
                    "Use personalized recommendations based on your history? (y/n) [default: y]: ").lower() != 'n'

                # Get personalized recommendations

                recommendations = self.get_recommendations(

                    song_input,

                    difficulty_increment=difficulty,

                    user_id=user_id if use_personalization else None

                )

                if recommendations is not None and not recommendations.empty:
                    print("\nRecommended songs to learn next:")
                    for i, (_, row) in enumerate(recommendations.iterrows()):
                        print(f"{i + 1}. {row['title']} by {row['artist']}", end='')
                        if 'genre' in row:
                            print(f" ({row['genre']})", end='')
                        print()

                        if 'difficulty' in row:
                            print(f"   Difficulty: {row['difficulty']:.1f}", end='')
                        print(f", Similarity: {row['similarity']:.2f}")
                        print(f"   Why: {row['explanation']}")

                        if 'user_note' in row and row['user_note']:
                            print(f"   Note: {row['user_note']}")

                        print()
                else:
                    print("No recommendations found. Try a different song or difficulty level.")

            elif choice == '3':
                # Add learning feedback
                print("\nAdd Feedback for a Song You've Learned:")

                # Get song ID
                song_input = input("Enter the ID or title of the song you learned: ")

                # Try to convert to int for ID
                try:
                    song_input = int(song_input)
                except ValueError:
                    pass  # Keep as string for title search

                # Find the song
                song_id = None
                song_title = None

                if isinstance(song_input, int) and song_input in self.dataset.index:
                    song_id = song_input
                    song_title = self.dataset.loc[song_input, 'title']
                else:
                    matches = self.dataset[self.dataset['title'].str.contains(str(song_input), case=False)]
                    if len(matches) > 0:
                        song_id = matches.iloc[0].name
                        song_title = matches.iloc[0]['title']

                if song_id is None:
                    print(f"Could not find song matching '{song_input}'")
                    continue

                print(f"\nAdding feedback for: {song_title}")

                # Get learning time
                while True:
                    try:
                        learning_time = float(input("How many minutes did it take you to learn this song? "))
                        if learning_time <= 0:
                            print("Please enter a positive number.")
                            continue
                        break
                    except ValueError:
                        print("Please enter a valid number.")

                # Get difficulty rating
                while True:
                    try:
                        difficulty = float(input("Rate the difficulty (1-10 where 10 is hardest): "))
                        if difficulty < 1 or difficulty > 10:
                            print("Please enter a number between 1 and 10.")
                            continue
                        break
                    except ValueError:
                        print("Please enter a valid number.")

                # Get memorability rating
                while True:
                    try:
                        memorability = float(
                            input("Rate how memorable the song was (1-10 where 10 is most memorable): "))
                        if memorability < 1 or memorability > 10:
                            print("Please enter a number between 1 and 10.")
                            continue
                        break
                    except ValueError:
                        print("Please enter a valid number.")

                # Add feedback
                self.add_feedback(user_id, song_id, learning_time, difficulty, memorability)
                print(f"Feedback added for {song_title}. Thank you!")

            elif choice == '4':
                # View your learning history
                history = self.get_user_history(user_id)

                if history is None or len(history) == 0:
                    print("\nYou haven't provided feedback on any songs yet.")
                    continue

                print(f"\nYour Learning History ({len(history)} songs):")
                print("-" * 80)
                print(f"{'Title':<30} {'Artist':<20} {'Difficulty':<10} {'Time (min)':<10} {'Date'}")
                print("-" * 80)

                for _, song in history.iterrows():
                    print(
                        f"{song['title']:<30} {song['artist']:<20} {song['difficulty_rating']:<10.1f} {song['learning_time']:<10.0f} {song['timestamp'][:10]}")

                print("-" * 80)

            elif choice == '5':
                # View your learning stats
                stats = self.get_learning_stats(user_id)

                if stats is None:
                    print("\nNot enough learning data to generate statistics.")
                    continue

                print("\n=== Your Learning Statistics ===")
                print(f"Total songs learned: {stats['total_songs_learned']}")
                print(f"Total learning time: {stats['total_learning_time']:.1f} minutes")
                print(f"Average learning time per song: {stats['avg_learning_time']:.1f} minutes")
                print(f"Average difficulty rating: {stats['avg_difficulty_rating']:.1f}/10")
                print(f"Average memorability rating: {stats['avg_memorability_rating']:.1f}/10")

                print("\nGenres learned:")
                for genre, count in stats['genres_learned'].items():
                    print(f"  {genre}: {count} songs")

                if 'learning_trend' in stats and 'time_improvement_percent' in stats['learning_trend']:
                    improvement = stats['learning_trend']['time_improvement_percent']
                    if improvement > 0:
                        print(f"\nLearning improvement: {improvement:.1f}% faster compared to your earlier songs")
                    elif improvement < 0:
                        print(f"\nLearning trend: {abs(improvement):.1f}% slower compared to your earlier songs")

                    difficulty_change = stats['learning_trend']['difficulty_increase']
                    if difficulty_change > 0.5:
                        print(f"You're tackling more difficult songs: +{difficulty_change:.1f} points in difficulty")
                    elif difficulty_change < -0.5:
                        print(f"You're choosing easier songs: {difficulty_change:.1f} points in difficulty")

            elif choice == '6':
                print("\nThanks for using the Lyrics Learning Recommendation System!")
                break

            else:
                print("Invalid choice. Please try again.")


# Outside the class - Main execution function
def main():
    """Main function to run the recommendation system."""
    try:
        print("Starting lyrics learning recommendation system...")
        print("Checking for song_dataset.csv in current directory...")

        # Check if dataset exists before creating recommender
        if not os.path.exists('song_dataset.csv'):
            print("\nERROR: song_dataset.csv not found!")
            print("Please make sure the song dataset file is in the same directory as this script.")
            print("Expected path: " + os.path.abspath('song_dataset.csv'))
            print("\nCurrent directory contains these files:")
            for file in os.listdir('.'):
                print(f"  - {file}")
            return

        # Create and run recommender
        recommender = LyricsLearningRecommender()
        recommender.run_interactive_cli()

    except Exception as e:
        print(f"\nUnexpected error occurred: {e}")
        import traceback
        print("\nError details:")
        traceback.print_exc()
        print("\nPlease check your dataset file and try again.")


# Ensure this script runs when executed directly
if __name__ == "__main__":
    print("Executing main function...")
    main()
else:
    print("This module is being imported, not executed directly.")