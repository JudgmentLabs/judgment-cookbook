"""
This script is a cookbook of how to create a custom scorer using a ClassifierScorer.

Simply use a natural language prompt and guide the LLM to output a score based on the input by 
choosing from a set of options.
"""

from judgeval import JudgmentClient
from judgeval.scorers import ClassifierScorer
from judgeval.data import Example

text2sql_scorer = ClassifierScorer(
    "Text to SQL",
    slug="text2sql-487126418",
    threshold=1.0,
    conversation=[{
        "role": "system",
        "content": """You will be given a natural language query, a corresponding LLM-generated SQL query, and a table schema + (optional) metadata.

** TASK INSTRUCTIONS **
Your task is to decide whether the LLM generated SQL query properly filters for what the natural language query is asking, based on the table schema + (optional) metadata. 
Additionally, you should check if the SQL query is valid based on the table schema (checking for syntax errors, false column names, etc.)

** TIPS **
- Look for correct references to the table schema for column names, table names, etc.
- Check that the SQL query can be executed; make sure JOINs, GROUP BYs, ORDER BYs, etc. are valid with respect to the table schema.
- Check that aggregation functions (COUNT, SUM, AVG, etc.) are used appropriately with GROUP BY clauses
- Verify that WHERE conditions use the correct operators and data types for comparisons
- Ensure LIMIT and OFFSET clauses make sense for the query's purpose
- Check that JOINs use the correct keys and maintain referential integrity
- Verify that ORDER BY clauses use valid column names and sort directions
- Check for proper handling of NULL values where relevant
- Ensure subqueries are properly constructed and correlated when needed
- EVEN IF THE QUERY IS VALID, IF IT DOESN'T WORK FOR THE NATURAL LANGUAGE QUERY, YOU SHOULD CHOOSE "N" AS THE ANSWER.

** FORMATTING YOUR ANSWER **
If the SQL query is valid and works for the natural language query, choose option "Y" and otherwise "N". Provide a justification for your decision; if you choose "N", explain what about the LLM-generated SQL query is incorrect, or explain why it doesn't address the natural language query. 
IF YOUR JUSTIFICATION SHOWS THAT THE SQL QUERY IS VALID AND WORKS FOR THE NATURAL LANGUAGE QUERY, YOU SHOULD CHOOSE "Y" AS THE ANSWER. 
IF THE SQL QUERY IS INVALID, YOU SHOULD CHOOSE "N" AS THE ANSWER. 

** YOUR TURN **
Natural language query:
{{input}}

LLM generated SQL query:
{{actual_output}}

Table schema:
{{context}}
        """
    }],
    options={
        "Y": 1.0, 
        "N": 0.0
    }
)


if __name__ == "__main__":
    client = JudgmentClient()

    table_schema = """CREATE TABLE Artists (
    artist_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    genre VARCHAR(100),
    followers INT,
    popularity INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE Albums (
    album_id VARCHAR(50) PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    artist_id VARCHAR(50) NOT NULL,
    release_date DATE,
    total_tracks INT,
    album_type VARCHAR(50) CHECK (album_type IN ('album', 'single', 'compilation')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (artist_id) REFERENCES Artists(artist_id) ON DELETE CASCADE
);

CREATE TABLE Tracks (
    track_id VARCHAR(50) PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    album_id VARCHAR(50) NOT NULL,
    artist_id VARCHAR(50) NOT NULL,
    duration_ms INT NOT NULL,
    explicit BOOLEAN DEFAULT FALSE,
    popularity INT DEFAULT 0,
    preview_url VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (album_id) REFERENCES Albums(album_id) ON DELETE CASCADE,
    FOREIGN KEY (artist_id) REFERENCES Artists(artist_id) ON DELETE CASCADE
);

CREATE TABLE Users (
    user_id VARCHAR(50) PRIMARY KEY,
    username VARCHAR(100) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE Playlists (
    playlist_id VARCHAR(50) PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES Users(user_id) ON DELETE CASCADE
);

CREATE TABLE PlaylistTracks (
    playlist_id VARCHAR(50),
    track_id VARCHAR(50),
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (playlist_id, track_id),
    FOREIGN KEY (playlist_id) REFERENCES Playlists(playlist_id) ON DELETE CASCADE,
    FOREIGN KEY (track_id) REFERENCES Tracks(track_id) ON DELETE CASCADE
);

CREATE TABLE UserListeningHistory (
    history_id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    track_id VARCHAR(50) NOT NULL,
    listened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES Users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (track_id) REFERENCES Tracks(track_id) ON DELETE CASCADE
);
"""

    all_tracks_one_artist_correct = Example(
        input="Find all tracks by the artist 'Drake', sorted by popularity.",
        actual_output="""SELECT t.track_id, t.title, t.popularity, a.name AS artist_name
FROM Tracks t
JOIN Artists a ON t.artist_id = a.artist_id
WHERE a.name = 'Drake'
ORDER BY t.popularity DESC;
""",
        retrieval_context=[table_schema]
    )

    most_listened_to_one_user_correct = Example(
        input="Find the most listened to track by user 'user123'.",
        actual_output="""SELECT t.track_id, t.title, COUNT(uh.history_id) AS play_count
FROM UserListeningHistory uh
JOIN Tracks t ON uh.track_id = t.track_id
WHERE uh.user_id = 'user123'
GROUP BY t.track_id, t.title
ORDER BY play_count DESC
LIMIT 1;
""",
        retrieval_context=[table_schema]
    )

    highest_num_playlists_correct = Example(
        input="Find the 5 users with the highest number of playlists.",
        actual_output="""SELECT u.user_id, u.username, COUNT(p.playlist_id) AS total_playlists
FROM Users u
JOIN Playlists p ON u.user_id = p.user_id
GROUP BY u.user_id, u.username
ORDER BY total_playlists DESC
LIMIT 5;
""",
        retrieval_context=[table_schema]
    )

    most_popular_tracks_all_users_correct = Example(
        input="Find the 10 most popular tracks across all users.",
        actual_output="""SELECT t.track_id, t.title, COUNT(uh.history_id) AS total_listens
FROM Tracks t
JOIN UserListeningHistory uh ON t.track_id = uh.track_id
GROUP BY t.track_id, t.title
ORDER BY total_listens DESC
LIMIT 10;
""",
        retrieval_context=[table_schema]
    )

    most_popular_tracks_all_users_incorrect = Example(
        input="Find the 10 most popular tracks across all users.",
        actual_output="""SELECT t.track_user, t.title, COUNT(uh.history_id) AS total_listens
FROM Tracks t
JOIN UserHistory uh ON t.track_user = uh.track_user
GROUP BY t.track_user, t.title
ORDER BY total_listens DESC
LIMIT 10;
""",
        retrieval_context=[table_schema]
    )


    client.run_evaluation(
        examples=[
            all_tracks_one_artist_correct, 
            most_listened_to_one_user_correct, 
            highest_num_playlists_correct, 
            most_popular_tracks_all_users_correct, 
            most_popular_tracks_all_users_incorrect
        ],
        scorers=[text2sql_scorer],
        model="gpt-4o-mini",
        project_name="text2sql",
        eval_run_name="text2sql_test",
        override=True
    )
