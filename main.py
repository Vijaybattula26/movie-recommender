from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import pickle
from datetime import datetime
import pandas as pd

# --- 1. DATABASE CONFIGURATION ---
DATABASE_URL = "postgresql://postgres:admin123@127.0.0.1:5433/postgres"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- TABLES ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)
    genres = Column(String) # Stores "Action|Comedy|Drama"

class Rating(Base):
    __tablename__ = "ratings"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    movie_id = Column(Integer)
    rating = Column(Integer)

class WatchHistory(Base):
    __tablename__ = "watch_history"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    movie_id = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Drop old tables and create new ones (To update Schema)
Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- 2. APP & ML SETUP ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML Models
movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

# --- 3. DATA MODELS ---
class UserSignup(BaseModel):
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class GenreUpdate(BaseModel):
    user_id: int
    genres: str # "Action,Comedy"

class MovieRating(BaseModel):
    user_id: int
    movie_id: int
    rating: int

class HistoryLog(BaseModel):
    user_id: int
    movie_id: int

# --- 4. ROUTES ---

@app.post("/signup")
def signup(user: UserSignup, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == user.email).first()
    if existing: raise HTTPException(status_code=400, detail="Email taken")
    new_user = User(email=user.email, password=user.password, genres="")
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User created", "user_id": new_user.id}

@app.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user or db_user.password != user.password:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    return {"message": "Login successful", "user_id": db_user.id, "genres": db_user.genres}

@app.post("/update_genres")
def update_genres(data: GenreUpdate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == data.user_id).first()
    if user:
        user.genres = data.genres
        db.commit()
    return {"message": "Genres updated"}

@app.post("/rate")
def rate_movie(rating: MovieRating, db: Session = Depends(get_db)):
    new_rating = Rating(user_id=rating.user_id, movie_id=rating.movie_id, rating=rating.rating)
    db.add(new_rating)
    db.commit()
    return {"message": "Rating saved"}

@app.post("/log_history")
def log_history(log: HistoryLog, db: Session = Depends(get_db)):
    history = WatchHistory(user_id=log.user_id, movie_id=log.movie_id)
    db.add(history)
    db.commit()
    return {"message": "History logged"}

# --- HYBRID RECOMMENDATION ENGINE ---
# This fulfills "Collaborative Filtering" & "Hybrid Models"
@app.get("/recommend_hybrid/{user_id}")
def recommend_hybrid(user_id: int, db: Session = Depends(get_db)):
    # 1. Get User Ratings
    user_ratings = db.query(Rating).filter(Rating.user_id == user_id).all()
    
    recommended_movies = []
    
    # STRATEGY A: If user has ratings, find similar movies (Item-Item Collaborative)
    if user_ratings:
        # Find the movie they rated highest (5 stars)
        high_rated = [r for r in user_ratings if r.rating >= 4]
        if high_rated:
            # Pick the last highly rated movie
            last_liked = high_rated[-1].movie_id
            
            # Find it in our dataset
            movie_row = movies[movies['movie_id'] == last_liked]
            if not movie_row.empty:
                movie_idx = movie_row.index[0]
                distances = similarity[movie_idx]
                movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
                
                for i in movies_list:
                    rec_id = movies.iloc[i[0]].movie_id
                    recommended_movies.append({
                        "title": movies.iloc[i[0]].title,
                        "id": int(rec_id),
                        "reason": "Because you liked " + movie_row.iloc[0].title
                    })
                return {"type": "Hybrid (Based on Rating)", "recommendations": recommended_movies}

    # STRATEGY B: If no ratings, use Preferred Genres (Onboarding)
    user = db.query(User).filter(User.id == user_id).first()
    if user and user.genres:
        user_genres = user.genres.split(",") # e.g. ["Action", "Comedy"]
        # Find movies that match these genres (Simple Filter)
        # Note: This is a simplified logic for speed. In a real app, use vector filtering.
        filtered = movies[movies['genres'].apply(lambda x: any(g in x for g in user_genres))].head(5)
        
        for index, row in filtered.iterrows():
            recommended_movies.append({
                "title": row['title'],
                "id": int(row['movie_id']),
                "reason": "Based on your interest in " + user_genres[0]
            })
        return {"type": "Content (Based on Genres)", "recommendations": recommended_movies}

    # STRATEGY C: Fallback to Trending (First 5 movies)
    for i in range(5):
        recommended_movies.append({
            "title": movies.iloc[i].title,
            "id": int(movies.iloc[i].movie_id),
            "reason": "Trending Now"
        })
    return {"type": "Trending", "recommendations": recommended_movies}

# Keep the standard search endpoint
@app.get("/recommend/{movie}")
def recommend(movie: str):
    if movie not in movies['title'].values:
        raise HTTPException(status_code=404, detail="Movie not found")
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movies = []
    for i in movies_list:
        recommended_movies.append({
            "title": movies.iloc[i[0]].title,
            "id": int(movies.iloc[i[0]].movie_id)
        })
    return {"input_movie": movie, "recommendations": recommended_movies}