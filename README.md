# 🎬 End-to-End Hybrid Movie Recommendation System
> **A Full-Stack AI Application that recommends movies based on Content Similarity and User Preferences**

![Header](https://capsule-render.vercel.app/api?type=waving&color=141414&height=300&section=header&text=Movie%20Recommender%20System&fontSize=70&fontColor=E50914&fontAlign=50&fontAlignY=40&desc=Hybrid%20AI%20Engine%20•%20React%20•%20FastAPI%20•%20PostgreSQL&descSize=20&descAlign=50&descAlignY=60)

---

## 🔗 Live Demo
🎥 **[👉 Click here to visit the Live Website](https://movie-recommender-snowy.vercel.app)**  
*(Hosted on Vercel for frontend and Render for backend)*  

---

## 🧩 Project Overview

This project is a **complete end-to-end software product** developed as an **MCA Final Year Project**.  
It solves the common question — _“What should I watch next?”_ — using a **Hybrid Machine Learning Engine**.  

Unlike traditional recommenders, this system addresses the **Cold Start Problem** by onboarding new users with their genre preferences. Over time, it adapts using **Collaborative Filtering** logic based on user ratings and watch history.  

The system includes secure login, a Netflix-style dark mode interface, and real-time movie data fetched from TMDB API.  

---

## 🚀 Key Features

### 🧠 Hybrid AI Engine
- **Cold Start Users:** Recommends movies based on selected **genres** during onboarding.  
- **Active Users:** Uses **Item-Item Similarity** from past ratings and watch history.  

### 💡 Additional Features
- **🔐 Secure Authentication:** User signup/login stored in PostgreSQL.  
- **⭐ Rating System:** Users can rate movies (1–5 stars), updating recommendations instantly.  
- **🎞️ Interactive Dashboard:** Netflix-style dark theme with responsive layout.  
- **⚡ Real-Time Data:** Integrated with **TMDB API** for live posters, cast, and budget.  
- **📜 History Tracking:** Every watched or clicked movie is stored in the user’s watch history.  

---

## 🛠️ Tech Stack & Tools

| **Component** | **Technology Used** | **Purpose** |
| :------------- | :----------------- | :----------- |
| **Frontend** | React.js | Build dynamic UI and handle user interactions |
| **Styling** | CSS3 (Flexbox/Grid) | Netflix-style dark theme and responsive layout |
| **Backend** | FastAPI (Python) | High-performance REST API for data and ML logic |
| **Database** | PostgreSQL | Store users, passwords, ratings, and history |
| **ORM** | SQLAlchemy | Connect and manage database models |
| **Machine Learning** | Scikit-Learn | Cosine Similarity, CountVectorizer |
| **Data Handling** | Pandas, NumPy | Clean and preprocess dataset |
| **External API** | TMDB API | Fetch movie posters and real-time details |

---

## 📊 Dataset Information

- **Source:** TMDB 5000 Movie Dataset (Kaggle)  
- **Total Movies:** 4,803 entries  
- **Features Used:** `genres`, `keywords`, `cast`, `crew`, `overview`  
- **Processing Steps:**  
  - Combine columns into a single `tag` column  
  - Convert tags into 5000-dimensional vectors using `CountVectorizer`  
  - Compute **Cosine Similarity** between movies  

---

## 🧠 How the Hybrid Model Works

| **Stage** | **Description** |
| :--------- | :-------------- |
| **1. Data Preprocessing** | Combine overview, genres, keywords, cast, and crew into a unified "tag". |
| **2. Vectorization** | Convert text into numerical vectors using CountVectorizer. |
| **3. Similarity Calculation** | Compute cosine similarity between all movie vectors. |
| **4. Hybrid Recommendation Logic** |  
   - **New User:** Filter movies by preferred genres.  
   - **Active User:** Recommend based on similarity to previously rated movies. |

---

## ⚙️ Installation & Setup Guide

### 🧩 Prerequisites
Ensure the following are installed:  
- Python 3.x  
- Node.js  
- PostgreSQL (with pgAdmin 4)  

---

### 🗄️ Step 1: Database Setup
1. Open **pgAdmin 4**  
2. Create a new database named `postgres` (or use the default one)  
3. The backend automatically creates tables (`users`, `ratings`, `watch_history`) on startup  

---

### 💻 Step 2: Backend Setup (FastAPI)
```bash
# Navigate to the backend folder
cd backend

# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install fastapi uvicorn sqlalchemy psycopg2 scikit-learn pandas requests

# Run preprocessing (first time only)
python preprocess.py
python build_engine.py

# Start backend server
uvicorn main:app --reload

# 🌐 Step 3: Frontend Setup (React.js)
# Open new terminal
cd frontend

# Install dependencies
npm install

# Start the frontend
npm start
# Your app will be live at: http://localhost:3000🎉

🔑 Login Screen
<img width="800" src="https://github.com/user-attachments/assets/7712c4e7-36dc-437c-951e-f14fdc09aa7b" />
🎬 Genre Onboarding (Cold Start)
<img width="800" src="https://github.com/user-attachments/assets/a60ec7d0-8e60-42cf-8dcf-38fa0b2eae45" />
🏠 Dashboard with Recommendations
<img width="800" src="https://github.com/user-attachments/assets/7039f001-d3ff-4dfe-91d6-40a164401840" />
💡 Hybrid Logic in Action
<img width="800" src="https://github.com/user-attachments/assets/8477fe82-2290-4fe7-80a7-b9e34eb57835" />
📖 Movie Details Popup
<img width="800" src="https://github.com/user-attachments/assets/ec5de133-b012-467b-afe1-f971bc31a414" />

# 🔮 Future Scope

Feature	Description
Advanced Collaborative Filtering	Implement SVD or Matrix Factorization for larger user bases (>10K users).
Deep Learning Integration	Use Autoencoders for feature extraction and more accurate recommendations.
Mobile App Version	Convert React web app into React Native for Android & iOS.
Social Features	Allow users to follow friends and share playlists or reviews.

# 👨‍💻 Author

# 👋 Vijay Battula
# 🎓 MCA Final Year Student | Data Science & Full Stack Developer

# GitHub: github.com/Vijaybattula26

# LinkedIn: linkedin.com/in/battulavijay

# 🏁 Project Summary

Aspect	Description
Project Title	End-to-End Hybrid Movie Recommendation System
Domain	Machine Learning + Full Stack Development
Frontend	React.js, CSS
Backend	FastAPI (Python)
Database	PostgreSQL
Dataset	TMDB 5000 Movie Dataset
Project Type	MCA Final Year Project
Goal	Recommend movies intelligently using hybrid ML logic

# 🧾 License

This project is licensed under the MIT License — feel free to use and modify it with attribution.
