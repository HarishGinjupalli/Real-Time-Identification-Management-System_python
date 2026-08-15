# Real-Time Identification Management System (rIMS)

Smart and secure identity verification using AI-based facial recognition. A face registered once through your webcam can be recognized live afterward — every recognition event is logged to SQL Server and shown on a real-time Streamlit dashboard.

> ⚠️ **All data is your own test data.** This project is for learning/portfolio purposes. It is not a production-grade security or access-control system — see the Limitations section below.

---

## What It Does

```
Webcam (registration)
        ↓
Face photos saved + trained into an LBPH model
        ↓
Webcam (live recognition)
        ↓
SQL Server (recognition_logs)
        ↓
Streamlit Dashboard (auto-refreshing)
```

- **Register** a person: capture ~30 face photos via webcam, store them, and add the person to the database
- **Train**: build a face recognition model from every registered person's photos
- **Recognize live**: continuously watch the webcam feed, identify known faces (or flag "Unknown"), and log every event
- **Dashboard**: live KPIs, recognition trend chart, per-person breakdown, recognized-vs-unknown pie chart, and a recent events table

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Registration, recognition, analytics, dashboard |
| OpenCV (`opencv-contrib-python`) | Face detection (Haar Cascade) + face recognition (LBPH) |
| Microsoft SQL Server (via SSMS) | Storing registered people and recognition logs |
| pyodbc | Writing data to SQL Server |
| SQLAlchemy | Reading data into Pandas |
| Plotly | Dashboard charts |
| Streamlit | Live dashboard UI |

### Why LBPH instead of `face_recognition`/`dlib`?

The popular `face_recognition` Python library depends on `dlib`, which needs to be **compiled from source** on Windows (requiring CMake and Visual Studio Build Tools) — a common source of install failures. OpenCV's built-in **LBPH (Local Binary Patterns Histograms)** recognizer ships inside `opencv-contrib-python`, installs with one `pip install`, and needs no GPU or compiler. It's less powerful than modern deep-learning face recognition, but it's more than accurate enough for a portfolio-scale project with a handful of registered people, and it's much easier to get running.

---

## Project Structure

```
rIMS_project/
│
├── database/
│   ├── database.py           # SQL Server connection helper (pyodbc + SQLAlchemy)
│   └── schema.sql             # Creates the database and both tables (T-SQL)
│
├── registration/
│   └── register_face.py       # Captures a new person's face photos + DB record
│
├── recognition/
│   ├── train_model.py         # Trains the LBPH model from all registered faces
│   └── recognize_live.py      # Real-time webcam recognition + logging
│
├── analytics/
│   └── recognition_analytics.py   # KPIs and chart data for the dashboard
│
├── dashboard/
│   └── app.py                 # Streamlit live dashboard
│
├── face_data/                 # Created at runtime — captured face photos (gitignored)
├── models/                    # Created at runtime — trained model file (gitignored)
│
├── requirements.txt
├── .env                       # Database connection settings (not committed to git)
├── .gitignore
└── README.md
```

---

## Setup Instructions

### 1. Find your SQL Server instance name
Open SSMS and check the "Server name" box on the connect screen (e.g. `localhost`, `.`, or `YOURPC\SQLEXPRESS`).

### 2. Create the database and tables
In SSMS: **File → Open → File...**, open `database/schema.sql`, then click **Execute** (or `F5`). This creates the `rims_db` database and both tables.

### 3. Confirm the ODBC driver is installed
Search Windows for **"ODBC Data Sources (64-bit)"** → **Drivers** tab. If SSMS is installed, you almost certainly already have `ODBC Driver 17 for SQL Server` or similar.

### 4. Set up a Python virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 5. Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Configure `.env`
Open `.env` and set `DB_SERVER` to match your SSMS instance name. Leave `DB_AUTH=windows` if SSMS connects for you automatically without a username/password.

### 7. Test the database connection
```bash
python database/database.py
```
You should see `Connected to SQL Server successfully!`

### 8. Register at least one person
```bash
python registration/register_face.py
```
Type a name, then look at your webcam while it captures ~30 photos. Repeat this for everyone you want the system to recognize.

### 9. Train the model
```bash
python recognition/train_model.py
```
Run this again any time you register a new person or add more photos.

### 10. Start live recognition (Terminal 1)
```bash
python recognition/recognize_live.py
```
A window opens showing your webcam feed with green boxes (recognized) or red boxes (unknown) around detected faces. Press `q` to stop.

### 11. Start the dashboard (Terminal 2 — keep Terminal 1 running)
```bash
streamlit run dashboard/app.py
```
Opens at `http://localhost:8501`, refreshing automatically as new recognition events come in.

---

## Tuning Recognition Accuracy

In `recognition/recognize_live.py`:

```python
CONFIDENCE_THRESHOLD = 70
```

- **Real people showing up as "Unknown" too often?** Raise this number (e.g. `80`).
- **Strangers getting matched to a real name?** Lower this number (e.g. `50`).
- More training photos per person (captured from different angles/lighting) also noticeably improves accuracy — you can re-run `register_face.py` for the same name to add more, then re-run `train_model.py`.

---

## Limitations (Read Before Treating This as "Secure")

- LBPH is a classic, lightweight algorithm — it's noticeably less accurate than modern deep-learning face recognition, especially with poor lighting, glasses, masks, or significant angle changes.
- There's no "liveness detection" — a printed photo or a phone screen showing someone's face could potentially fool it. A real access-control product would add anti-spoofing checks.
- Recognition quality scales with how many good, varied training photos each person has.

This project is meant to demonstrate the **end-to-end pipeline** (capture → train → recognize → log → visualize), which is the same architecture real identity-verification products use — just with a simpler, more transparent recognition algorithm at its core.

---

## Robustness Improvements (Why It Shouldn't Crash From Normal Problems)

These changes were added specifically so the app survives everyday hiccups instead of crashing outright:

| Problem | What was added |
|---|---|
| Database briefly unreachable when writing a recognition event | `log_recognition_event()` catches the error, prints a warning, and lets the webcam loop keep running instead of dying |
| Database unreachable when the dashboard reads data | `safe_read_sql()` catches the error and returns an empty table instead of crashing the whole Streamlit app |
| Database unreachable right when `recognize_live.py` starts | `load_person_names()` falls back to an empty list (faces show as "Unknown") instead of refusing to start |
| A single dropped webcam frame | The capture loop skips it and tries again, only stopping if **30 frames in a row** fail (a real disconnect) |
| Every query opening a brand new database connection | `get_engine()` now uses a **connection pool** (5 reusable connections, up to 10 extra under load, auto-recycled every 30 minutes, pre-checked before use) |
| A single failed connection attempt | `get_db_connection()` now retries up to 3 times with a short delay before giving up |
| Dashboard queries slowing down as data grows | Added indexes on `event_timestamp` and `person_id` in `recognition_logs` (see `database/schema.sql`) |

These are genuine engineering practices, and they do make a real difference: the app will now ride out a dropped connection, a slow query, or a bad camera frame instead of crashing. That is a meaningfully different (and honest) claim from "this handles 70 lakh users" — see below.

---

## Scalability Considerations

This project is a **single-machine, single-user demo** — one webcam, one local SQL Server instance, one Python process per script. It's built to clearly show the full pipeline end-to-end, not to handle production traffic. Here's honestly what it can and can't do, and what would need to change to go further:

### What this setup can realistically handle
- A handful to a few dozen registered people
- One camera feed at a time, on one machine
- A single person (you) using the dashboard at once
- Demo/portfolio/interview purposes
- Now, with the robustness improvements above: surviving normal transient problems (a dropped connection, a slow query, a bad frame) without crashing

### What would break it at real scale (e.g. many cameras, many locations, many simultaneous dashboard users — like 55,000+ institutes and 70 lakh people)

No code change closes this gap — it requires actual infrastructure, deployed and paid for:

| Concern | Why this setup breaks | What real scale needs |
|---|---|---|
| Many camera feeds at once | `recognize_live.py` is a single loop tied to one `cv2.VideoCapture(0)` | Separate recognition worker processes per camera, distributed across machines/edge devices |
| Millions of concurrent users | One SQL Server instance, one small connection pool | Database clusters, read replicas, and a properly capacity-planned connection pool per app server |
| Dashboard read load at that volume | Even with indexes, a single database server has a hard ceiling | A **caching layer** (e.g. Redis) so repeated dashboard reads don't hit the database directly every time |
| Many simultaneous dashboard viewers | One Streamlit process serving one browser session well | Multiple app servers behind a **load balancer**, with session state designed for many concurrent users |
| Recognition accuracy/speed at hundreds+ of people | LBPH slows down and gets less accurate as the registered population grows | A deep-learning-based recognizer with face embeddings in a vector database for fast similarity search |
| Traffic bursts (admissions day, results day) | Fixed, single-instance capacity | **Auto-scaling** cloud infrastructure that adds capacity on demand, plus rate limiting |
| Reliability across many institutes/locations | No redundancy — this one machine is a single point of failure | Multiple app instances, database replication, monitoring/alerting, multi-region deployment |

### The honest takeaway
What was added in this update makes the project **robust** — it won't crash from the kind of ordinary problems any running system hits (a dropped connection, a slow query, a flaky camera frame). That's a real, defensible engineering claim you can make about this project.

What it does **not** do is give it the capacity to serve 70 lakh people across 55,000+ institutes — that requires cloud infrastructure (multiple servers, database clusters, load balancers, CDNs) that costs real money and is normally built and operated by a dedicated infrastructure team, completely separate from the application code itself. Claiming this local project "handles" that scale wouldn't hold up under any technical scrutiny (e.g. in an interview) — but explaining that you understand *both* what it takes to make code robust *and* what additional infrastructure real scale requires is a genuinely strong thing to be able to talk about.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Could not open the webcam` | Close any other app using the camera (Zoom, Teams, etc.), or check Windows camera privacy settings allow desktop apps |
| `No trained model found` | Run `register_face.py` then `train_model.py` before `recognize_live.py` |
| Everyone shows as "Unknown" | Re-check `CONFIDENCE_THRESHOLD`, and make sure `train_model.py` was re-run after the most recent registration |
| `Login failed` / connection errors | Double-check `DB_SERVER` in `.env` matches SSMS exactly |
| Dashboard shows no data | Make sure `recognize_live.py` is running and has actually detected a face |
| `pip install` fails building a package from source | Run `python -m pip install --upgrade pip` first, then retry |

## Disclaimer

This project uses your own webcam and voluntarily-registered test data. It's built for learning and portfolio purposes and is not intended as a production security or access-control system.
