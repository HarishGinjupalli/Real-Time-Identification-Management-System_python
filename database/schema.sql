-- =========================================================
-- Real-Time Identification Management System (rIMS)
-- Database Schema (Microsoft SQL Server)
-- =========================================================
-- Run this file in SQL Server Management Studio (SSMS):
--   1. Open SSMS and connect to your local server
--   2. Click "New Query"
--   3. Paste this entire file in and click "Execute" (or F5)
-- =========================================================

IF DB_ID('rims_db') IS NULL
BEGIN
    CREATE DATABASE rims_db;
END
GO

USE rims_db;
GO

-- ---------------------------------------------------------
-- Table: persons
-- Every person registered into the system (their face has
-- been captured and used to train the recognizer).
-- ---------------------------------------------------------
IF OBJECT_ID('dbo.persons', 'U') IS NULL
BEGIN
    CREATE TABLE persons (
        person_id INT IDENTITY(1,1) PRIMARY KEY,
        full_name VARCHAR(100) NOT NULL,
        registered_at DATETIME NOT NULL DEFAULT GETDATE(),
        photo_count INT NOT NULL DEFAULT 0,
        status VARCHAR(20) NOT NULL DEFAULT 'ACTIVE'   -- ACTIVE, INACTIVE
    );
END
GO

-- ---------------------------------------------------------
-- Table: recognition_logs
-- Every time the live camera sees a face, one row is written
-- here - whether it was recognized as a known person or not.
--
-- person_name is stored directly (not just looked up via
-- person_id) so the dashboard can show "Unknown" attempts
-- too, which have no matching person_id.
-- ---------------------------------------------------------
IF OBJECT_ID('dbo.recognition_logs', 'U') IS NULL
BEGIN
    CREATE TABLE recognition_logs (
        log_id INT IDENTITY(1,1) PRIMARY KEY,
        person_id INT NULL,
        person_name VARCHAR(100) NOT NULL,        -- "Unknown" if not recognized
        status VARCHAR(20) NOT NULL,               -- RECOGNIZED, UNKNOWN
        confidence_score FLOAT NULL,                -- lower = more confident match (LBPH)
        camera_source VARCHAR(50) NOT NULL DEFAULT 'Webcam-0',
        event_timestamp DATETIME NOT NULL DEFAULT GETDATE(),
        CONSTRAINT FK_logs_person FOREIGN KEY (person_id) REFERENCES persons(person_id)
    );
END
GO

-- ---------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------
-- The dashboard's most common query is "give me the most recent
-- rows, ordered by time" (see analytics/recognition_analytics.py).
-- Without an index, SQL Server has to scan the ENTIRE table to
-- sort it every time - fine with a few hundred rows, but it gets
-- noticeably slower as recognition_logs grows into the thousands
-- or millions of rows. This index lets SQL Server jump straight
-- to the newest rows instead of scanning everything.
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = 'IX_recognition_logs_timestamp' AND object_id = OBJECT_ID('dbo.recognition_logs')
)
BEGIN
    CREATE INDEX IX_recognition_logs_timestamp
    ON recognition_logs (event_timestamp DESC);
END
GO

-- Speeds up "how many events does this person have" style queries
-- (used by get_events_by_person in the dashboard).
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = 'IX_recognition_logs_person' AND object_id = OBJECT_ID('dbo.recognition_logs')
)
BEGIN
    CREATE INDEX IX_recognition_logs_person
    ON recognition_logs (person_id);
END
GO
