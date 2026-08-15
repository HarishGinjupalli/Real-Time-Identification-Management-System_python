"""
recognition_analytics.py
-------------------------
Functions that pull recognition log data out of SQL Server and
turn it into useful KPIs and tables using Pandas, for the
Streamlit dashboard to display.
"""

import pandas as pd

from database.database import safe_read_sql


def load_recognition_logs(engine, limit=2000):
    """
    Loads the most recent recognition log rows.

    Uses safe_read_sql instead of pd.read_sql directly, so that if
    the database is briefly unreachable (network blip, SQL Server
    restarting, etc.), this returns an empty table instead of
    crashing the dashboard.

    Input: SQLAlchemy engine, how many rows to load
    Returns: a Pandas DataFrame (empty if the read failed)
    """
    query = f"""
        SELECT TOP {limit} * FROM recognition_logs
        ORDER BY event_timestamp DESC
    """
    return safe_read_sql(query, engine)


def load_persons(engine):
    """
    Loads every registered person.
    Input: SQLAlchemy engine
    Returns: a Pandas DataFrame (empty if the read failed)
    """
    return safe_read_sql("SELECT * FROM persons", engine)


def get_recognition_kpis(logs_df):
    """
    Calculates the main KPIs shown at the top of the dashboard.
    Input: recognition logs DataFrame
    Returns: a dictionary of KPI values
    """
    if logs_df.empty:
        return {
            "total_events": 0, "recognized_events": 0, "unknown_events": 0,
            "unique_people_seen": 0,
        }

    recognized_df = logs_df[logs_df["status"] == "RECOGNIZED"]

    return {
        "total_events": len(logs_df),
        "recognized_events": len(recognized_df),
        "unknown_events": len(logs_df[logs_df["status"] == "UNKNOWN"]),
        "unique_people_seen": recognized_df["person_id"].nunique(),
    }


def get_events_by_person(logs_df, top_n=10):
    """
    Counts recognition events per person (recognized only).
    Input: recognition logs DataFrame, how many to return
    Returns: a DataFrame with columns [person_name, event_count]
    """
    if logs_df.empty:
        return pd.DataFrame(columns=["person_name", "event_count"])

    recognized_df = logs_df[logs_df["status"] == "RECOGNIZED"]
    if recognized_df.empty:
        return pd.DataFrame(columns=["person_name", "event_count"])

    grouped = recognized_df.groupby("person_name").agg(
        event_count=("log_id", "count")
    ).reset_index()
    return grouped.sort_values("event_count", ascending=False).head(top_n)


def get_status_breakdown(logs_df):
    """
    Counts RECOGNIZED vs UNKNOWN events - good for a pie chart.
    Input: recognition logs DataFrame
    Returns: a DataFrame with columns [status, count]
    """
    if logs_df.empty:
        return pd.DataFrame(columns=["status", "count"])

    grouped = logs_df.groupby("status").agg(count=("log_id", "count")).reset_index()
    return grouped


def get_events_over_time(logs_df, freq="min"):
    """
    Groups recognition events into time buckets for a trend chart.
    Input: recognition logs DataFrame, freq ("min" = per minute)
    Returns: a DataFrame with columns [time_bucket, count]
    """
    if logs_df.empty:
        return pd.DataFrame(columns=["time_bucket", "count"])

    df = logs_df.copy()
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    df["time_bucket"] = df["event_timestamp"].dt.floor(freq)

    grouped = df.groupby("time_bucket").agg(count=("log_id", "count")).reset_index()
    return grouped.sort_values("time_bucket")
