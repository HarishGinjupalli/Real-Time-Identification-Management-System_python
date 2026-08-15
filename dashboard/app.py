"""
app.py
------
The live dashboard for the Real-Time Identification Management
System (rIMS).

It connects to SQL Server, pulls the latest recognition logs with
Pandas, and shows KPI cards + Plotly charts. It refreshes itself
on a timer so you can watch recognition events appear as
recognize_live.py sees people in front of the camera.

Run with:
    streamlit run dashboard/app.py

Make sure recognize_live.py is ALSO running in another terminal,
otherwise there's no new data to show.

WHY THIS VERSION IS DIFFERENT FROM A "while True" LOOP:
Streamlit scripts are meant to run top-to-bottom and then stop -
Streamlit itself handles re-running your script when needed. A
"while True: ... time.sleep(...)" loop fights against that: it
never lets the script finish, which can make the page feel less
responsive and isn't the pattern Streamlit is built around.

The correct tool for "refresh this part of the page every few
seconds" is a FRAGMENT: a function decorated with
@st.fragment(run_every=...). Streamlit reruns just that function
on the given interval, without reloading the whole page or
blocking anything else. This is what actually makes the dashboard
feel real-time and stay responsive.
"""

import sys
import os

import streamlit as st
import pandas as pd
import plotly.express as px

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database import get_engine
from analytics.recognition_analytics import (
    load_recognition_logs, load_persons, get_recognition_kpis,
    get_events_by_person, get_status_breakdown, get_events_over_time,
)

st.set_page_config(page_title="rIMS - Real-Time Identification Dashboard", layout="wide")

# How often the live section refreshes itself, in seconds. Lower =
# feels more "instant" but hits the database more often. 2 seconds
# is a good balance for a single-viewer dashboard like this one.
REFRESH_SECONDS = 2

# get_engine() only needs to run once per browser session, not on
# every single refresh - @st.cache_resource keeps the same engine
# (and its connection pool) alive across reruns instead of
# reconnecting every 2 seconds.
@st.cache_resource
def get_cached_engine():
    return get_engine()


def load_all_data(engine):
    """Pulls everything the dashboard needs from SQL Server."""
    logs_df = load_recognition_logs(engine, limit=2000)
    persons_df = load_persons(engine)
    return {"logs": logs_df, "persons": persons_df}


def render_kpi_cards(data):
    """Draws the KPI number cards."""
    kpis = get_recognition_kpis(data["logs"])

    row1 = st.columns(4)
    row1[0].metric("Registered People", f"{len(data['persons']):,}")
    row1[1].metric("Total Recognition Events", f"{kpis['total_events']:,}")
    row1[2].metric("Recognized Events", f"{kpis['recognized_events']:,}")
    row1[3].metric("Unknown Attempts", f"{kpis['unknown_events']:,}")


def render_trend_chart(data):
    """Draws the recognition events over time trend line."""
    st.subheader("Recognition Activity Over Time")

    trend_df = get_events_over_time(data["logs"], freq="min")
    if not trend_df.empty:
        fig = px.line(trend_df, x="time_bucket", y="count",
                       title="Events Per Minute", markers=True)
        st.plotly_chart(fig, use_container_width=True, key="trend_chart")
    else:
        st.info("No recognition events yet. Start recognize_live.py and "
                "let it see a face.")


def render_breakdown_charts(data):
    """Draws the person breakdown and recognized-vs-unknown charts."""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Recognized People")
        person_df = get_events_by_person(data["logs"])
        if not person_df.empty:
            fig = px.bar(person_df, x="person_name", y="event_count",
                          title="Recognition Events by Person")
            st.plotly_chart(fig, use_container_width=True, key="person_chart")
        else:
            st.info("No recognized events yet.")

    with col2:
        st.subheader("Recognized vs Unknown")
        status_df = get_status_breakdown(data["logs"])
        if not status_df.empty:
            fig = px.pie(status_df, names="status", values="count",
                          title="Recognition Outcome Breakdown")
            st.plotly_chart(fig, use_container_width=True, key="status_chart")
        else:
            st.info("No events yet.")


def render_recent_events_table(data):
    """Draws a table of the most recent recognition events."""
    st.subheader("Recent Recognition Events")

    logs_df = data["logs"]
    if logs_df.empty:
        st.info("No events logged yet.")
        return

    display_cols = ["log_id", "person_name", "status", "confidence_score",
                     "camera_source", "event_timestamp"]
    st.dataframe(logs_df[display_cols].head(20), use_container_width=True,
                 hide_index=True, key="recent_events_table")


def render_registered_people_table(data):
    """Draws a table of everyone registered in the system."""
    st.subheader("Registered People")

    persons_df = data["persons"]
    if persons_df.empty:
        st.info("No one is registered yet. Run registration/register_face.py "
                "to add your first person.")
        return

    st.dataframe(persons_df, use_container_width=True, hide_index=True,
                 key="persons_table")


@st.fragment(run_every=REFRESH_SECONDS)
def render_live_dashboard():
    """
    This is the part of the page that auto-refreshes.

    @st.fragment(run_every=REFRESH_SECONDS) tells Streamlit: "re-run
    just this function every REFRESH_SECONDS, without touching the
    rest of the page." That's what gives us the live-updating feel
    without a blocking while-loop.

    We also compare the new event count to the last time this ran,
    and pop up a small toast notification when a NEW recognition
    event has come in since the last refresh - a nice, clear signal
    that things are genuinely live.
    """
    engine = get_cached_engine()
    data = load_all_data(engine)

    current_event_count = len(data["logs"])
    previous_event_count = st.session_state.get("last_event_count")

    if previous_event_count is not None and current_event_count > previous_event_count:
        newest_row = data["logs"].iloc[0]
        st.toast(
            f"New event: {newest_row['person_name']} ({newest_row['status']})",
            icon="👋"
        )

    st.session_state["last_event_count"] = current_event_count

    render_kpi_cards(data)
    st.divider()
    render_trend_chart(data)
    st.divider()
    render_breakdown_charts(data)
    st.divider()
    render_recent_events_table(data)
    st.divider()
    render_registered_people_table(data)
    st.caption(f"Last updated: {pd.Timestamp.now().strftime('%H:%M:%S')} "
               f"(live - refreshes every {REFRESH_SECONDS}s)")


def main():
    """Builds the dashboard page. The live section refreshes itself."""
    st.title("Real-Time Identification Management System (rIMS)")
    st.caption("Smart and secure identity verification using AI-based facial recognition.")

    render_live_dashboard()


if __name__ == "__main__":
    main()
