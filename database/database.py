"""
database.py
------------
This file has ONE job: connect Python to our SQL Server database.

We give TWO ways to connect, because different parts of the
project need different tools:

1. get_db_connection()  -> uses pyodbc.
   We use this when we want to INSERT data (for example, logging
   a new recognition event, or registering a new person).

2. get_engine()         -> uses SQLAlchemy (built on top of pyodbc).
   Pandas works best with SQLAlchemy when READING data (for
   example, pulling recognition logs into a DataFrame for the
   dashboard).

All the connection details (server name, database name, etc.)
are stored in the .env file, NOT written directly in the code.

IMPORTANT: pyodbc needs the "ODBC Driver for SQL Server" to be
installed on your Windows machine. If you have SSMS installed,
you almost certainly already have it. You can double check by
opening "ODBC Data Sources (64-bit)" from the Windows Start Menu
and looking under the "Drivers" tab.
"""

import os
import time
import urllib.parse

import pyodbc
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv

load_dotenv()

DB_SERVER = os.getenv("DB_SERVER", "localhost")
DB_NAME = os.getenv("DB_NAME", "rims_db")
DB_DRIVER = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")

# DB_AUTH can be "windows" (use your Windows login - the SSMS default)
# or "sql" (use a SQL Server username/password you created)
DB_AUTH = os.getenv("DB_AUTH", "windows")
DB_USER = os.getenv("DB_USER", "")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# How many times to retry a failed connection before giving up, and
# how long to wait between tries. This is what protects the app from
# crashing outright if SQL Server has a brief hiccup (e.g. it's busy,
# or a network blip) - we retry a few times instead of failing on
# the very first attempt.
MAX_CONNECTION_RETRIES = 3
RETRY_DELAY_SECONDS = 2


def _build_odbc_connection_string():
    """
    Builds the raw ODBC connection string used by pyodbc.
    """
    if DB_AUTH.lower() == "windows":
        return (
            f"DRIVER={{{DB_DRIVER}}};"
            f"SERVER={DB_SERVER};"
            f"DATABASE={DB_NAME};"
            f"Trusted_Connection=yes;"
        )
    else:
        return (
            f"DRIVER={{{DB_DRIVER}}};"
            f"SERVER={DB_SERVER};"
            f"DATABASE={DB_NAME};"
            f"UID={DB_USER};"
            f"PWD={DB_PASSWORD};"
        )


def get_db_connection():
    """
    Creates and returns a pyodbc connection, retrying a few times
    if the first attempt fails (e.g. SQL Server is briefly busy).

    Input: none
    Returns: a live SQL Server connection object.

    Use this connection for writing data (INSERT/UPDATE).
    Remember to close the connection when you are done with it.

    Raises the last error if every retry attempt fails - callers
    should still handle that (see the try/except examples in
    recognize_live.py and dashboard/app.py) so ONE failed
    connection doesn't take down the whole script.
    """
    conn_str = _build_odbc_connection_string()

    last_error = None
    for attempt in range(1, MAX_CONNECTION_RETRIES + 1):
        try:
            return pyodbc.connect(conn_str, timeout=5)
        except pyodbc.Error as error:
            last_error = error
            if attempt < MAX_CONNECTION_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)

    raise last_error


def get_engine():
    """
    Creates and returns a SQLAlchemy engine for SQL Server, with
    CONNECTION POOLING enabled.

    Input: none
    Returns: a SQLAlchemy engine object.

    WHY POOLING MATTERS:
    Without pooling, every single query would open a brand new
    connection to SQL Server and close it right after - that's slow,
    and SQL Server only allows a limited number of connections at
    once. A connection pool keeps a small set of connections open
    and reuses them, which is both faster and far more reliable
    once more than one part of the app is reading from the database
    at the same time.

    - pool_size: how many connections to keep ready and reused
    - max_overflow: how many extra connections are allowed to open
      temporarily if pool_size isn't enough
    - pool_recycle: forces connections to be refreshed periodically,
      so we never try to reuse one SQL Server has quietly dropped
    - pool_pre_ping: tests each connection with a quick ping before
      handing it out, so a dead connection is caught and replaced
      automatically instead of causing a crash

    Use this engine for reading data with pandas, for example:
        df = pd.read_sql("SELECT * FROM recognition_logs", engine)
    """
    conn_str = _build_odbc_connection_string()
    quoted_conn_str = urllib.parse.quote_plus(conn_str)
    engine = create_engine(
        f"mssql+pyodbc:///?odbc_connect={quoted_conn_str}",
        pool_size=5,
        max_overflow=10,
        pool_recycle=1800,
        pool_pre_ping=True,
    )
    return engine


def safe_read_sql(query, engine, default=None):
    """
    A protective wrapper around pandas.read_sql(): if the database
    read fails for any reason (connection dropped, query timeout,
    etc.), this returns an empty/default result instead of crashing
    whatever called it.

    This is the key piece for "the dashboard should not crash" -
    a single bad query becomes a friendly empty table instead of
    an unhandled exception that stops the whole app.

    Input:
        query   - the SQL query string to run
        engine  - a SQLAlchemy engine (from get_engine())
        default - what to return if the query fails (an empty
                   DataFrame is used if you don't provide one)

    Returns: a pandas DataFrame (either the real result, or the
             fallback if something went wrong)
    """
    import pandas as pd

    if default is None:
        default = pd.DataFrame()

    try:
        return pd.read_sql(query, engine)
    except OperationalError as error:
        print(f"Database read failed, showing empty data instead. Error: {error}")
        return default


def insert_and_get_id(cursor, sql, params):
    """
    Runs an INSERT statement and returns the new row's identity
    (auto-generated primary key) value.

    SQL Server does not have a simple "cursor.lastrowid", so we
    ask SQL Server directly with "SCOPE_IDENTITY()".

    IMPORTANT: we run the INSERT and the SCOPE_IDENTITY() lookup
    as ONE combined statement (separated by a semicolon), and then
    explicitly move past the INSERT's empty result with
    cursor.nextset() before reading the identity value. Doing this
    as two separate cursor.execute() calls, or forgetting the
    nextset() step, can make pyodbc lose track of the result and
    either return nothing or raise "No results" errors.

    Input:
        cursor - an open pyodbc cursor
        sql    - the INSERT statement, using ? placeholders
        params - a tuple of values matching the ? placeholders

    Returns: the new row's ID as an integer
    """
    combined_sql = f"{sql}; SELECT SCOPE_IDENTITY();"
    cursor.execute(combined_sql, params)

    while cursor.description is None:
        cursor.nextset()

    row = cursor.fetchone()
    return int(row[0]) if row and row[0] is not None else None


if __name__ == "__main__":
    # Run:  python database/database.py
    try:
        conn = get_db_connection()
        print("Connected to SQL Server successfully!")
        conn.close()
    except Exception as error:
        print("Connection failed. Error details below:")
        print(error)
