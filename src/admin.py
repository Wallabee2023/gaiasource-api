"""
Database Operations Script

This script provides functionality to interact with a MySQL database, including:
- Establishing a database connection
- Inserting and querying data
- Fetching data in batches
- Testing database connectivity and querying capabilities
- Retrieving database diagnostics such as size, row count, and uptime

Dependencies:
- mysql-connector-python
- json
- time
- logging

Configuration:
- Database connection parameters (server host, username, password) should be set in the 'config' module.

Usage:
- Update the `main()` function to test specific database operations as needed.
"""

import mysql.connector
from mysql.connector import Error
from config import *
import json
import time
import logging
#import multiprocessing as mp
from util import *
import functools

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{LOG_PATH}/db_operations.log'),  # Log messages to 'db_operations.log'
        logging.StreamHandler()  # Also log messages to the console
    ]
)
logger = logging.getLogger(__name__)

def get_db_connection(db_name, verbose=False):
    """Create and return a MySQL connection."""
    connection = None
    try:
        connection = mysql.connector.connect(
            host=SERVER_HOST,
            user=ADMIN_USERNAME,
            password=ADMIN_PASSWORD,
            database=db_name
        )
        if connection.is_connected() and verbose:
            logger.info("Connection to MySQL DB successful")
    except Error as e:
        if verbose:
            logger.error(f"Error connecting to MySQL DB: {e}")
    return connection

def add_data(connection, table, data, verbose=False):
    """Add data to the specified table."""
    cursor = connection.cursor()
    try:
        # Construct SQL query with placeholders
        placeholders = ", ".join(["%s"] * len(data))
        sql = f"INSERT INTO {table} VALUES ({placeholders})"

        cursor.execute(sql, data)
        connection.commit()
        if verbose:
            logger.info("Data inserted successfully")
    except Error as e:
        if verbose:
            logger.error(f"Error inserting data: {e}")
    finally:
        cursor.close()

def query_data(connection, table):
    """Query all data from the specified table."""
    cursor = connection.cursor()
    try:
        sql = f"SELECT * FROM {table}"
        cursor.execute(sql)
        results = cursor.fetchall()
        return json.dumps({"results": results})
    except Error as e:
        logger.error(f"Error querying data: {e}")
    finally:
        cursor.close()

def query_batch(table_name, limit, start_id, db_path):
    """Query a batch of data from the database."""
    connection = get_db_connection(GAIA_DATABASE_NAME)
    cursor = connection.cursor()

    query = f"SELECT * FROM {table_name} WHERE id > {start_id} ORDER BY id LIMIT {limit}"
    cursor.execute(query)
    rows = cursor.fetchmany(limit)

    # Close the connection after use
    connection.close()

    # Convert rows to JSON format
    return rows, start_id

def test_connection(db_name, table):
    """Test the database connection and perform a sample read operation."""
    logger.info("Running database connection test...")
    connection = get_db_connection(db_name)

    if connection:
        try:
            logger.info(f"Successfully connected to database: {db_name}")

            # Perform a simple query to ensure the connection is functional
            query = f"SELECT COUNT(*) FROM {table};"
            cursor = connection.cursor()
            cursor.execute(query)
            count = cursor.fetchone()[0]
            logger.info(f"Table '{table}' contains {count} rows.")
            
            # Optionally, fetch and print a few rows of data
            cursor.execute(f"SELECT * FROM {table} LIMIT 5;")
            rows = cursor.fetchall()
            logger.info(f"Sample data from table '{table}': {rows}")

        except Error as e:
            logger.error(f"Error during database operation: {e}")

        finally:
            cursor.close()
            connection.close()
            logger.info("Connection closed.")

def get_total_row_count(connection, table_name):
    """Get the total number of rows in the specified table."""
    cursor = connection.cursor()
    query = f"SELECT COUNT(*) FROM {table_name}"
    cursor.execute(query)
    total_rows = cursor.fetchone()[0]  # Fetch the count from the query result
    cursor.close()
    return total_rows

def get_min_max_ids(table_name):
    connection = get_db_connection("gaia")
    cursor = connection.cursor()
    
    query = f"SELECT MIN(id), MAX(id) FROM {table_name}"
    cursor.execute(query)
    
    min_id, max_id = cursor.fetchone()
    
    connection.close()
    
    return min_id, max_id

def get_database_diagnostics(start_time):
    """Get diagnostics data including the number of entries, database size, and server uptime."""
    connection = get_db_connection(GAIA_DATABASE_NAME)
    cursor = connection.cursor()

    # Query for database size in GB
    size_query = f"""
    SELECT ROUND(SUM(data_length + index_length) / 1024 / 1024 / 1024, 2)
    FROM information_schema.TABLES 
    WHERE table_schema = '{GAIA_DATABASE_NAME}' 
    AND table_name = '{GAIA_TABLE_NAME}';
    """
    cursor.execute(size_query)
    size_result = cursor.fetchone()[0]  # Fetch the size in GB

    # Get the number of entries in a specific table
    count_query = f"SELECT COUNT(*) FROM {GAIA_TABLE_NAME};"
    cursor.execute(count_query)
    count_result = cursor.fetchone()[0]

    # Calculate server uptime
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    uptime = "{:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds))

    cursor.close()
    connection.close()

    return {
        "number_of_entries": count_result,
        "database_size_gb": size_result,
        "server_uptime": uptime
    }

def main():
    """Run the test connection function."""
    #test_connection("gaia", "GaiaData")
    #connection=get_db_connection("gaia")
    #for batch, start_id in query_data_generator(connection, 'GaiaData', 100000, 0):
    #    print(start_id)
    #process(6, [0,0,0])
    query_data_multiprocessing_generator([4500000,25000000])


if __name__ == "__main__":
    main()
