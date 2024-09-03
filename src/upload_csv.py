import csv
import mysql.connector
from mysql.connector import Error
from config import *
from admin import *
import requests
import os
import shutil
import sys
import gzip
from util import *

# Progress bar function
def print_progress_bar(iteration, total, bar_length=40):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(bar_length * iteration // total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write(f'\r[{bar}] {percent}% Complete')
    sys.stdout.flush() 

def download_file(url, local_filename, total_size = -1):
    """Download a file from a URL and save it to a local file with a progress bar."""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()

        if total_size == -1:
        	total_size = int(r.headers.get('content-length', 0))
        
        with open(local_filename, 'wb') as f:
            chunk_size = 8192
            # Download the file in chunks
            downloaded_size = 0
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    print_progress_bar(downloaded_size, total_size)

            # Finalize progress bar
            print()  # To move to the next line after progress bar

def install_gaia_data(FILENAME):
    """Install GaiaSource data"""

    print("Installing Gaia Star Catalogue...")

    URL = f"http://cdn.gea.esac.esa.int/Gaia/gdr3/gaia_source/{FILENAME}.gz"
    file_path = f"./data/{FILENAME}"
    
    # Check if the file already exists
    if os.path.exists(file_path):
        user_input = input(f"'{file_path}' already exists. Do you want to override it? (y/n): ").strip().lower()
        if user_input != 'y':
            print("Download skipped.")
            print()
            return

    # Download file
    download_file(URL, file_path+".gz")

    # Unpack the .gz file
    with gzip.open(file_path+".gz", "rb") as f_in:
        with open(file_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Delete .gz file after unpacking
    os.remove(file_path+".gz")

    print()

def append_gaiadata(file_path):
    """Read a CSV file and append the data to the GaiaData table."""
    connection = get_db_connection(GAIA_DATABASE_NAME)  # Replace with your database name
    if not connection:
        return

    bulk_size = 5000  # Number of rows per batch
    batch = []
    
    with open(file_path, "r") as f:
        # Skip initial comment lines
        lines = f.readlines()
        filtered_lines = [line for line in lines if not line.startswith("#")]

        csvreader = csv.DictReader(filtered_lines)

        total = len(filtered_lines)
        idx = 0

        for row in csvreader:
            idx += 1
            print(f"Processing row {idx} out of {total}",end='\r')

            try:
                # Extract data from row based on headers

                # Ensure distance is positive and known
                parallax_str = row["parallax"]
                if parallax_str.lower() in ("", "null", None):
                    continue

                parallax = float(parallax_str)
                if parallax <= 0:
                    continue

                dist = 1 / parallax 

                # Simple data to extract
                GaiaID = row['designation']
                ra = float(row['ra'])
                dec = float(row['dec'])

                # Ensure temperature is stored correctly
                temp_str = row["teff_gspphot"]
                temp = None
                if temp_str.lower() not in ("", "null", None):
                    temp = float(temp_str)

                # Ensure visual magnitude is stored correctly
                v_mag_str = row["phot_g_mean_mag"]
                v_mag = None
                if v_mag_str.lower() not in ("", "null", None):
                    v_mag = float(v_mag_str)

                # Convert spherical to cartesian
                x, y, z = spherical_to_cartesian(ra, dec, dist)

                # Final data tuple
                data = (
                    GaiaID,
                    x,
                    y,
                    z,
                    temp,
                    v_mag
                )

                batch.append(data)

                # Bulk insert in batches
                if len(batch) >= bulk_size:
                    insert_data(connection, batch)
                    batch = []  # Clear the batch

            except ValueError as e:
                print(f"ValueError: {e}")

    # Insert any remaining rows
    if batch:
        insert_data(connection, batch)

    connection.close()

def insert_data(connection, data_batch):
    """Insert a batch of data into the database."""
    cursor = connection.cursor()
    try:
        query = """
        INSERT INTO GaiaData (GaiaID, x, y, z, temp, v_mag)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        cursor.executemany(query, data_batch)
        connection.commit()
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        cursor.close()

def autoPopulate(gaia_names_path):
    filenames = []
    
    # Open the file for reading
    with open(gaia_names_path, 'r') as file:
        for line in file:
            # Strip any leading/trailing whitespace and split by spaces
            parts = line.strip().split()
            
            # The filename is the last part of the line
            if len(parts) > 1:
                filenames.append(parts[-1])

    for FILENAME in filenames:
        FILENAME = FILENAME[:-3]
        print(f"Installing {FILENAME}")
        # For each gaia data file:
        install_gaia_data(FILENAME)
        append_gaiadata(f'./data/{FILENAME}')

        # Delete old csv
        os.remove(f'./data/{FILENAME}')


def main():
    autoPopulate('./data/gaia_names.txt')

if __name__ == "__main__":
	main()