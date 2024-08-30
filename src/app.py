from flask import Flask, request, Response, stream_with_context, jsonify, abort
from werkzeug.middleware.proxy_fix import ProxyFix
import mysql.connector
from mysql.connector import Error
from config import *
from util import *
import logging
import time

# Record the start time when the program starts
start_time = time.time()

app = Flask(__name__)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{LOG_PATH}/app.log'),  # Log messages to 'app.log'
        logging.StreamHandler()  # Also log messages to the console
    ]
)
logger = logging.getLogger(__name__)

# Allowed hosts for the API
ALLOWED_HOSTS = ['api.exosky.org']

@app.before_request
def check_host():
    """Check if the request comes from an allowed host."""
    if request.host not in ALLOWED_HOSTS:
        logger.warning(f"Blocked request from: {request.host}")
        abort(403)
    logger.info(f"Accepted request from: {request.host}")

def get_db_connection(db_name):
    """Create and return a MySQL connection."""
    try:
        connection = mysql.connector.connect(
            host=SERVER_HOST,
            user=API_USERNAME,
            password=API_PASSWORD,
            database=db_name
        )
        return connection
    except Error as e:
        logger.error(f"Error connecting to database: {str(e)}")
        raise

@app.route('/stars', methods=['GET'])
def get_stars():
    """Fetch star data from the database with pagination."""
    page = request.args.get('page', default=1, type=int)
    per_page = request.args.get('per_page', default=10, type=int)

    if page < 1 or per_page < 1:
        return jsonify({"error": "Page and per_page parameters must be positive integers."}), 400

    offset = (page - 1) * per_page

    conn = None
    try:
        conn = get_db_connection(GAIA_DATABASE_NAME)
        cursor = conn.cursor(dictionary=True)

        # Query for the star data with pagination
        query = "SELECT * FROM GaiaData LIMIT %s OFFSET %s"
        cursor.execute(query, (per_page, offset))
        stars = cursor.fetchall()

        # Get the total number of records
        cursor.execute("SELECT COUNT(*) FROM GaiaData")
        total_records = cursor.fetchone()['COUNT(*)']

        # Calculate total pages
        total_pages = (total_records + per_page - 1) // per_page

        response = {
            "page": page,
            "per_page": per_page,
            "total_records": total_records,
            "total_pages": total_pages,
            "data": stars
        }
        return jsonify(response)

    except Error as e:
        logger.error(f"Database query error: {str(e)}")
        return jsonify({"error": str(e)}), 500

    finally:
        if conn:
            conn.close()

@app.route('/visible_stars', methods=['POST'])
def process_data():
    """Process star data based on input JSON and return results."""
    try:
        # Get JSON data from the request
        data = request.get_json()

        if data is None:
            logger.error('No JSON data provided')
            return jsonify({'error': 'No JSON data provided'}), 400

        # Log incoming request data
        logger.info(f'Incoming request data: {data}')

        # Extract specific fields from the JSON data
        visual_magnitude_cutoff = data.get('visual_magnitude_cutoff')
        exoplanet_cartesian_coordinates = np.array(data.get('exoplanet_cartesian_coordinates'))

        # Process and stream the result
        return Response(stream_with_context(process(visual_magnitude_cutoff, exoplanet_cartesian_coordinates)),
                        content_type='application/json')

    except Exception as e:
        logger.error(f'Exception occurred: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/exoplanet_data/<name>', methods=['GET'])
def exoplanet_position(name):
    """
    Retrieve the position of an exoplanet.

    Parameters
    ----------
    name : str
        The name of the exoplanet.

    Returns
    -------
    JSON
        A JSON response containing the x, y, z coordinates of the exoplanet.
    """
    position_list, _ = fetch_exoplanet_position(name)

    # Catch an error
    if position_list == -1:
        return jsonify({"exoplanet": name, "error": f"{_}"})

    # Return the position data as a JSON response
    return jsonify({"exoplanet": name, "position": dict(zip(["x","y","z"],position_list))})

@app.route('/diagnostics', methods=['GET'])
def diagnostics():
    """Provide database diagnostics and server uptime."""
    diagnostics_data = get_database_diagnostics(start_time)
    return jsonify(diagnostics_data)

if __name__ == '__main__':
    app.run(debug=True)
