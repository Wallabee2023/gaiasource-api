#optimised version
import numpy as np
import math
from scipy.integrate import simpson
from astropy import units as u
import astropy.constants as cc
from astropy.constants import h, c, k_B
import mysql.connector
from admin import *
from config import *
import json
from tqdm import tqdm
from PyAstronomy import pyasl
import multiprocessing as mp
import numba
from numba.typed import Dict

def spherical_to_cartesian(ra, dec, dist):
	# Convert angles from degrees to radians
	ra = np.radians(ra)
	dec = np.radians(dec)
	
	# Calculate the Cartesian coordinates
	x = dist * np.cos(dec) * np.cos(ra)
	y = dist * np.cos(dec) * np.sin(ra)
	z = dist * np.sin(dec)
	
	return x, y, z

def planck(lam, T):
    """
    Spectral radiance of a blackbody of temperature T.

    Keywords
    -------
    lam     Wavelength in nm (array)
    T       Temperature in K (array or scalar)

    Returns
    -------
    Spectral radiance in cgs units (erg/s/sr/cm2)
    """
    lam = lam * u.nm
    T = np.asarray(T) * u.K  # Ensure T is an array with units
    
    # Broadcasting will handle element-wise operations
    x = (cc.h * cc.c) / (lam * cc.k_B * T[:, np.newaxis])
    B = 2 * cc.h * cc.c**2 / lam**5 / (np.exp(x) - 1)
    
    return B.cgs.value

def rgb_from_T(T, std=False, ncol=1, showit=False):
    """
    Calculate RGB color of a Planck spectrum of temperature T.

    Parameters
    ----------
    T : array-like
        Temperature in K.
    std : bool, optional
        If True, use the standard RGB color space conversion matrix.
    ncol : int, optional
        Normalization constant, e.g., set ncol=255 for color range [0-255].
    showit : bool, optional
        If True, display the RGB values.

    Returns
    -------
    array-like
        RGB color values.
    """
    lam = np.linspace(350, 800, 100)
    B = planck(lam, T)

    RGB = np.zeros((len(T), 3))

    for i, temp in enumerate(T):
        if temp < 670:
            RGB[i] = [1, 0, 0]
        elif 670 <= temp < 675:
            rgb_vals = rgb(lam, B[i], std=std, ncol=1, showit=False)
            rgb_vals[2] = 0
            RGB[i] = rgb_vals
        elif 675 <= temp < 1e7:
            RGB[i] = rgb(lam, B[i], std=std, ncol=1, showit=False)
        else:
            RGB[i] = [0.63130101, 0.71233531, 1.]

    return ncol * RGB

def rgb(lam, spec, std=False, ncol=1, showit=False):
    """
    Return RGB color of a spectrum.

    Keywords
    --------
    lam:        Wavelength in nm
    spec:       Radiance, or intensity, or brightness, or luminosity, or
                whatever quantity with units that are sorta kinda
                energy/s/sterad/cm2/wavelength. Normalization doesn't matter.
    ncol:       Normalization constant, e.g. set ncol=255 for color range [0-255]
    """
    x, y = xy(lam, spec)
    z = 1 - x - y
    Y = 1.
    X = (Y / y) * x
    Z = (Y / y) * z
    XYZ = np.array([X, Y, Z])

    # Matrix for Wide RGB D65 conversion
    if std:
        XYZ2RGB = np.array([[3.2406, -1.5372, -0.4986],
                            [-0.9689,  1.8758,  0.0415],
                            [0.0557, -0.2040,  1.0570]])
    else:
        XYZ2RGB = np.array([[1.656492, -0.354851, -0.255038],
                            [-0.707196,  1.655397,  0.036152],
                            [0.051713, -0.121364,  1.011530]])

    RGB = np.dot(XYZ2RGB, XYZ)  # Map XYZ to RGB
    RGB = adjust_gamma(RGB)     # Adjust gamma
    maxRGB = np.max(RGB)
    if maxRGB > 1:
        RGB = RGB / maxRGB  # Normalize to 1 if there are values above
    RGB = RGB.clip(min=0)  # Clip negative values
    RGB = ncol * RGB  # Normalize to number of colors

    return RGB.flatten()  # Flatten to ensure it's a 1D array

def xy(lam,L):
    """
    Return x,y position in CIE 1931 color space chromaticity diagram for an
    arbitrary spectrum.

    Keywords
    -------
    lam:    Wavelength in nm
    L:      Spectral radiance
    """
    lamcie,xbar,ybar,zbar = cie()               #Color matching functions
    L = np.interp(lamcie,lam,L)                 #Interpolate to same axis

    #Tristimulus values
    X = simpson(L*xbar,x=lamcie)
    Y = simpson(L*ybar,x=lamcie)
    Z = simpson(L*zbar,x=lamcie)
    XYZ = np.array([X,Y,Z])
    x = X / np.sum(XYZ)
    y = Y / np.sum(XYZ)
    z = Z / np.sum(XYZ)
    return x,y

def cie():
    """
    Color matching functions. Columns are wavelength in nm, and xbar, ybar,
    and zbar, are the functions for R, G, and B, respectively.
    """
    lxyz = np.array([[380., 0.0014, 0.0000, 0.0065],
                     [385., 0.0022, 0.0001, 0.0105],
                     [390., 0.0042, 0.0001, 0.0201],
                     [395., 0.0076, 0.0002, 0.0362],
                     [400., 0.0143, 0.0004, 0.0679],
                     [405., 0.0232, 0.0006, 0.1102],
                     [410., 0.0435, 0.0012, 0.2074],
                     [415., 0.0776, 0.0022, 0.3713],
                     [420., 0.1344, 0.0040, 0.6456],
                     [425., 0.2148, 0.0073, 1.0391],
                     [430., 0.2839, 0.0116, 1.3856],
                     [435., 0.3285, 0.0168, 1.6230],
                     [440., 0.3483, 0.0230, 1.7471],
                     [445., 0.3481, 0.0298, 1.7826],
                     [450., 0.3362, 0.0380, 1.7721],
                     [455., 0.3187, 0.0480, 1.7441],
                     [460., 0.2908, 0.0600, 1.6692],
                     [465., 0.2511, 0.0739, 1.5281],
                     [470., 0.1954, 0.0910, 1.2876],
                     [475., 0.1421, 0.1126, 1.0419],
                     [480., 0.0956, 0.1390, 0.8130],
                     [485., 0.0580, 0.1693, 0.6162],
                     [490., 0.0320, 0.2080, 0.4652],
                     [495., 0.0147, 0.2586, 0.3533],
                     [500., 0.0049, 0.3230, 0.2720],
                     [505., 0.0024, 0.4073, 0.2123],
                     [510., 0.0093, 0.5030, 0.1582],
                     [515., 0.0291, 0.6082, 0.1117],
                     [520., 0.0633, 0.7100, 0.0782],
                     [525., 0.1096, 0.7932, 0.0573],
                     [530., 0.1655, 0.8620, 0.0422],
                     [535., 0.2257, 0.9149, 0.0298],
                     [540., 0.2904, 0.9540, 0.0203],
                     [545., 0.3597, 0.9803, 0.0134],
                     [550., 0.4334, 0.9950, 0.0087],
                     [555., 0.5121, 1.0000, 0.0057],
                     [560., 0.5945, 0.9950, 0.0039],
                     [565., 0.6784, 0.9786, 0.0027],
                     [570., 0.7621, 0.9520, 0.0021],
                     [575., 0.8425, 0.9154, 0.0018],
                     [580., 0.9163, 0.8700, 0.0017],
                     [585., 0.9786, 0.8163, 0.0014],
                     [590., 1.0263, 0.7570, 0.0011],
                     [595., 1.0567, 0.6949, 0.0010],
                     [600., 1.0622, 0.6310, 0.0008],
                     [605., 1.0456, 0.5668, 0.0006],
                     [610., 1.0026, 0.5030, 0.0003],
                     [615., 0.9384, 0.4412, 0.0002],
                     [620., 0.8544, 0.3810, 0.0002],
                     [625., 0.7514, 0.3210, 0.0001],
                     [630., 0.6424, 0.2650, 0.0000],
                     [635., 0.5419, 0.2170, 0.0000],
                     [640., 0.4479, 0.1750, 0.0000],
                     [645., 0.3608, 0.1382, 0.0000],
                     [650., 0.2835, 0.1070, 0.0000],
                     [655., 0.2187, 0.0816, 0.0000],
                     [660., 0.1649, 0.0610, 0.0000],
                     [665., 0.1212, 0.0446, 0.0000],
                     [670., 0.0874, 0.0320, 0.0000],
                     [675., 0.0636, 0.0232, 0.0000],
                     [680., 0.0468, 0.0170, 0.0000],
                     [685., 0.0329, 0.0119, 0.0000],
                     [690., 0.0227, 0.0082, 0.0000],
                     [695., 0.0158, 0.0057, 0.0000],
                     [700., 0.0114, 0.0041, 0.0000],
                     [705., 0.0081, 0.0029, 0.0000],
                     [710., 0.0058, 0.0021, 0.0000],
                     [715., 0.0041, 0.0015, 0.0000],
                     [720., 0.0029, 0.0010, 0.0000],
                     [725., 0.0020, 0.0007, 0.0000],
                     [730., 0.0014, 0.0005, 0.0000],
                     [735., 0.0010, 0.0004, 0.0000],
                     [740., 0.0007, 0.0002, 0.0000],
                     [745., 0.0005, 0.0002, 0.0000],
                     [750., 0.0003, 0.0001, 0.0000],
                     [755., 0.0002, 0.0001, 0.0000],
                     [760., 0.0002, 0.0001, 0.0000],
                     [765., 0.0001, 0.0000, 0.0000],
                     [770., 0.0001, 0.0000, 0.0000],
                     [775., 0.0001, 0.0000, 0.0000],
                     [780., 0.0000, 0.0000, 0.0000]])
    return lxyz.T

def adjust_gamma(RGB):
    """
    Adjust gamma value of RGB color.
    
    Parameters
    ----------
    RGB : array-like
        RGB color values.
    
    Returns
    -------
    array-like
        Gamma-adjusted RGB color values.
    """
    a = 0.055
    mask = RGB <= 0.0031308
    RGB[mask] *= 12.92
    RGB[~mask] = (1 + a) * RGB[~mask]**(1 / 2.4) - a
    return RGB

def precompute_colors(temp_range, step=COLOR_TEMPERATURE_COARSENESS, filename=COLOR_LOOKUP_TABLE_PATH):
    """
    Precompute RGB colors for a range of temperatures and save to a JSON file.
    
    Parameters
    ----------
    temp_range : tuple
        The range of temperatures (min_temp, max_temp) in Kelvin.
    step : int
        The step size between temperatures.
    filename : str
        The name of the file to save the lookup table.
    """
    temperatures = np.arange(temp_range[0], temp_range[1] + step, step)
    colors = {}

    for temp in temperatures:
        # Calculate the RGB color for each temperature
        rgb = rgb_from_T([temp])[0]  # Assuming rgb_from_T returns an array of RGB values
        colors[int(temp)] = rgb.tolist()  # Convert NumPy array to list for JSON serialization

    # Save the lookup table to a JSON file
    with open(filename, 'w') as file:
        json.dump(colors, file, indent=4)
    
    print(f"Lookup table saved to {filename}")

#precompute_colors((1000,50000))

def load_color_lookup_table(filename=COLOR_LOOKUP_TABLE_PATH):
    """
    Load the color lookup table from a JSON file.
    
    Parameters
    ----------
    filename : str
        The name of the file containing the color lookup table.
    
    Returns
    -------
    dict
        A dictionary mapping temperatures to RGB values.
    """
    with open(filename, 'r') as file:
        colors = json.load(file)

    colors = {int(key): value for key, value in colors.items()}

    return colors

def process_temperatures(temp_array, lookup_table):
    """
    Map a NumPy array of temperatures to their corresponding RGB values using a lookup table.
    
    Parameters
    ----------
    temp_array : numpy.ndarray
        Array of temperatures in Kelvin.
    lookup_table : dict
        Dictionary mapping temperatures to RGB values.
    
    Returns
    -------
    numpy.ndarray
        Array of RGB values corresponding to the input temperatures.
    """
    # Ensure temperatures are integers for lookup
    temp_array = np.nan_to_num(temp_array, nan=6000)
    temp_array = np.round(temp_array).astype(int)
    rounded_temps = np.round(temp_array / COLOR_TEMPERATURE_COARSENESS) * COLOR_TEMPERATURE_COARSENESS

    # Initialize RGB array
    rgb_array = np.zeros((len(rounded_temps), 3))
    
    # Map each rounded temperature to its corresponding RGB value
    for i, temp in enumerate(rounded_temps):
        if temp in lookup_table:
            rgb_array[i] = lookup_table[temp]
        else:
            rgb_array[i] = [0, 255, 0]  # Default color (red) if temperature is not in lookup table
    
    return (rgb_array*255).astype(int)

def rgb_to_hex(rgb):
    """
    Convert an array of RGB color values to hexadecimal format in a vectorized manner.

    Parameters
    ----------
    rgb : np.ndarray
        The RGB color values as an array of shape (n, 3) with values in the range [0, 1] or [0, 255].

    Returns
    -------
    np.ndarray
        The colors in hexadecimal format as an array of strings.
    """
    # Ensure RGB values are in the range [0, 255]
    if np.max(rgb) <= 1:
        rgb = (rgb * 255).astype(int)
    
    # Convert RGB to hexadecimal format
    hex_colors = np.apply_along_axis(lambda c: '#{:02X}{:02X}{:02X}'.format(*c), 1, rgb)
    return hex_colors

# Convert the NumPy array to a list of dictionaries
def convert_array_to_dict(data_array):
    # Define column names
    columns = ['GaiaID', 'x', 'y', 'z', 'hex_color', 'magnitude']
    
    # Convert array to a list of dictionaries
    list_of_dicts = [
        dict(zip(columns, row)) for row in data_array
    ]
    
    return list_of_dicts

@numba.njit()
def process_rows(rows, exoplanet_coords, visual_magnitude_cutoff, lookup):
    
    matrix = np.array(rows).T
    coordinates = matrix[1:4, :].astype(np.float32)
    temperatures = matrix[4, :].astype(np.float32)
    visual_magnitudes = matrix[5, :].astype(np.float32)

    #print("Parsed out components")

    # Compute all distances from exoplanet to the stars
    delta = coordinates - exoplanet_coords[:, np.newaxis]
    dist_ex = np.linalg.norm(delta, axis=0)

    # Compute all distances from exoplanet to the stars
    dist = np.linalg.norm(coordinates, axis=0)

    #print("Computed Distances...")

    # Compute relative magnitudes
    relative_magnitudes = visual_magnitudes + 5 * np.log10(dist_ex / dist)

    # Filter stars based on visual magnitude cutoff
    mask = relative_magnitudes < visual_magnitude_cutoff
    batch_star_list = matrix[:, mask]

    batch_star_list[1:4] = delta[:, mask]
    batch_star_list[5] = relative_magnitudes[mask]

    #print("Filtered stars...")

    # Check if temperatures array is not empty
    if temperatures[mask].size > 0:
        rgb_colors = process_temperatures(temperatures[mask], lookup)
        batch_star_list[4] = rgb_to_hex(rgb_colors)
    else:
        batch_star_list[4] = np.array([])  # Or handle as needed

    return convert_array_to_dict(batch_star_list.T)

def query_data_multiprocessing_generator(id_min, id_max, lookup, exoplanet_coords, visual_magnitude_cutoff):
    """Main function to manage multiprocessing of database queries"""
    num_rows = id_max - id_min
    n_workers = 4


    # Number of batches, rounded up
    n_batches = num_rows // BATCH_SIZE + (num_rows % BATCH_SIZE > 0)

    #print(id_max, id_min, n_batches)

    # Create a queue of batches to be processed
    batch_queue = [i * BATCH_SIZE + id_min for i in range(n_batches)]

    # Create a pool of n_workers processes
    with mp.Pool(n_workers) as pool:
        for result in pool.imap(
            functools.partial(worker, lookup=lookup, exoplanet_coords=exoplanet_coords, visual_magnitude_cutoff=visual_magnitude_cutoff), 
            batch_queue):

            if result:
                yield json.dumps(result) + '\n'

def worker(start_id, lookup, exoplanet_coords, visual_magnitude_cutoff):
    """Query a batch of data from the database."""
    connection = get_db_connection(GAIA_DATABASE_NAME)
    cursor = connection.cursor()

    query = f"SELECT * FROM {GAIA_TABLE_NAME} WHERE id > {start_id} LIMIT {BATCH_SIZE}"
    cursor.execute(query)
    rows = cursor.fetchmany(BATCH_SIZE)

    # Close the connection after use
    connection.close()

    # After the rows are fetched, we now process them

    # If the database query for the range of IDs returns an empty list, we can just terminate early
    if len(rows) == 0:
        return None

    star_list = process_rows(rows, exoplanet_coords, visual_magnitude_cutoff, lookup)

    # Return the processed rows

    if len(star_list) == 0:
        return None

    return star_list

def fetch_exoplanet_position(pl_name):
    try:
        response = pyasl.NasaExoplanetArchive().selectByPlanetName(pl_name)
    
        # Extract the information we care about
        
        ra = response['ra']
        dec = response['dec']
        dist = response['sy_dist'] / 1000 # Convert to kpc

        x, y, z = spherical_to_cartesian(ra, dec, dist)
        
        # Return the cartesian coords
        return [x, y, z], 0
    except Exception as e:
        return -1, e

if __name__ == "__main__":
    process(6,np.array([0,0,0]))
