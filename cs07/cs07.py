import matplotlib.pyplot as plt
import numpy as np


def construct_file_name(lat, lon):
    """ Takes the latitude and longitude as signed integers and
    constructs the appropriate file name for the TIF file. """
    file_name = 'USGS_NED_1_'
    if lat > 0:
        file_name += 'n' + str(lat)
    else:
        file_name += 's' + str((lat * -1))
    if lon > 0:
        file_name += 'e' + str('{:0>3}'.format(lon))
    else:
        file_name += 'w' + str('{:0>3}'.format(lon * -1))
    file_name += '_IMG.tif'
    return file_name


def load_trim_image(lat, lon):
    """ Takes the latitude and longitude as signed integers and
    loads the appropriate file. It then trims off the boundary
    of six pixels on all four sides. """
    # file = 'usgs_images/' + construct_file_name(lat, lon)
    file = construct_file_name(int(lat), int(lon))
    im = plt.imread(file)
    # np.set_printoptions(threshold=sys.maxsize)
    im = im[6:-6, 6:-6]
    return im


def stitch_four(lat, lon):
    im1 = load_trim_image(lat, lon)
    im2 = load_trim_image(lat, lon + 1)
    im3 = load_trim_image(lat - 1, lon)
    im4 = load_trim_image(lat - 1, lon + 1)
    # (nw_lat, nw_lon), (nw_lat, nw_lon+1)
    # (nw_lat-1, nw_lon), (nw_lat-1, nw_lon+1)
    image1 = np.hstack([im1, im2])
    image2 = np.hstack([im3, im4])
    image = np.vstack([image1, image2])
    return image


def get_row(lat, lon_min, num_tiles):
    """ Takes the latitude, minimum longitude, and number of tiles and
    returns an image that combines tiles along a row of different
    longitudes. """
    image = load_trim_image(lat, lon_min)
    x = 1
    while x < num_tiles:
        image = np.hstack([image, load_trim_image(lat, lon_min + x)])
        x += 1
    return image


def get_northwest(lat, lon):
    """ Get the integer coordinates of the northwest corner of the tile
    that contains this decimal (lat, lon) coordinate. """
    nw_lat = np.ceil(lat)
    if lon < 0:
        nw_lon = np.ceil(lon * -1)
        nw_lon = nw_lon * -1
    else:
        nw_lon = np.ceil(lon)
    return nw_lat, nw_lon


def get_tile_grid(lat_max, lon_min, num_lat, num_lon):
    """ Takes the northwest coordinate (maximum latitude, minimum longitude)
    and the number of tiles in each dimension (num_lat, num_lon) and
    constructs the image containing the entire range. """
    nw_lat, nw_lon = get_northwest(lat_max, lon_min)
    im = get_row(nw_lat, nw_lon, num_lon)
    x = 1
    while x < num_lat:
        image = get_row(nw_lat - x, nw_lon, num_lon)
        im = np.vstack([im, image])
        x = x + 1
    return im


def get_tile_grid_decimal(northwest, southeast):
    """ Construct the tiled grid of TIF images that contains these
    northwest and southeast decimal coordinates. Each corner
    is a tuple, (lat, lon). """
    nw1, nw2 = northwest
    se1, se2 = southeast
    nwlat, nwlon = get_northwest(nw1, nw2)
    selat, selon = get_northwest(se1, se2)
    nlat = (nwlon - selon) * -1 + 1
    nlon = nwlat - selat + 1
    return get_tile_grid(nwlat, nwlon, nlon, nlat)


def dec_to_dms(dec):
    """ Convert a decimal longitude or latitude into a DMS tuple
    (degrees, minutes, seconds). """
    degrees = int(dec)
    if degrees != 0:
        m = dec % degrees * -60
    else:
        m = dec * -60
    if np.isnan(m):
        minutes = int(np.nan_to_num(m))
    else:
        minutes = int(m)
    if minutes != 0:
        seconds = m % minutes * 60
    else:
        seconds = m * 60
    if np.isnan(seconds):
        seconds = int(np.nan_to_num(seconds))
    else:
        seconds = int(np.round(seconds, 0))
    if minutes < 0:
        minutes = minutes * -1
    if seconds < 0:
        seconds = seconds * -1
    return degrees, minutes, seconds


def seconds_to_index(seconds):
    """ Convert seconds into an image index. If the seconds are zero, the
    index should be zero. Otherwise, 3599 seconds should map to index 1
    and 1 second should map to 3599. """
    if seconds != 0:
        index = 3600 - seconds
    else:
        index = 0
    return index


def get_trim(northwest, southeast):
    """ Determine the number of pixels to crop based on the
    northwest and southeast corner of the region of interest (as tuples). """
    nw1, nw2 = northwest
    se1, se2 = southeast
    nwlat, nwlon = get_northwest(nw1, nw2)
    selat, selon = get_northwest(se1, se2)
    selat = selat - (3599/3600)
    selon = selon + (3599/3600)
    d1, m1, s1 = dec_to_dms(abs(nw1 - nwlat))
    top = m1 * 60 + s1
    d2, m2, s2 = dec_to_dms(abs(nw2 - nwlon))
    left = m2 * 60 + s2
    d3, m3, s3 = dec_to_dms(abs(se1 - selat))
    bottom = m3 * 60 + s3
    d4, m4, s4 = dec_to_dms(abs(se2 - selon))
    right = m4 * 60 + s4

    return left, right, bottom, top


def get_roi(center, n):
    """ Given the center (lat, lon) coordinate and a number of arc-seconds
    to either side (north, south, east, and west), return the northwest
    and southeast coordinate that define the region of interest:
    (north_latitude, west_longitude), (south_latitude, east_longitude)"""
    n = n / 3600
    lat, lon = center
    northwest = lat + n, lon - n
    southeast = lat - n, lon + n
    return northwest, southeast


def crop(im, trim):
    """ Crop the image by the number of pixels specified in 'trim':
    trim = [left, right, bottom, top]. """
    left, right, bottom, top = trim
    if right == 0:
        image = im[top:-bottom, left:]
    elif bottom == 0:
        image = im[top:, left:-right]
    elif bottom == 0 and right == 0:
        image = im[top:, left:]
    else:
        image = im[top:-bottom, left:-right]
    return image


def get_extent(northwest, southeast):
    """ Return a 4-tuple containing the extent of the region of interest
    for the plt.imshow function: [left, right, bottom, top]."""
    nwlat, nwlon = northwest
    selat, selon = southeast
    return nwlon, selon, selat, nwlat


def zoom(center, n):
    """ Create a square image centered at (lat, lon) with 2n+1 arc-seconds (pixels)
    high and wide. Also return a list of the extent of the image
    [west_lon, east_lon, south_lat, north_lat]. """
    northwest, southeast = get_roi(center, n)
    image = get_tile_grid_decimal(northwest, southeast)
    cut = get_trim(northwest, southeast)
    image = crop(image, cut)
    extent = get_extent(northwest, southeast)
    return extent, image
