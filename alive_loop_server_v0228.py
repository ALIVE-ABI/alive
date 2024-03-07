#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

print("Python executable:", sys.executable)
print("Python version:", sys.version)


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.cloud import storage
import os
import datetime
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import CubicSpline
from multiprocessing import Pool
from pyproj import CRS
import rioxarray as rxr
import joblib
from datetime import date
import time
import fsspec
import zarr
import h5netcdf

today = date.today()
doy = date.today().timetuple().tm_yday
print(today)
print('Day of year:',doy)

year = str(today).split('-')[0]

import warnings
warnings.filterwarnings("ignore")


# In[3]:


import sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.neural_network import MLPRegressor


# In[4]:


print(sklearn.__version__)
print(pd.__version__)
print(xr.__version__)
print(rxr.__version__)
print(zarr.__version__)


# In[5]:


# DSRmodel = joblib.load('/Users/Sophie/Desktop/DSR_GBR_2024-02-27.pkl')
# GPPmodel = joblib.load('/Users/Sophie/Desktop/GPP_GBR_2024-02-27.pkl')
DSRmodel = joblib.load('/home/shoffman/DSR_GBR_2024-02-29.pkl')
GPPmodel = joblib.load('/home/shoffman/GPP_GBR_2024-02-29.pkl')


# In[6]:


from datetime import datetime, timedelta
from numpy import radians, ndarray, sin, cos, degrees, arctan2, arcsin, tan, arccos

class solar:
    """
    from: https://github.com/NASA-DEVELOP/dnppy/blob/master/dnppy/solar/solar.py
    author: 'Jwely'

    Object class for handling solar calculations. Many equations are taken from the
    excel sheet at this url : [http://www.esrl.noaa.gov/gmd/grad/solcalc/calcdetails.html]

    It requires a physical location on the earth and a datetime object

    :param lat:             decimal degrees latitude (float OR numpy array)
    :param lon:             decimal degrees longitude (float OR numpy array)
    :param time_zone:       float of time shift from GMT (such as "-5" for EST)
    :param date_time_obj:   either a timestamp string following fmt or a datetime obj
    :param fmt:             if date_time_obj is a string, fmt is required to interpret it
    :param slope:           slope of land at lat,lon for solar energy calculations
    :param aspect:          aspect of land at lat,lon for solar energy calculations

    An instance of this class may have the following attributes:

        =================== =========================================== ========
        attribute           description                                 type
        =================== =========================================== ========
        lat                 latitude                                    (array)
        lon                 longitude                                   (array)
        tz                  time zone                                   (scalar)
        rdt                 reference datetime object (date_time_obj)   (scalar)
        slope               slope, derivative of DEM                    (array)
        aspect              aspect (north is 0, south is 180)           (array)
        ajd                 absolute julian day                         (scalar)
        ajc                 absolute julian century                     (scalar)
        geomean_long        geometric mean longitude of the sun         (scalar)
        geomean_anom        geometric mean longitude anomaly of the sun (scalar)
        earth_eccent        eccentricity of earths orbit                (scalar)
        sun_eq_of_center    the suns equation of center                 (scalar)
        true_long           true longitude of the sun                   (scalar)
        true_anom           true longitude anomaly of the sun           (scalar)
        app_long            the suns apparent longitude                 (scalar)
        oblique_mean_elip   earth oblique mean ellipse                  (scalar)
        oblique_corr        correction to earths oblique elipse         (scalar)
        right_ascension     suns right ascension angle                  (scalar)
        declination         solar declination angle                     (scalar)
        equation_of_time    equation of time (minutes)                  (scalar)
        hour_angle_sunrise  the hour angle at sunrise                   (array)
        solar_noon          LST of solar noon                           (array)
        sunrise             LST of sunrise time                         (array)
        sunset              LST of sunset time                          (array)
        sunlight            LST fractional days of sunlight             (array)
        true_solar          LST for true solar time                     (array)
        hour_angle          total hour angle                            (array)
        zenith              zenith angle                                (array)
        elevation           elevation angle                             (array)
        azimuth             azimuthal angle                             (array)
        rad_vector          radiation vector (distance in AU)           (scalar)
        earth_distance      earths distance to sun in meters            (scalar)
        norm_irradiance     incident solar energy at earth distance     (scalar)
        =================== =========================================== ========

    Units used by this class unless otherwise labeled

      - angle =     degrees
      - distance =  meters
      - energy =    watts or joules
      - time =      mostly in datetime objects. labeled in most cases.

    """


    def __init__(self, lat, lon, date_time_obj, time_zone = 0,
                         fmt = False, slope = None, aspect = None):
        """
        Initializes critical spatial and temporal information for solar object.
        """

        # empty list of class attributes
        self.ajc                = None          # abs julian century (defined on __init__)
        self.ajd                = None          # abs julian day (defined on __init__)
        self.app_long           = None
        self.atmo_refraction    = None
        self.azimuth            = None
        self.declination        = None
        self.earth_distance     = None
        self.earth_eccent       = None
        self.elevation          = None
        self.elevation_noatmo   = None
        self.equation_of_time   = None
        self.frac_day           = None
        self.geomean_anom       = None
        self.geomean_long       = None
        self.hour_angle         = None
        self.hour_angle_sunrise = None
        self.lat                = lat           # lattitude (E positive)- float
        self.lat_r              = radians(lat)  # lattitude in radians
        self.lon                = lon           # longitude (N positive)- float
        self.lon_r              = radians(lon)  # longitude in radians
        self.norm_irradiance    = None
        self.oblique_corr       = None
        self.oblique_mean_elip  = None
        self.rad_vector         = None
        self.rdt                = None          # reference datetime (defined on __init__)
        self.right_ascension    = None
        self.solar_noon         = None
        self.solar_noon_time    = None
        self.sun_eq_of_center   = None
        self.sunlight           = None
        self.sunlight_time      = None
        self.sunrise            = None
        self.sunrise_time       = None
        self.sunset             = None
        self.sunset_time        = None
        self.true_anom          = None
        self.true_long          = None
        self.true_solar         = None
        self.true_solar_time    = None
        self.tz                 = None          # time zone (defined on __init__)
        self.zenith             = None

        # slope and aspect
        self.slope = slope
        self.aspect = aspect

        # Constants as attributes
        self.sun_surf_rad       = 63156942.6    # radiation at suns surface (W/m^2)
        self.sun_radius         = 695800000.    # radius of the sun in meters
        self.orbital_period     = 365.2563630   # num of days it takes earth to revolve
        self.altitude           = -0.01448623   # altitude of center of solar disk


        # sets up the object with some subfunctions
        self._set_datetime(date_time_obj, fmt, GMT_hour_offset = time_zone)

        # specify if attributes are scalar floats or numpy array floats
        if isinstance(lat, ndarray) and isinstance(lon, ndarray):
            self.is_numpy   = True
        else:
            self.is_numpy   = False

        return


    def _set_datetime(self, date_time_obj, fmt = False, GMT_hour_offset = 0):
        """
        sets the critical time information including absolute julian day/century.
        Accepts datetime objects or a datetime string with format

        :param date_time_obj:   datetime object for time of solar calculations. Will also
                                accept string input with matching value for "fmt" param
        :param fmt:             if date_time_obj is input as a string, fmt allows it to be
                                interpreted
        :param GMT_hour_offset: Number of hours from GMT for timezone of calculation area.
        """

        # if input is datetime_obj set it
        if isinstance(date_time_obj, datetime):
            self.rdt =      date_time_obj
            self.rdt +=     timedelta(hours = -GMT_hour_offset)

        elif isinstance(date_time_obj, str) and isinstance(fmt, str):
            self.rdt =      datetime.strptime(date_time_obj,fmt)
            self.rdt +=     timedelta(hours = -GMT_hour_offset)
        else:
            raise Exception("bad datetime!")

        self.tz = GMT_hour_offset


        # uses the reference day of january 1st 2000
        jan_1st_2000_jd   = 2451545
        jan_1st_2000      = datetime(2000,1,1,12,0,0)

        time_del = self.rdt - jan_1st_2000
        self.ajd = float(jan_1st_2000_jd) + float(time_del.total_seconds())/86400
        self.ajc = (self.ajd - 2451545)/36525.0

        return


    def get_geomean_long(self):
        """ :return geomean_long: geometric mean longitude of the sun"""

        if not self.geomean_long is None:
            return self.geomean_long

        self.geomean_long = (280.46646 + self.ajc * (36000.76983 + self.ajc*0.0003032)) % 360
        return self.geomean_long


    def get_geomean_anom(self):
        """calculates the geometric mean anomoly of the sun"""

        if not self.geomean_anom is None:
            return self.geomean_anom

        self.geomean_anom = (357.52911 + self.ajc * (35999.05029 - 0.0001537 * self.ajc))
        return self.geomean_anom


    def get_earth_eccent(self):
        """ :return earth_eccent: precise eccentricity of earths orbit at referece datetime """

        if not self.earth_eccent is None:
            return self.earth_eccent

        self.earth_eccent = 0.016708634 - self.ajc * (4.2037e-5 + 1.267e-7 * self.ajc)

        return self.earth_eccent


    def get_sun_eq_of_center(self):
        """ :return sun_eq_of_center: the suns equation of center"""

        if not self.sun_eq_of_center is None:
            return self.sun_eq_of_center

        if self.geomean_anom is None:
            self.get_geomean_anom()

        ajc = self.ajc
        gma = radians(self.geomean_anom)

        self.sun_eq_of_center = sin(gma) * (1.914602 - ajc*(0.004817 + 0.000014 * ajc)) + \
                                sin(2*gma) * (0.019993 - 0.000101 * ajc) + \
                                sin(3*gma) * 0.000289

        return self.sun_eq_of_center


    def get_true_long(self):
        """ :return true_long: the tru longitude of the sun"""

        if not self.true_long is None:
            return self.true_long

        if self.geomean_long is None:
            self.get_geomean_long()

        if self.sun_eq_of_center is None:
            self.get_sun_eq_of_center()

        self.true_long = self.geomean_long + self.sun_eq_of_center
        return self.true_long


    def get_app_long(self):
        """ :return app_long: calculates apparent longitude of the sun"""

        if not self.app_long is None:
            return self.app_long

        if self.true_long is None:
            self.get_true_long()

        stl = self.true_long
        ajc = self.ajc

        self.app_long = stl - 0.00569 - 0.00478 * sin(radians(125.04 - 1934.136 * ajc))
        return self.app_long


    def get_oblique_mean_elip(self):
        """ :return oblique_mean_elip: oblique mean elliptic of earth orbit """

        if not self.oblique_mean_elip is None:
            return self.oblique_mean_elip

        ajc = self.ajc

        self.oblique_mean_elip = 23 + (26 + (21.448 - ajc * (46.815 + ajc * (0.00059 - ajc * 0.001813)))/60)/60
        return self.oblique_mean_elip


    def get_oblique_corr(self):
        """ :return oblique_corr:  the oblique correction """

        if not self.oblique_corr is None:
            return self.oblique_corr

        if self.oblique_mean_elip is None:
            self.get_oblique_mean_elip()

        ome = self.oblique_mean_elip
        ajc = self.ajc

        self.oblique_corr = ome + 0.00256 * cos(radians(125.04 - 1934.136 * ajc))
        return self.oblique_corr


    def get_declination(self):
        """ :return declination: solar declination angle at ref_datetime"""

        if not self.declination is None:
            return self.declination

        if self.app_long is None:
            self.get_app_long()

        if self.oblique_corr is None:
            self.get_oblique_corr()

        sal = radians(self.app_long)
        oc  = radians(self.oblique_corr)

        self.declination = degrees(arcsin((sin(oc) * sin(sal))))
        return self.declination


    def get_equation_of_time(self):
        """ :return equation_of_time: the equation of time in minutes """

        if not self.equation_of_time is None:
            return self.equation_of_time

        if self.oblique_corr is None:
            self.get_oblique_corr()

        if self.geomean_long is None:
            self.get_geomean_long()

        if self.geomean_anom is None:
            self.get_geomean_anom()

        if self.earth_eccent is None:
            self.get_earth_eccent()

        oc  = radians(self.oblique_corr)
        gml = radians(self.geomean_long)
        gma = radians(self.geomean_anom)
        ec  = self.earth_eccent

        vary = tan(oc/2)**2

        self.equation_of_time = 4 * degrees(vary * sin(2*gml) - 2 * ec * sin(gma) +
                                4 * ec * vary * sin(gma) * cos(2 * gml) -
                                0.5 * vary * vary * sin(4 * gml) -
                                1.25 * ec * ec * sin(2 * gma))

        return self.equation_of_time


    def get_solar_noon(self):
        """ :return solar_noon: solar noon in (local sidereal time LST)"""

        if not self.solar_noon is None:
            return self.solar_noon

        if self.equation_of_time is None:
            self.get_equation_of_time()

        lon = self.lon
        eot = self.equation_of_time
        tz  = self.tz

        self.solar_noon = (720 - 4 * lon - eot + tz * 60)/1440

        # format this as a time for display purposes (Hours:Minutes:Seconds)
        if self.is_numpy:
            self.solar_noon_time = timedelta(days = self.solar_noon.mean())
        else:
            self.solar_noon_time = timedelta(days = self.solar_noon)

        return self.solar_noon


    def get_true_solar(self):
        """ :return true_solar: true solar time at ref_datetime"""

        if not self.true_solar is None:
            return self.true_solar

        if self.equation_of_time is None:
            self.get_equation_of_time()

        lon = self.lon
        eot = self.equation_of_time


        # turn reference datetime into fractional days
        frac_sec = (self.rdt - datetime(self.rdt.year, self.rdt.month, self.rdt.day)).total_seconds()
        frac_hr  = frac_sec / (60 * 60) + self.tz
        frac_day = frac_hr / 24

        self.frac_day = frac_day

        # now get true solar time
        self.true_solar = (frac_day * 1440 + eot + 4 * lon - 60 * self.tz) % 1440

        # format this as a time for display purposes (Hours:Minutes:Seconds)
        if self.is_numpy:
            self.true_solar_time = timedelta(days = self.true_solar.mean() / (60*24))
        else:
            self.true_solar_time = timedelta(days = self.true_solar / (60*24))

        return self.true_solar


    def get_hour_angle(self):
        """ :return hour_angle: returns hour angle at ref_datetime"""

        if not self.hour_angle is None:
            return self.hour_angle

        if self.true_solar is None:
            self.get_true_solar()

        ts = self.true_solar

        # matrix hour_angle calculations
        if self.is_numpy:
            ha = ts
            ha[ha <= 0] = ha[ha <= 0]/4 + 180
            ha[ha >  0] = ha[ha >  0]/4 - 180
            self.hour_angle = ha

        # scalar hour_angle calculations
        else:
            if ts <= 0:
                self.hour_angle = ts/4 + 180
            else:
                self.hour_angle = ts/4 - 180

        return self.hour_angle


    def get_zenith(self):
        """ :return zenith: returns solar zenith angle at ref_datetime"""

        if not self.zenith is None:
            return self.zenith

        if self.declination is None:
            self.get_declination()

        if self.hour_angle is None:
            self.get_hour_angle()

        d   = radians(self.declination)
        ha  = radians(self.hour_angle)
        lat = self.lat_r

        self.zenith = degrees(arccos(sin(lat) * sin(d) + cos(lat) * cos(d) * cos(ha)))

        return self.zenith


    def get_azimuth(self):
        """ :return azimuth: returns solar azimuth angle at ref_datetime"""

        if not self.azimuth is None:
            return self.azimuth

        if self.declination is None:
            self.get_declination()

        if self.hour_angle is None:
            self.get_hour_angle()

        if self.zenith is None:
            self.get_zenith()

        lat = self.lat_r
        d   = radians(self.declination)
        ha  = radians(self.hour_angle)
        z   = radians(self.zenith)

        # matrix azimuth calculations
        # these equations are hideous monsters, but im not sure how to improve them without adding computational complexity.
        if self.is_numpy:

            az = ha * 0

            az[ha > 0] = (degrees(arccos(((sin(lat[ha > 0]) * cos(z[ha > 0])) - sin(d)) / (cos(lat[ha > 0]) * sin(z[ha > 0])))) + 180) % 360
            az[ha <=0] = (540 - degrees(arccos(((sin(lat[ha <=0]) * cos(z[ha <=0])) -sin(d))/ (cos(lat[ha <=0]) * sin(z[ha <=0]))))) % 360

            self.azimuth = az

        else:
            if ha > 0:
                self.azimuth = (degrees(arccos(((sin(lat) * cos(z)) - sin(d)) / (cos(lat) * sin(z)))) + 180) % 360
            else:
                self.azimuth = (540 - degrees(arccos(((sin(lat) * cos(z)) -sin(d))/ (cos(lat) * sin(z))))) % 360

        return self.azimuth


# In[7]:


def ABIangle2LonLat(x, y, H, req, rpol, lon_0_deg):
    '''This function finds the latitude and longitude (degrees) of point P
    given x and y, the ABI elevation and scanning angle (radians)'''

    # Intermediate calculations
    a = np.sin(x)**2 + ( np.cos(x)**2 * ( np.cos(y)**2 + ( req**2 / rpol**2 ) * np.sin(y)**2 ) )
    b = -2 * H * np.cos(x) * np.cos(y)
    c = H**2 - req**2

    rs = ( -b - np.sqrt( b**2 - 4*a*c ) ) / ( 2 * a ) # distance from satellite point (S) to P

    Sx = rs * np.cos(x) * np.cos(y)
    Sy = -rs * np.sin(x)
    Sz = rs * np.cos(x) * np.sin(y)

    # Calculate lat and lon
    lat = np.arctan( ( req**2 / rpol**2 ) * ( Sz / np.sqrt( ( H - Sx )**2 + Sy**2 ) ) )
    lat = np.degrees(lat) #*
    lon = lon_0_deg - np.degrees( np.arctan( Sy / ( H - Sx )) )

    return (lon,lat)


# In[8]:


def find_current_day(bucket_name,file_prefix): # find the most recent day available on google-cloud
    storage_client = storage.Client.create_anonymous_client()
    blobs = storage_client.list_blobs(bucket_name, prefix = file_prefix)
    available_doy = []
    for blob in blobs:
        path = blob.name
        doy = path.split('/')[2]  
        if doy not in available_doy:
            available_doy.append(doy)
        
    current_day = max(available_doy)
    return(current_day)


# In[9]:


def list_blobs(bucket_name,file_prefix):
    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.get_bucket(bucket_name)
    blobs = storage_client.list_blobs(bucket_name, prefix = file_prefix) # list of files from that bucket & prefix

    blob_list = list(blobs)
    count = len(blob_list)
    return(count,blob_list,bucket) # number of files, list of files, bucket name


# In[10]:


#def get_images(product, year, day, hour, bucket_name):
#    folder_path = 'ABI-L2-{product}/{year}/{day}/{hour}/'.format(product = product, year = year,
#                                                                 day = day, hour = hour)

#    numBlobs, productBlobs, bucket = list_blobs(bucket_name, folder_path)
#    image_list = []

#    for i in range(0, numBlobs):
#        theBlob = productBlobs[i]
#        location, blob_filename = os.path.split(theBlob.name)
#        print(blob_filename)
#        url = "https://storage.googleapis.com/gcp-public-data-goes-16/" + theBlob.name
#        image_list.append(url)

#   return(image_list)

def image_lists(product, year, day, hour, local_path, bucket_name):
    folder_path = 'ABI-L2-{product}/{year}/{day}/{hour}/'.format(product = product, year = year, 
                                                                 day = day, hour = hour)
    local_folder = os.path.join(local_path, folder_path)
    if not os.path.exists(local_folder):
        os.makedirs(local_folder)
    numBlobs, productBlobs, bucket = list_blobs(bucket_name, folder_path)
    image_list = []
    
    for i in range(0, numBlobs):
        theBlob = productBlobs[i]
        location, blob_filename = os.path.split(theBlob.name)
        img_path = local_folder + blob_filename
        blobName = storage.Blob(theBlob.name, bucket)
        blobName.download_to_filename(img_path)
        image_list.append(img_path)
        
    return(image_list)

# In[11]:

rd_path = '/mnt/researchdrive/pcstoy/ALIVE/'+year+'/'+str(doy).zfill(3)+'/'
os.makedirs(rd_path, exist_ok=True)

# Function to process a single image
def process_image(i, img):
    print("Image:", i + 1)

    #brfImg = rxr.open_rasterio(img, mask_and_scale=True)
    fp = fsspec.open(img)
    brfImg = xr.open_dataset(fp.open(), engine="h5netcdf", mask_and_scale=True)

    #time
    time = brfImg.t.item()
    dt = datetime.utcfromtimestamp(time/1000000000).strftime('%Y-%m-%d %H:%M:%S')

    #brfDQF = brfImg['DQF'].data
    #BitMask_0 = 1 << 0
    #BRFmask = (np.bitwise_and(brfDQF.astype(int), BitMask_0)) == 0

    # Make DSR
    cmiFile = CMIimages[i]
    fp2 = fsspec.open(cmiFile)
    cmiImg = xr.open_dataset(fp2.open(), engine="h5netcdf", mask_and_scale=True)

    DSRpredictArray = np.zeros([1500, 2500, 8])
    cmiBands = [1, 2, 4, 6, 12, 15]
    for j in range(0, 5):
        b_num = str(cmiBands[j]).zfill(2)
        cmi_band = cmiImg['CMI_C{band}'.format(band=b_num)]
        DSRpredictArray[:, :, j] = cmi_band.data
    SZA_SAA = calcSolar(cmiImg, dt)
    SZAarray = SZA_SAA[0]
    SAAarray = SZA_SAA[1]
    reshapedSZA = np.swapaxes(SZAarray,0,1)
    reshapedSAA = np.swapaxes(SAAarray,0,1)
    DSRpredictArray[:, :, 6] = reshapedSZA
    DSRpredictArray[:, :, 7] = reshapedSAA
    # DSR modeling
    print("DSR prediction array created")
    reshapedArr = np.reshape(DSRpredictArray, (DSRpredictArray.shape[0] * DSRpredictArray.shape[1], DSRpredictArray.shape[2]))
    maskedArr = np.nan_to_num(reshapedArr)
    dsrArr = DSRmodel.predict(maskedArr)
    dsrImg = dsrArr.reshape(DSRpredictArray[:, :, 0].shape)
    print("DSR image predicted")

    # Add BRF bands as predictors
    GPPpredictArray = np.zeros([1500, 2500, 8])  # set up array with 7 bands
    brfBands = ['1', '2', '3', '5', '6']
    for k in range(0, 5):
        b = brfBands[k]
        brfBand = brfImg['BRF{band}'.format(band=b)]
        maskedBands = brfBand
        GPPpredictArray[:, :, k] = maskedBands.data
    BRF2 = GPPpredictArray[:, :, 1]
    BRF3 = GPPpredictArray[:, :, 2]
    BRF6 = GPPpredictArray[:, :, 4]

    # Add DSR and indices as predictors
    DSR = dsrImg
    NIRvP = ((BRF3 - BRF2) / (BRF3 + BRF2)) * BRF3 * DSR
    SWIRvP = ((BRF3 - BRF2 - BRF6) / (BRF3 + BRF2 + BRF6)) * BRF3 * DSR
    GPPpredictArray[:, :, 5] = DSR
    GPPpredictArray[:, :, 6] = NIRvP
    GPPpredictArray[:, :, 7] = SWIRvP
    print("GPP prediction array created")

    # GPP modeling
    reshapedArr = np.reshape(GPPpredictArray, (GPPpredictArray.shape[0] * GPPpredictArray.shape[1], GPPpredictArray.shape[2]))
    maskedArr = np.nan_to_num(reshapedArr)
    gppArr = GPPmodel.predict(maskedArr)
    gppImg = gppArr.reshape(GPPpredictArray[:, :, 0].shape)
    print("GPP image predicted")

    brfImg['ALIVE'] = (('y', 'x'), gppImg)
    brfImg['ALIVE'].rio.to_raster(rd_path + "ALIVE_{day}.tif".format(day = day_start)) # research drive
    return (gppImg, dt)

# In[12]:


def calcSolar (ds, dt):
    x_rad = ds.x
    y_rad = ds.y
    lon, lat = ABIangle2LonLat(x_rad, y_rad,
                              35786023.0 + ds.goes_imager_projection.semi_major_axis,
                              ds.goes_imager_projection.semi_major_axis,
                              ds.goes_imager_projection.semi_minor_axis,
                              -75.0)
    lats = np.array(lat)
    lons = np.array(lon)

    latArray = np.nan_to_num(lats, nan = 0)
    lonArray = np.nan_to_num(lons, nan = 0)

    # from https://github.com/NASA-DEVELOP/dnppy/blob/master/dnppy/solar/README.md
    if __name__ == "__main__":
        my_datestamp   = dt                     # date stamp
        my_fmt         = "%Y-%m-%d %H:%M:%S"    # datestamp format
        my_tz          = 0                      # timezone (GMT/UTC) offset
        my_lat = latArray                       # lat (N positive)
        my_lon = lonArray                       # lon (E positive)

        sc  = solar(my_lat, my_lon, my_datestamp, my_tz, my_fmt)
        SZA = sc.get_zenith()
        SAA = sc.get_azimuth()
    return SZA, SAA


# In[13]:


# os.makedirs('/Users/Sophie/Desktop/alive_test2/', exist_ok=True) # local
# local_path = '/Users/Sophie/Desktop/alive_test2/'
os.makedirs('/home/shoffman/alive_test/', exist_ok=True) # server
local_path = '/home/shoffman/alive_test/'

year = str(today).split('-')[0]
year_str = str(year)

bucket_name = 'gcp-public-data-goes-16'
brfc_prefix = 'ABI-L2-BRFC/'+year

day_start = int(find_current_day(bucket_name,brfc_prefix)) - 1
day_stop = int(find_current_day(bucket_name,brfc_prefix))

BRFimages = []
CMIimages = []

import time
start_time = time.time()
for day in range(day_start,day_stop+1,1):
    if day == day_start:
        day_start_str = str(day_start).zfill(3)
        print('Year: ',year,' Day: ',day_start_str)
        
        hour_start = 10
        hour_stop = 24
        for hour in range(hour_start,hour_stop,1):
            hour_str = str(hour).zfill(2)
            print('Hour', hour)

            #BRF_img_list = get_images('BRFC', year_str, day_start_str, hour_str, bucket_name)
            #CMI_img_list = get_images('MCMIPC', year_str, day_start_str, hour_str, bucket_name)

            BRF_img_list = image_lists('BRFC', year_str, day_start_str, hour_str, local_path,  bucket_name)
            CMI_img_list = image_lists('MCMIPC', year_str, day_start_str, hour_str, local_path,  bucket_name)

            BRFimages.extend(BRF_img_list)
            CMIimages.extend(CMI_img_list)
    
    elif day == day_stop:
        day_stop_str = str(day_stop).zfill(3)
        print('Year: ',year,' Day: ',day_stop_str)
        
        hour_start = 0
        hour_stop = 2
        for hour in range(hour_start,hour_stop,1):
            hour_str = str(hour).zfill(2)
            print('Hour', hour)

            #BRF_img_list = get_images('BRFC', year_str, day_stop_str, hour_str, bucket_name)
            #CMI_img_list = get_images('MCMIPC', year_str, day_stop_str, hour_str, bucket_name)
            BRF_img_list = image_lists('BRFC', year_str, day_stop_str, hour_str, local_path,  bucket_name)
            CMI_img_list = image_lists('MCMIPC', year_str, day_stop_str, hour_str, local_path,  bucket_name)

            BRFimages.extend(BRF_img_list)
            CMIimages.extend(CMI_img_list)

print('IMAGE DOWNLOAD')
print(float(time.time() - start_time) / 60 , 'minutes')


# In[14]:


print(len(CMIimages))
print(len(BRFimages))


# In[ ]:

#local_save_path = local_path+year_str+'/'+str(day_start).zfill(3)+'/'
#os.makedirs(local_save_path, exist_ok=True)

# In[15]:


start_time = time.time()
aliveImages = []
aliveTimeArr = []
aliveTuples = []

# Run ALIVE preprocessing and modeling
for i, img in enumerate(BRFimages):
    aliveTuple = process_image(i, img)
    aliveTuples.append(aliveTuple)

aliveTuples.sort(key = lambda x: x[1])   # Ensure images are sorted by time
aliveImages = list(zip(*aliveTuples))[0]
aliveTimeArr = list(zip(*aliveTuples))[1]

print('IMAGE ARRAY PREDICTIONS')
print(float(time.time() - start_time) / 3600 , 'hours')


# In[17]:

start_time = time.time()
# Make ALIVE animation for day
frames = []
plt.rcParams['figure.figsize'] = [15, 15]
plt.rcParams['font.size'] = 22
fig, ax = plt.subplots()
plt.title('ALIVE GPP (02-21 Model): ' + str(day_start) + '-' + str(year))
plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
for i, img in enumerate(aliveImages):
    im = ax.imshow(img, cmap=plt.cm.inferno, vmin = 0, vmax = 35, animated=True,)
    t = ax.annotate('{0} UTC'.format(aliveTimeArr[i]),(0.007,0.96), xycoords ='axes fraction', color = 'white', fontsize=18)
    oneframe = [im,t]
    frames.append(oneframe)
cb = fig.colorbar(im, label = r'GPP $({\mu}{mol}$ ${CO_{2}}$ $m^{-2} s^{-1})$', orientation = 'horizontal', fraction=0.046, pad=0.04)
ani = animation.ArtistAnimation(fig, frames, interval=200, blit=True)
ani.save(rd_path + 'ALIVE_GPP_{day}_{year}.mp4'.format(day=day_start,year= year), writer='ffmpeg') # research drive
ani.save('/home/shoffman/website_github/alive/images/ALIVE_GPP.mp4', writer='ffmpeg') # website
#ani.save('/home/shoffman/website_github/alive/images/daily-loops/' + 'ALIVE_GPP_{day}_{year}.mp4'.format(day=day_start,year= year), writer='ffmpeg') # github

print('VIDEO')
print(float(time.time() - start_time) / 60 , 'minutes')

