#!/usr/bin/env python
# coding: utf-8

# In[79]:

import warnings
warnings.filterwarnings("ignore")

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

from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")

today = date.today()
doy = date.today().timetuple().tm_yday
print(today,current_time)
print('Day of year:',doy)

# In[80]:

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


# In[81]:


def list_blobs(bucket_name,file_prefix):
    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.get_bucket(bucket_name)
    blobs = storage_client.list_blobs(bucket_name, prefix = file_prefix) # list of files from that bucket & prefix

    blob_list = list(blobs)
    count = len(blob_list)
    return(count,blob_list,bucket) # number of files, list of files, bucket name


# In[82]:


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


# In[83]:


os.makedirs('/path/to/folder/', exist_ok=True)
local_path = '/path/to/folder/'

year = str(today).split('-')[0]
year_str = str(year)

bucket_name = 'gcp-public-data-goes-16'
brfc_prefix = 'ABI-L2-BRFC/'+year
dsrc_prefix = 'ABI-L2-DSRC/'+year

day_start = int(find_current_day(bucket_name,brfc_prefix)) - 1
day_stop = int(find_current_day(bucket_name,brfc_prefix))

BRFimages = []
DSRimages = []

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

            BRF_img_list = image_lists('BRFC', year_str, day_start_str, hour_str, local_path, bucket_name)
            DSR_img_list = image_lists('DSRC', year_str, day_start_str, hour_str, local_path, bucket_name)

            BRFimages.extend(BRF_img_list)
            DSRimages.extend(DSR_img_list)
    elif day == day_stop:
        day_stop_str = str(day_stop).zfill(3)
        print('Year: ',year,' Day: ',day_stop_str)
        
        hour_start = 0
        hour_stop = 2
        for hour in range(hour_start,hour_stop,1):
            hour_str = str(hour).zfill(2)
            print('Hour', hour)

            BRF_img_list = image_lists('BRFC', year_str, day_stop_str, hour_str, local_path, bucket_name)
            DSR_img_list = image_lists('DSRC', year_str, day_stop_str, hour_str, local_path, bucket_name)

            BRFimages.extend(BRF_img_list)
            DSRimages.extend(DSR_img_list)
        

print('GOES image download --')
print(float(time.time() - start_time) / 60 , 'minutes')


# In[84]:


print(len(DSRimages))
# for im in DSRimages:
#     print(im)
print(len(BRFimages))


# In[85]:


start_time = time.time()
# Mask and reproject DSR images
brfImg = rxr.open_rasterio(BRFimages[0],mask_and_scale = True) #Open BRF image to get ABI Fied Grid projection
DSRtimeseries = []
for i, img in enumerate(DSRimages):
    dsrImg = rxr.open_rasterio(img, mask_and_scale = True)
    dsr_reproj = dsrImg.rio.reproject_match(brfImg)
    dsrData = dsr_reproj['DSR'].data[0,:,:]
    DSRtimeseries.append(dsrData)
    
# Interpolate new DSR images
nimages = len(BRFimages)
nhours = len(DSRimages)
x = np.arange(0,nhours,1)
y = np.array(DSRtimeseries)
y_valid = np.nan_to_num(y)
print(len(x),len(y),len(y_valid))
CS = CubicSpline(x, y_valid,0)

new_times = np.linspace(0, 24, nimages)  # Adjust as needed
newDSR = CS(new_times)
DSRimages = list(newDSR)

print('Interpolate DSR --')
print(float(time.time() - start_time) / 60 , 'minutes')


# In[87]:

# ML model below ------
loaded_model = joblib.load('/path/to/model/model.joblib')
print(loaded_model.predict([[0.1, 0.2, 0.4, 0.1, 0.1, 200.1, 0.2, 0.3]]))

# In[88]:

# Function to process a single image
def process_image(i, img):
    print("Image:", i + 1)

    brfImg = rxr.open_rasterio(img, mask_and_scale=True)
    brfDQF = brfImg['DQF'].data
    brfTime = pd.to_datetime(brfImg.attrs['time_coverage_start'])
    BitMask_0 = 1 << 0
    BRFmask = (np.bitwise_and(brfDQF.astype(int), BitMask_0)) == 0

    # Add BRF bands as predictors
    predictArray = np.zeros([1500, 2500, 8])  # set up array with 7 bands
    brfBands = ['1', '2', '3', '5', '6']
    for j in range(0, 5):
        b = brfBands[j]
        brfBand = brfImg['BRF{band}'.format(band=b)]
        maskedBands = brfBand
        predictArray[:, :, j] = maskedBands.data
    BRF2 = predictArray[:, :, 1]
    BRF3 = predictArray[:, :, 2]
    BRF6 = predictArray[:, :, 4]

    # Add DSR and indices as predictors
    DSR = DSRimages[i]
    NIRv = ((BRF3 - BRF2) / (BRF3 + BRF2)) * BRF3
    NIRvP = NIRv * DSR
    SWIRv = ((BRF3 - BRF2 - BRF6) / (BRF3 + BRF2 + BRF6)) * BRF3
    SWIRvP = SWIRv * DSR

    predictArray[:, :, 5] = DSR
    predictArray[:, :, 6] = NIRvP
    predictArray[:, :, 7] = SWIRvP
    print("Prediction array created")

    # ALIVE modeling
    reshapedArr = np.reshape(predictArray, (predictArray.shape[0] * predictArray.shape[1], predictArray.shape[2]))
    maskedArr = np.nan_to_num(reshapedArr)
    aliveArr = loaded_model.predict(maskedArr)
    aliveImg = aliveArr.reshape(predictArray[:, :, 0].shape)
    print("ALIVE image predicted")

    brfImg['ALIVE'] = (('y', 'x'), aliveImg)
    brfImg['ALIVE'].rio.to_raster(local_path + "ALIVE.tif")

    return (aliveImg, brfTime)


# In[89]:

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

print('Preprocessing and modeling --')
print(float(time.time() - start_time) / 3600 , 'hours')

# In[90]:

local_save_path = local_path+year_str+'/'+str(day_start).zfill(3)+'/' # local
os.makedirs(local_save_path, exist_ok=True)

# In[97]:


start_time = time.time()
# Make ALIVE animation for day
frames = []
plt.rcParams['figure.figsize'] = [15, 15]
plt.rcParams['font.size'] = 22
fig, ax = plt.subplots()
plt.title('ALIVE GPP (Fs2 Model): ' +str(day_start)+'/'+year_str)
plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
plt.rcParams['animation.writer'] = 'ffmpeg'
for i, img in enumerate(aliveImages):
    im = ax.imshow(img, cmap=plt.cm.inferno, vmin = 0, vmax = 35, animated=True,)
    t = ax.annotate('{0} UTC'.format(aliveTimeArr[i].time()),(0.007,0.96), xycoords ='axes fraction', color = 'white', fontsize=18)
    oneframe = [im,t]
    frames.append(oneframe)
cb = fig.colorbar(im, label = r'GPP $({\mu}{mol}$ ${CO_{2}}$ $m^{-2} s^{-1})$', orientation = 'horizontal', fraction=0.046, pad=0.04)
ani = animation.ArtistAnimation(fig, frames, interval=200, blit=True)
ani.save(local_save_path + 'ALIVE_GPP_{day}_{year}.mp4'.format(day=day_start,year= year), writer='ffmpeg')

print('Animation for day --')
print(float(time.time() - start_time) / 60 , 'minutes')


# In[92]:


start_time = time.time()
# Make daily sum of ALIVE GPP
oneDay = np.dstack(aliveImages)
daySum = np.sum(oneDay,axis = 2)
brfImg['ALIVE_daily'] = (('y', 'x'), daySum)
brfImg['ALIVE_daily'].rio.to_raster(local_save_path + "ALIVE_daily_{day}.tif".format(day = day_start))

print('Daily sum of GPP --')
print(float(time.time() - start_time) / 60 , 'minutes')


