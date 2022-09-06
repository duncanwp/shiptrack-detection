#---------------------------------------------------------------
# TOP LEVEL CODE
#---------------------------------------------------------------
# PROCESS_TRACK_IMAGES
#
# Description
# General program to read MODIS reflectance and shiptrack hand-logged
# files with some basic plots to illustrate where ship tracks are located
# in the satellite granule. Uses PyTroll to plot/read the MODIS image.
#
# Notes: this code currently crops 250 pixels off the edge of each
#        image. Ship tracks are logged based on the lower left corner
#        starting at x=0 and y=0. The ML algorithm requires the top-left
#        corner to start at x=0 and y=0 therefore we tranform the y-coord
#        for bounding box locations.
#
# Output
# 1) Plots of NIR composite images for each MODIS granule
#       path ---> /images
# 2) Plots of the bounding boxes (used as a sanity check)
#       path ---> /images_bbox/
# 3) Text File: contains the bounding boxes according to DIGITS
#       path --> /labels/
#    (FORMAT: see 
#    https://github.com/NVIDIA/DIGITS/blob/master/digits/extensions/data/objectDetection/README.md)
#
# Example
# python2.7 -i process_track_images.py
#
# History
# 11/12/18, MC: upload initial version of the code to the repo
#---------------------------------------------------------------

#---------------------------------------------------------------
# Libraries
#---------------------------------------------------------------
from subroutines_track_images import *
import datetime
import numpy as np
import os,sys,glob
from netCDF4 import Dataset
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw
import numpy.ma as ma
import pdb
from satpy import Scene
from satpy.composites import GenericCompositor, DifferenceCompositor, NIREmissivePartFromReflectance
from satpy.dataset import DatasetID
from satpy.writers import to_image
from pyhdf.SD import SD, SDC
from multiprocessing import Pool, Value

#Crop 250 pixels off each edge of the MODIS granule
## DONT CROP
cropped_area = 0


#---------------------------------------------------------------
# Paths
#---------------------------------------------------------------
#Ship Track Hand-logged Files

osu_tracks = False
if osu_tracks:
    path_track_root = '/group_workspaces/jasmin2/aopp/mchristensen/shiptrack/shiptrack_logged_files/osu_ship_log/tfiles/'
else:
    path_track_root = '/group_workspaces/jasmin2/aopp/mchristensen/shiptrack/shiptrack_logged_files/combined/'

path_track_root = '/gws/nopw/j04/aopp/mchristensen/shiptrack/shiptrack_logged_files/combined_v2/'

#MODIS Files
path_modis_root = '/gws/nopw/j04/eo_shared_data_vol1/satellite/modis/modis_c61/'

#Name of the experiment (here we are calling it crop_partial)
expName = 'crop'
expName = 'crop_partial'
expName = 'nocrop_combined_points'
expName = 'nocrop_combined_points_filled_in'
expName = 'nocrop_combined_points_typed_composite_niremi'
expName = 'nocrop_combined_points_typed_composite_367'
expName = 'nocrop_combined_points_typed_true_color'
expName = 'nocrop_combined_points_typed_composite_367_terra'

#Output Path (user-defined)
root_output = '/gws/nopw/j04/impala/public/dwatsonparris/shiptrack'

#images directory
path_output_images = root_output+'/'+expName+'/images/'
ferr = file_mkdir(path_output_images)

#bounding box directory
path_output_images_bbox = root_output+'/'+expName+'/images_bbox/'
ferr = file_mkdir(path_output_images_bbox)

path_output_points = root_output+'/'+expName+'/points/'
ferr = file_mkdir(path_output_points)

#create labels based on full extent of bounding box
path_output_labels = root_output+'/'+expName+'/labels/'
ferr = file_mkdir(path_output_labels)

#---------------------------------------------------------------
# Fetch Ship Track Files
#---------------------------------------------------------------
trackfiles = file_search_tracks( path_track_root )
tfiles = trackfiles['tfiles']
lfiles = trackfiles['lfiles']
fct = len(tfiles)

CT = None

def init(args):
    """ Store the counter ready for later use """
    global CT
    CT = args


def main(i):
    global CT

    # Read track lat/lon locations
    fileInfo = os.path.split(lfiles[i])
    lfilename = fileInfo[0]+'/'+fileInfo[1]
    track_geo = read_osu_shiptrack_file(lfilename)

    # Read track locations from file
    fileInfo = os.path.split(tfiles[i])
    tfilepath = fileInfo[0]
    tfile = fileInfo[1]
    tfilename = tfilepath+'/'+tfile
    track_points = read_osu_shiptrack_file(tfilename)

    # Get the track source
    tSource = tfile[1:4]

    print(tfilename)
    print(tSource)

    # Fetch corresponding MODIS granule
    mtype=''
    if osu_tracks:
        if 'aqua' in fileInfo[0]:
            mtype = 'myd'
        elif 'terra' in fileInfo[0]:
            mtype = 'mod'
        else:
            print("ERROR - platform type not found in filepath")
        
        ed=len(tfile)
        YYYY = tfile[ed-15:ed-11]
        DDD  = tfile[ed-11:ed-8]
        HHHH = tfile[ed-8:ed-4]

    else:
        if "MOD" in tfile:
            mtype = 'mod'
        if 'MYD' in tfile:
            mtype = 'myd'

        ed=len(tfile)
        YYYY = tfile[ed-16:ed-12]
        DDD  = tfile[ed-12:ed-9]
        HHHH = tfile[ed-8:ed-4]

    print(tfile,'  ',mtype,'  ',YYYY,'  ',DDD,'  ',HHHH)   
    # Now fetch MODIS file for selected instrument and time
    file02 = glob.glob( path_modis_root + mtype+'021km/' + YYYY + '/' + DDD + '/' + '*.A' + YYYY + DDD + '.'+HHHH+'*.hdf') # calibrated radiances
    file03 = glob.glob( path_modis_root + mtype+'03/' + YYYY + '/' + DDD + '/' + '*.A' + YYYY + DDD + '.'+HHHH+'*.hdf')   # geolocation
    #file04 = glob.glob( path_modis_root + mtype+'04_l2/' + YYYY + '/' + DDD + '/' + '*.A' + YYYY + DDD + '.'+HHHH+'*.hdf') # aerosol
    #file06 = glob.glob( path_modis_root + mtype+'06_l2/' + YYYY + '/' + DDD + '/' + '*.A' + YYYY + DDD + '.'+HHHH+'*.hdf') # cloud

    # All MODIS L2 files must exist to proceed
    #### ONLY INCLUDE TERRA FILES (AQUA BAND 6 IS BROKEN)
    if (len(file02) > 0) and (len(file03) > 0) and (mtype=='mod'): #and len(file04) > 0 and len(file06) > 0:
        file02 = file02[0]
        file03 = file03[0]
        #file04 = file04[0]
        #file06 = file06[0]

        # Read MODIS Attribute Data (to extract size of image)
        file = SD(file02, SDC.READ)
        sds_obj = file.select('EV_500_Aggr1km_RefSB')
        xN = int((sds_obj.dimensions(0))['Max_EV_frames:MODIS_SWATH_Type_L1B'])
        yN = int((sds_obj.dimensions(0))['10*nscans:MODIS_SWATH_Type_L1B'])

        #Calculate bounding boxes
        trainData = track_to_bbox(track_points,xN,yN,cropped_area,'full')

        #Extract flags from trainData (array of dictionaries)
        flags = []
        for iTRK in range(len(trainData)):
            flags.append( (trainData[iTRK])['flag'] )

        #Process image and label files if at least 1 ship track satisfies the cropped region condition
        if np.sum(flags) < len(trainData):
        
            with CT.get_lock():
                CT_value = CT.value
                CT.value += 1

            #Create image
            pngFile = path_output_images + str(CT_value).zfill(4)+'.png'
            ## Don't create images
            if os.path.isfile(pngFile):
                print('exists: '+pngFile)
            else:
                global_scene = (Scene(reader="modis_l1b", filenames=[file02,file03]))
                
                # Band 367 - Terra only
                global_scene.load(['3', '6', '7'], resolution=1000)
                compositor = GenericCompositor("rgb")
                composite = compositor([global_scene['3'],global_scene['6'],global_scene['7']])
                global_scene = global_scene[0:yN,cropped_area:xN-cropped_area]
                img = to_image(composite)
                img.stretch_hist_equalize("linear")
                
                # True Color
                #global_scene.load(['true_color', '7'], resolution=1000)
                #global_scene = global_scene[0:yN,cropped_area:xN-cropped_area]
                #img = to_image(global_scene['true_color'])
                
                # BTD (e.g. Yuan 2019) - Nighttime only
                #global_scene.load(['31'], resolution=1000)
                #global_scene.load([DatasetID(name='20', modifiers=('nir_emissive',))])
                #compositor = DifferenceCompositor("diffcomp")
                #composite = compositor([global_scene['20'], global_scene['31']])
                #global_scene = global_scene[0:yN,cropped_area:xN-cropped_area]
                #img = to_image(composite)
                #img.stretch_hist_equalize("linear")

                img.save(pngFile)
                print(pngFile)

            #Create labels - valid (within cropped area) training data only
            txtFile = path_output_labels + str(CT_value).zfill(4)+'.txt'
            text_file = open(txtFile, "w")
            for iTRK in range(len(trainData)):
                if (trainData[iTRK])['flag'] == 0:
                    text_file.write(tSource+' '+
                                    (trainData[iTRK])['truncated']+' '+
                                    (trainData[iTRK])['occluded']+' '+
                                    (trainData[iTRK])['alpha']+' '+
                                    (trainData[iTRK])['bbox_left']+' '+
                                    (trainData[iTRK])['bbox_top']+' '+
                                    (trainData[iTRK])['bbox_right']+' '+
                                    (trainData[iTRK])['bbox_bottom']+' '+
                                    (trainData[iTRK])['dimensions_height']+' '+
                                    (trainData[iTRK])['dimensions_width']+' '+
                                    (trainData[iTRK])['dimensions_length']+' '+
                                    (trainData[iTRK])['location_x']+' '+
                                    (trainData[iTRK])['location_y']+' '+
                                    (trainData[iTRK])['location_z']+' '+
                                    (trainData[iTRK])['rotation_y']+' '+
                                    (trainData[iTRK])['score']+"\n")
            text_file.close()

            #Create points - valid (within cropped area) training data only
            txtFile = path_output_points + str(CT_value).zfill(4)+'.txt'
            text_file = open(txtFile, "w")
            for iTRK in range(len(trainData)):
                if (trainData[iTRK])['flag'] == 0:
                    text_file.write(tSource + " " + " ".join(str(xp)+" "+str(yp) for xp, yp in zip(track_points['xpt'][iTRK], track_points['ypt'][iTRK]))+'\n')
            text_file.close()

            #read bounding boxes
            text_file = open(txtFile, "r")
            lines = text_file.readlines()
            text_file.close()
            pngFileBBox = path_output_images_bbox + str(CT_value).zfill(4)+'.png'

            # Don't make bounding box examples
            if True or os.path.isfile(pngFileBBox):
                print('exists: '+pngFileBBox)
            else:

                #plot bounding boxes over the top of the satellite image
                pil_im = img.pil_image() #convert XRIMAGE to PIL
                draw = ImageDraw.Draw(pil_im)

                #loop over each ship track
                for j in range(len(lines)):
                    line = lines[j].split()
                    xL = int(line[4])
                    yT = int(line[5])
                    xR = int(line[6])
                    yB = int(line[7])

                    #draw bounding boxes over image
                    truncateFlag = int(line[1])
                    if truncateFlag == 0:
                        col = "black"
                    if truncateFlag == 1:
                        col = "white"

                    print(xL,yT,xR,yB,col)
                    draw.line((xL,yT, xR, yT), fill=col, width=5)
                    draw.line((xL,yB, xR, yB), fill=col, width=5)
                    draw.line((xL,yB, xL, yT), fill=col, width=5)
                    draw.line((xR,yB, xR, yT), fill=col, width=5)
                pil_im.save(pngFileBBox)

        else:
            print('outside of cropped area')

    else:
        print('missing: ',mtype+'021km/' + YYYY + '/' + DDD + '/' + '*.A' + YYYY + DDD + '.'+HHHH+'*.hdf')


p=Pool(initializer=init, initargs=(Value('i', 0),), processes=4)
# Loop over each track file
p.map(main, range(fct))

#tarball directory for easier downloading
os.system("tar -czvf "+root_output+"/"+expName+".tar.gz"+" "+root_output+'/'+expName+"/")
