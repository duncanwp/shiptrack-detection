#-----------------------------------------------------------------
#Module Containing all Subroutines
#-----------------------------------------------------------------
import datetime
import os
import numpy as np
import numpy.ma as ma

def get_modis_path(filestub, dtype='06', path_modis_root=''):
    """
    MYD03.A2009325.1900
    """

    mtype = filestub[0:3]
    YYYY = filestub[10:14]
    DDD = filestub[14:17]

    date = datetime.datetime.strptime(YYYY+DDD, '%Y%j')
    month = "{:02d}".format(date.month)
    day = "{:02d}".format(date.day)

    HHHH = filestub[18:22]

    print(filestub,'  ',mtype,'  ',YYYY,'  ',DDD,'  ',HHHH)
    # Now fetch MODIS file for selected instrument and time
    #file02 = glob.glob( path_modis_root + mtype+'021km/' + YYYY + '/' + DDD + '/' + '*.A' + YYYY + DDD + '.'+HHHH+'*.hdf') # calibrated radiances
    print(path_modis_root + mtype+'03/collection61/' + YYYY + '/' + month + '/' + day + '/' + '*.A' + YYYY + DDD + '.'+HHHH+'*.hdf')

    #return f"{path_modis_root}/{mtype}{dtype}_L2/collection61/{YYYY}/{DDD}/{mtype}{dtype}_L2.A{YYYY}{DDD}.{HHHH}*.hdf"
    return f"{path_modis_root}/{mtype}{dtype}_L2/collection61/{YYYY}/{month}/{day}/{mtype}{dtype}_L2.A{YYYY}{DDD}.{HHHH}*.hdf"


def file_mkdir(dirpath):
    if not os.path.exists(os.path.dirname(dirpath)):
        try:
            os.makedirs(os.path.dirname(dirpath))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    errval=0
    return errval


#Generic file search routine
def file_search(path,suffix,startswith):
    list_of_files = []
    for root, dirs, files in os.walk(path): 
        for file in files:
            if file.endswith(suffix) and file[0] == startswith:
                file_path = (os.path.join(root, file))
                list_of_files.append(file_path) # writes each full file path to the empty list defined above
    return list_of_files

#Specific file search routine to output both t & l track stat files
def file_search_tracks(path):
    tfiles = file_search( path, '.dat', 't')
    lfiles = file_search( path, '.dat', 'l')

    nfiles=0
    if len(tfiles) <= len(lfiles):
        nfiles = len(tfiles)
        file_set = tfiles
    if len(lfiles) < len(tfiles):
        nfiles = len(lfiles)
        file_set = lfiles

    #Find where they match
    mergedLfiles = []
    mergedTfiles = []
    for i in range(nfiles):
        fileInfo = os.path.split(file_set[i])
        tmpfilepath = fileInfo[0]
        tmpfile = fileInfo[1]
        ed=len(tmpfile)
        Tindex=((np.where(np.array(tfiles) == tmpfilepath+'/t'+tmpfile[1:ed]))[0])[0]
        Lindex=((np.where(np.array(lfiles) == tmpfilepath+'/l'+tmpfile[1:ed]))[0])[0]
        tmpLfile = lfiles[Lindex]
        tmpTfile = tfiles[Tindex]
        mergedLfiles.append(tmpLfile)
        mergedTfiles.append(tmpTfile)

    list_of_files = { 'tfiles':mergedTfiles, 'lfiles':mergedLfiles, 'names':['tfiles contain the x and y positions of track locations within MODIS image coordinates','lfiles contain longitude and latitude positions of track locations within MODIS lat/lon coordinates'] }
    return list_of_files

#Function to extract track positions from ascii file
def read_osu_shiptrack_file(tfilename):
    f = open(tfilename,'r')
    lines = f.readlines()
    f.close()
    #Number of ship tracks in granule
    tnum = int((lines[1].split())[0])
    all_xvals = []
    all_yvals = []
    all_pts   = []
    ival = 0
    for i in range(tnum):
        ival = ival+2
        pts = int(lines[ival].split()[0]) #track bends
        geo = np.array(lines[ival+1].split())
        xvals = geo[ np.arange(0,pts*2,2) ]
        yvals = geo[ np.arange(1,pts*2,2) ]
        all_xvals.append(xvals)
        all_yvals.append(yvals)
        all_pts.append(pts)
    track_points = { 'ntracks':tnum, 'pts':all_pts, 'xpt':all_xvals, 'ypt':all_yvals, 'names':['ntracks: number of ship tracks','pts: number of bends in ship track','xpt: x-position of nth bend in ship track','ypt: y-position of nth bend in ship track'] }
    return track_points



#Function to write track positions to bounding box
def track_to_bbox(track_points,xN,yN,cropwidth,bbox_range):
        structs = []
        for iT in range(track_points['ntracks']):
            xpt = ((track_points['xpt'])[iT]).astype(np.float)
            ypt = ((track_points['ypt'])[iT]).astype(np.float)
            pts = ((track_points['pts'])[iT])
            xW0 = 0  + cropwidth
            xW1 = xN - cropwidth
            NxN = xW1 - xW0
            if bbox_range == 'full':
                xrng = [np.min(xpt[:]),np.max(xpt[:])]
                yrng = [np.max(ypt[:]),np.min(ypt[:])]
            if bbox_range == 'partial':
                xrng = [xpt[0]-25,xpt[0]+25]
                yrng = [ypt[0]+25,ypt[0]-25]
                

            #left outside right outside
            if xrng[0] <= xW0 and xrng[1] >= xW1:
                flag = 0
                Nxrng0 = 0
                Nxrng1 = NxN
                truncateFlag = 1

            #both inside left
            if xrng[1] <= xW0:
                flag=1
                Nxrng0 = -999
                Nxrng1 = -999
                truncateFlag = 1

            #left outside and right inside
            if xrng[0] <= xW0 and xrng[1] > xW0 and xrng[1] <= xW1:
                Nxrng0 = 0
                Nxrng1 = xrng[1]-cropwidth
                flag=0
                truncateFlag = 1

            #both inside
            if xrng[0] >= xW0 and xrng[1] <= xW1:
                flag=0
                Nxrng0 = xrng[0] - cropwidth
                Nxrng1 = xrng[1] - cropwidth
                truncateFlag = 0

            #left inside right outside
            if xrng[0] >= xW0 and xrng[1] >= xW1:
                flag=0
                Nxrng0 = xrng[0] - cropwidth
                Nxrng1 = NxN
                truncateFlag = 1

            #both inside right
            if xrng[0] >= xW1:
                flag=1
                Nxrng0 = -999
                Nxrng1 = -999
                truncateFlag = 1
            #pdb.set_trace()
            print(iT,flag,truncateFlag,xrng[0],xrng[1],Nxrng0,Nxrng1,xW0,xW1)

            #pdb.set_trace()
            #if flag == 0:
            #    pdb.set_trace()

            K_truncated = str(truncateFlag)
            K_type = 'shiptrack'
            K_occluded = str(0)
            K_alpha = str(0)
            #K_bbox_left = str(int(xrng[0]))
            K_bbox_left = str(int(Nxrng0))
            K_bbox_top = str(int(yrng[1]))
            #K_bbox_right = str(int(xrng[1]))
            K_bbox_right = str(int(Nxrng1))
            K_bbox_bottom = str(int(yrng[0]))
            K_dimensions_height = str(0)
            K_dimensions_width = str(0)
            K_dimensions_length = str(0)
            K_location_x = str(0)
            K_location_y = str(0)
            K_location_z = str(0)
            K_rotation_y = str(0)
            K_score = str(0)
            struct = {'type':K_type,
                      'truncated':K_truncated,
                      'occluded':K_occluded,
                      'alpha':K_alpha,
                      'bbox_left':K_bbox_left,
                      'bbox_top':K_bbox_top,
                      'bbox_right':K_bbox_right,
                      'bbox_bottom':K_bbox_bottom,
                      'dimensions_height':K_dimensions_height,
                      'dimensions_width':K_dimensions_width,
                      'dimensions_length':K_dimensions_length,
                      'location_x':K_location_x,
                      'location_y':K_location_y,
                      'location_z':K_location_z,
                      'rotation_y':K_rotation_y,
                      'score':K_score,
                      'flag':flag}

            structs.append(struct)
            
        return structs
