import numpy as np
import pandas as pd
from kapteyn import maputils
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy.random as random
import os

from astropy.wcs import WCS
from astropy.io import fits
            
from datetime import datetime,date

import sys
sys.path.append('/home/eduardo/Documents/TESIS FIRST/pyAstronometry')

from conversion_tools import date_str2jd,angle_decimal2str,mas2deg,deg2as
from astrometry import positions
from plotting_tools import draw_predictions,draw_predictions_MC_histograms

from IPython.display import clear_output

from multiprocessing import Pool

df = pd.read_csv('matches_250pc_simabad_new.csv')
n_proc = 6

# format date '01-01-2016/12:00:00'
def date_to_float(date):
    year = int(date[6:10])
    month = int(date[3:5])
    day = int(date[:2])
    return year+month/12+day/365

def get_pixel_pos(graticule, ra, dec):
    g_ra = graticule.wxlim
    if g_ra[0]<0 and g_ra[1]<0:
        g_ra = (360+g_ra[0],360+g_ra[1])
    g_dec = graticule.wylim
    g_pix_ra = graticule.pxlim
    g_pix_dec = graticule.pylim
    #print(g_pix_dec, g_pix_ra)
    pix_ra = round(g_pix_ra[1]-(ra-g_ra[0])*((g_pix_ra[1]-g_pix_ra[0])/(g_ra[1]-g_ra[0])))
    pix_dec = round(g_pix_dec[0]+(dec-g_dec[0])*((g_pix_dec[1]-g_pix_dec[0])/(g_dec[1]-g_dec[0])))
    return pix_ra, pix_dec

def get_number(file):
    number = ""
    flag = False
    for i in range(len(file)):
        if file[i]=='(':
            flag = False
            return number
        elif flag:
            number = number + file[i]
        elif file[i]=='_':
            flag = True
            
def get_epoch(file):
    epoch = ""
    flag = False
    for i in range(len(file)):
        if file[i]==')':
            flag = False
            return epoch
        elif flag:
            epoch = epoch + file[i]
        elif file[i]=='(':
            flag = True

def reformat_date(str_date):
    obsdate,obstime = str_date.split('T')
    year,month,day = obsdate.split("-")
    obsdate = day+"-"+month+"-"+year
    str_date = obsdate+"/"+obstime
    return str_date

# plotting trajectories point per point
def plot_trajectory(annim, traj_ra, traj_dec, color_):
    for i in range(len(traj_ra)):
        annim.Marker(pos=f'{traj_ra[i]} deg {traj_dec[i]} deg', marker='.', color=color_, markersize=0.5)

# plotting position and trajectories
# argumets(data=data of the star, annim=where to plot, obs_date=observation date reformated, 
#           color=color of the plot for the principal white, plot_final=if we want a cross in the final)
def plot_star(data, annim, obs_date, color_='w', plot_final=False): 
    # getting astro elements
    gaia_id = str(data['source_id'])
    RA_0 = data['ra'] 
    Dec_0 = data['dec']
    prlx = data['parallax']
    muRA = data['pmra']
    muDec = data['pmdec']
    jd0,mjd0 = date_str2jd('01-01-2016/12:00:00')
    jd,mjd = date_str2jd(obs_date)
    astro_elements = (jd0,RA_0,Dec_0,muRA,muDec,prlx)

    # getting trajectory with pyAstrometry
    predicted_position = positions(astro_elements,jd)
    if jd>=jd0:
        interval = range(int(jd0)-30,int(jd)+30)
    else:
        interval = range(int(jd0)+30,int(jd)-30,-1)
    jd_trajectory = np.array([day for day in interval])
    trajectory_star = positions(astro_elements,jd_trajectory)

    # plotting initial position of star
    annim.Marker(pos=f'{RA_0} deg {Dec_0} deg', 
                 marker='o', fillstyle='none', color=color_, markersize=10, markeredgewidth=2, label="Gaia source ID "+gaia_id)
    # plotting trajectory
    plot_trajectory(annim, trajectory_star[3], trajectory_star[4], color_)

    # if plot_final=True plotting with a cross the final position considering only proper motion
    if plot_final:
        time = date_to_float(obs_date)-2016.0
        annim.Marker(pos=f'{RA_0+muRA*time/3600000} deg {Dec_0+muDec*time/3600000} deg', 
                 marker='x', fillstyle='none', color=color_, markersize=15, markeredgewidth=2, label="Gaia source ID "+gaia_id)

# if there are additional stars, plotting these with another colors
def plot_other_stars(repetitions, annim, obs_date, plot_final=False):
    # colors other than white 'w', for maximum two aditional stars
    colors = ['k', 'silver']
    for i in range(len(repetitions)):
        plot_star(repetitions.loc[i], annim, obs_date, color_=colors[i], plot_final=plot_final)

def date_to_float(date):
    year = int(date[6:10])
    month = int(date[3:5])
    day = int(date[:2])
    return year+month/12+day/365

def float2date(epoch):
    date = epoch
    year = int(date)
    month_ = (date-year)*12+1
    month = int(month_)
    day = int((month_-month)*30+1)
    if month<10:
        month='0'+str(month)
    if day<10:
        day='0'+str(day)
    obs_date = f'{year}-{month}-{day}T12:00:00.0'
    return obs_date

def plot_fits_p(directory, file, i=-1, epoch=-1, ep=None, obs_date=None, pix_size=15, plot_final=False, name_star=None):
    if name_star==None:
        name_star = df['name'][i]
    
    # Vlass image an data
    if i==-1 and epoch==-1 and ep==None:
        i = int(get_number(file)) #getting the number from the name
        epoch = "VLASS "+ get_epoch(file)  #getting the epoch from the date
        ep = file[:2]    # getting the epoch matches ('e1' or 'e2')
    # reading the file fits file
    fitsobj = maputils.FITSimage(directory+file)
    # getting the dimenstions pixels
    shape = fitsobj.imshape

    # discarding fits with epoch 'its'
    if epoch != 'its':
        # getting the data with the same radio source
        print(i)
        data = df[df['i_system'] == df['i_system'][i]]
        
        # changing the unit Jy->mJy
        fitsobj.multiply_data(1000)

        # getting the values of the matrix
        boxdat = fitsobj.boxdat
        
        # reformating the observation date
        hdr = fitsobj.hdr
        if obs_date==None:
            obs_date = reformat_date(hdr['DATE-OBS'])
        else:
            obs_date = reformat_date(obs_date)
        
        # getting graticule
        grat = fitsobj.Annotatedimage().Graticule()

        # getting predicted pos and pix
        ra_p, dec_p = get_predicted_pos(data, obs_date)
        pix_pred_x, pix_pred_y = get_pixel_pos(grat, ra_p, dec_p)

        # Centering image in FIRST position
        pix_pos_x, pix_pos_y = get_pixel_pos(grat, df["ra"][i], df["dec"][i])
        plt.close()
        #print(pix_pos_x, pix_pos_y)
        fitsobj.set_limits(pxlim=(pix_pos_x-pix_size,pix_pos_x+pix_size), pylim=(pix_pos_y-pix_size,pix_pos_y+pix_size))
        
        # Plotting
        fig = plt.figure(figsize=(15,18))
        frame = fig.add_axes((0.2, 0.05, 0.7, 0.7))
        cbframe = fig.add_axes((0.91, 0.1085, 0.02, 0.5830))
        
        # writting the epoch image
        frame.text(0.98, 0.985, epoch, horizontalalignment='right', verticalalignment='top',
              transform = frame.transAxes, fontsize=45, color='k', bbox=dict(facecolor='w', alpha=0.8))

        # writting the name of star
        if name_star[:4]!="Gaia":
            name_size = 45
            if name_star[:5]=="2MASS":
                name_size = 30
            frame.text(0.02, 0.982, name_star, horizontalalignment='left', verticalalignment='top',
                  transform = frame.transAxes, fontsize=name_size, color='k', bbox=dict(facecolor='w', alpha=0.8))

        # writting the source_id of the stars
        legends=[]
        gaia_id = str(data['source_id'][i])
        legend_star = mlines.Line2D([], [], color='w', marker='o', fillstyle='none',
                      markersize=7, label="Gaia source ID "+gaia_id)
        legends.append(legend_star)
        data_c = data.copy().drop(i).reset_index()
        colors = ['k', 'silver']
        for j in range(len(data_c)):
            gaia_id = str(data_c['source_id'][j])
            legend_star = mlines.Line2D([], [], color=colors[j], marker='o', fillstyle='none',
                          markersize=7, label="Gaia source ID "+gaia_id)
            legends.append(legend_star)
            
        #legends_id = frame.legend(handles=legends, prop = { "size": 17 }, loc='lower right')
        #frame.add_artist(legends_id)
        frame.legend(handles=legends, prop = { "size": 17 }, loc='lower right')

        # getting clipmax
        clipmax = np.amax(boxdat[pix_pred_y-2:pix_pred_y+3, pix_pred_x-2:pix_pred_x+3])
        """print("pix_x", pix_pred_x)
        print("pix_y", pix_pred_y)
        print(boxdat.shape)
        dim_x, dim_y = boxdat.shape
        #pix_pred_x = dim_x-pix_pred_x
        print(boxdat)
        print("hola",boxdat[pix_pos_x-pix_size-1:pix_pos_x+pix_size+2, pix_pos_y-pix_size-1:pix_pos_y+pix_size+2])
        print("max",np.amax(boxdat[pix_pos_x-pix_size-1:pix_pos_x+pix_size+2, pix_pos_y-pix_size-1:pix_pos_y+pix_size+2]))
        print("hola",boxdat[pix_pred_y-2:pix_pred_y+3, pix_pred_x-2:pix_pred_x+3])"""

        # plotting the pixels map
        annim = fitsobj.Annotatedimage(cmap="viridis", frame=frame, clipmax=clipmax)
        annim.Image()

        # contornos
        max_flux = np.amax(annim.data)
        ini = 0.45
        fin = 1.35
        step = 0.2
        if max_flux>1.3:
            fin = max_flux
            step = (max_flux-ini)/3.5
        annim.Contours(levels=np.arange(ini,fin,step))

        # beam
        pos = f'{pix_pos_x-(pix_size-2)} {pix_pos_y-(pix_size-2)}'  # Pixels
        bmaj = hdr['BMAJ']               
        bmin = hdr['BMIN']                         
        bpa = hdr['BPA']
        beam = annim.Beam(bmaj, bmin, pa=bpa, pos=pos, fc='w', fill=True, alpha=0.8)

        # defining font size
        grat = annim.Graticule()
        grat.setp_ticklabel(plotaxis="left", fontsize=20, rotation=90)
        grat.setp_ticklabel(plotaxis="bottom", fontsize=20)
        grat.setp_axislabel(plotaxis="left", fontsize=30)
        grat.setp_axislabel(plotaxis="bottom", fontsize=30)

        # plotting the star and trajectory of final position
        plot_star(data.loc[i], annim, obs_date, plot_final=plot_final)    

        # plotting the other stars if there are
        repetitions = data.drop(i).reset_index()
        if len(repetitions!=0):
            plot_other_stars(repetitions, annim, obs_date, plot_final=plot_final)

        # colorbar ticks
        max_flux = np.nanmax(annim.data)
        min_flux = np.nanmin(annim.data)
        interval = max_flux-min_flux
        max_tick = round(max_flux-interval/15,1)
        min_tick = round(min_flux+interval/15,1)

        #print('while 1')
        for k in [5,4,3]:
            step = round((max_tick-min_tick)/k,1)
            if step>0:
                num_ticks = k
                break
        #print('salida while 1, step:',step)
        k = 0
        levels = []
        tick = max_tick
        while tick>min_tick:
            tick = round(max_tick-k*step,1)
            levels.append(tick)
            k += 1

        if max_tick>10:
            max_tick = round(max_tick,0)
            min_tick = round(min_tick,0)
            step = round((max_tick-min_tick)/5,0)
            tick = max_tick
            levels = []
            k = 0
            while tick>min_tick:
                tick = round(max_tick-k*step,0)
                levels.append(tick)
                k += 1
                

        units = r'$mJy/beam $'
        colbar = annim.Colorbar(fontsize=20, orientation='vertical', frame=cbframe, ticks=levels)
        #colbar = annim.Colorbar(fontsize=20, clines=True, linewidths=6)
        colbar.set_label(label=units, fontsize=30)
        annim.plot()
        annim.interact_imagecolors()
        
        ideal_shapes = [120,121,122,123]
        dir_images = 'images/'
        name = f'{file[:-5]}.jpg'
        plt.savefig(dir_images+name, format='jpg', bbox_inches='tight')
        #plt.show()
        plt.close()


def get_predicted_pos(data, obs_date): 
    # getting astro elements
    data = data.reset_index(drop=True).loc[0]
    gaia_id = str(data['source_id'])
    RA_0 = data['ra'] 
    Dec_0 = data['dec']
    prlx = data['parallax']
    muRA = data['pmra']
    muDec = data['pmdec']
    jd0,mjd0 = date_str2jd('01-01-2016/12:00:00')
    jd,mjd = date_str2jd(obs_date)
    astro_elements = (jd0,RA_0,Dec_0,muRA,muDec,prlx)

    # getting trajectory with pyAstrometry
    predicted_position = positions(astro_elements,jd)
    return predicted_position[3], predicted_position[4]


def plot_individual(j):
    file = list_fits[j]
    size = 8
    if j in [0,1, 8, 12]:
        size = 18
    elif j in [3,4,5]:
        size = 18
    elif j<7:
        size = 18
    elif j == 11:
        size = 22
    date = float2date(df['epoch'][j])
    plot_fits_p(directory, file, i=j, epoch="FIRST", ep="e1", obs_date=date, pix_size=size)


directory = 'fits/'
list_fits = sorted(os.listdir(directory))


if __name__  == '__main__':
    list_index = [i for i in range(len(df))]
    start = datetime.now()
    with Pool(processes=n_proc) as pool:
        pool.map(plot_individual, list_index[287:300])
    end = datetime.now()
    print("Imagenes generadas")
    print(f'Tiempo: {end-start}')