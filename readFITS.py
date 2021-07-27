from fnmatch import fnmatch
from radio_beam import Beam
from astropy import units as u
from astropy.io import fits
from astropy.time import Time
from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel
from astropy.modeling.models import Gaussian2D
from datetime import date, datetime
from skyfield.api import load, wgs84
import numpy
import matplotlib.pyplot as plt
from matplotlib.pyplot import text
import scipy.stats as stats
from numpy import sqrt, pi, exp, linspace
from scipy import optimize
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import warnings
from astropy.modeling import models, fitting
from astropy.table import Table, Column
from astropy.io.fits.verify import VerifyWarning
import pandas
import pathlib

ephem   = load('de421.bsp') #1900-2050 only
sun     = ephem['sun']
earth   = ephem['earth']

taitam  = wgs84.latlon(22+14/60+33/3600, 114+13/60+24/3600)

ts      = load.timescale()

path20 = pathlib.Path.cwd().joinpath('20200620fits')
path21 = pathlib.Path.cwd().joinpath('20200621fits')

data20list = ['20200620-030641_TPI-PROJ01-SUN_01#_01#.fits',
              '20200620-031851_TPI-PROJ01-SUN_02#_01#.fits',
              '20200620-033155_TPI-PROJ01-SUN_03#_01#.fits',
              '20200620-043506_TPI-PROJ01-SUN_04#_01#.fits',
              '20200620-055043_TPI-PROJ01-SUN_05#_01#.fits',
              '20200620-060921_TPI-PROJ01-SUN_06#_01#.fits',
              '20200620-074952_TPI-PROJ01-SUN_07#_01#.fits',
              '20200620-075403_TPI-PROJ01-SUN_01#_01#.fits',
              '20200620-081857_TPI-PROJ01-SUN_01#_01#.fits',
              '20200620-083627_TPI-PROJ01-SUN_02#_01#.fits',
              '20200620-084950_TPI-PROJ01-SUN_03#_01#.fits']

image20list = ['20200620-030606_IMAGE-PROJ01-SUN_01#_01#.fits',
               '20200620-031845_IMAGE-PROJ01-SUN_01#_01#.fits',
               '20200620-055034_IMAGE-PROJ01-SUN_01#_01#.fits',
               '20200620-074928_IMAGE-PROJ01-SUN_01#_01#.fits',
               '20200620-075359_IMAGE-PROJ01-SUN_01#_01#.fits',
               '20200620-081836_IMAGE-PROJ01-SUN_01#_01#.fits',
               '20200620-083643_IMAGE-PROJ01-SUN_01#_01#.fits',
               '20200620-084946_IMAGE-PROJ01-SUN_01#_01#.fits']

data21list = ['20200621-002329_TPI-PROJ01-SUN_01#_01#.fits',
              '20200621-003906_TPI-PROJ01-SUN_02#_01#.fits',
              '20200621-012259_TPI-PROJ01-SUN_03#_01#.fits',
              '20200621-015252_TPI-PROJ01-SUN_04#_01#.fits',
              '20200621-022328_TPI-PROJ01-SUN_05#_01#.fits',
              '20200621-023853_TPI-PROJ01-SUN_06#_01#.fits',
              '20200621-025201_TPI-PROJ01-SUN_07#_01#.fits',
              '20200621-030632_TPI-PROJ01-SUN_08#_01#.fits',
              '20200621-032143_TPI-PROJ01-SUN_09#_01#.fits',
              '20200621-033759_TPI-PROJ01-SUN_10#_01#.fits',
              '20200621-035710_TPI-PROJ01-SUN_11#_01#.fits',
              '20200621-041618_TPI-PROJ01-SUN_12#_01#.fits',
              '20200621-043040_TPI-PROJ01-SUN_13#_01#.fits',
              '20200621-043138_TPI-PROJ01-SUN_14#_01#.fits',
              #'20200621-044722_TPI-PROJ01-SUN_15#_01#.fits',
              '20200621-050243_TPI-PROJ01-SUN_16#_01#.fits',
              '20200621-051846_TPI-PROJ01-SUN_17#_01#.fits',
              '20200621-053409_TPI-PROJ01-SUN_18#_01#.fits',
              '20200621-054732_TPI-PROJ01-SUN_19#_01#.fits',
              '20200621-060918_TPI-PROJ01-SUN_20#_01#.fits',
              '20200621-061827_TPI-PROJ01-SUN_21#_01#.fits',
              '20200621-062827_TPI-PROJ01-SUN_01#_01#.fits',
              '20200621-065248_TPI-PROJ01-SUN_02#_01#.fits',
              '20200621-070535_TPI-PROJ01-SUN_03#_01#.fits',
              '20200621-071804_TPI-PROJ01-SUN_04#_01#.fits',
              '20200621-072942_TPI-PROJ01-SUN_05#_01#.fits',
              '20200621-074139_TPI-PROJ01-SUN_06#_01#.fits',
              '20200621-075330_TPI-PROJ01-SUN_07#_01#.fits',
              '20200621-080624_TPI-PROJ01-SUN_08#_01#.fits',
              '20200621-081859_TPI-PROJ01-SUN_09#_01#.fits',
              '20200621-083037_TPI-PROJ01-SUN_10#_01#.fits',
              #'20200621-084235_TPI-PROJ01-SUN_11#_01#.fits',
              '20200621-085613_TPI-PROJ01-SUN_12#_01#.fits',
              '20200621-090800_TPI-PROJ01-SUN_13#_01#.fits']

image21list = ['20200621-002320_IMAGE-PROJ01-SUN_01#_01#.fits',
               '20200621-003855_IMAGE-PROJ01-SUN_01#_01#.fits',
               '20200621-012249_IMAGE-PROJ01-SUN_01#_01#.fits',
               '20200621-015248_IMAGE-PROJ01-SUN_01#_01#.fits',
               '20200621-022340_IMAGE-PROJ01-SUN_01#_01#.fits',
               '20200621-023843_IMAGE-PROJ01-SUN_01#_01#.fits',
               '20200621-025205_IMAGE-PROJ01-SUN_01#_01#.fits',
               '20200621-030637_IMAGE-PROJ01-SUN_01#_01#.fits',
               '20200621-032148_IMAGE-PROJ01-SUN_01#_01#.fits',
               '20200621-033808_IMAGE-PROJ01-SUN_01#_01#.fits',
               '20200621-035717_IMAGE-PROJ01-SUN_01#_01#.fits',
               '20200621-041627_IMAGE-PROJ01-SUN_01#_01#.fits',
               '20200621-043037_IMAGE-PROJ01-SUN_01#_01#.fits',
               '20200621-043130_IMAGE-PROJ01-SUN_01#_01#.fits',
               '20200621-050240_IMAGE-PROJ01-SUN_01#_01#.fits',
               '20200621-051853_IMAGE-PROJ01-SUN_01#_01#.fits',
               '20200621-053430_IMAGE-PROJ01-SUN_01#_01#.fits',
               '20200621-054743_IMAGE-PROJ01-SUN_01#_01#.fits',
               '20200621-060910_IMAGE-PROJ01-SUN_01#_01#.fits',
               '20200621-061834_IMAGE-PROJ01-SUN_01#_01#.fits',
               '20200621-062811_IMAGE-PROJ01-SUN_01#_01#.fits',
               '20200621-065243_IMAGE-PROJ01-SUN_01#_01#.fits',
               '20200621-070530_IMAGE-PROJ01-SUN_01#_01#.fits',
               '20200621-071802_IMAGE-PROJ01-SUN_01#_01#.fits',
               '20200621-072946_IMAGE-PROJ01-SUN_01#_01#.fits',
               '20200621-074143_IMAGE-PROJ01-SUN_01#_01#.fits',
               '20200621-075336_IMAGE-PROJ01-SUN_01#_01#.fits',
               '20200621-080631_IMAGE-PROJ01-SUN_01#_01#.fits',
               '20200621-081904_IMAGE-PROJ01-SUN_01#_01#.fits',
               '20200621-083041_IMAGE-PROJ01-SUN_01#_01#.fits',
               '20200621-085617_IMAGE-PROJ01-SUN_01#_01#.fits',
               '20200621-090805_IMAGE-PROJ01-SUN_01#_01#.fits']

def readimage(name):
    warnings.simplefilter('ignore', category=VerifyWarning)

    image_data      = fits.getdata(name, ext=0)

    plt.figure()
    plt.imshow(image_data, cmap='coolwarm')
    plt.colorbar()
    plt.show()

#readimage('20200621-002320_IMAGE-PROJ01-SUN_01#_01#.fits')

def readdata(name,ax_a,ax_b,ax_c):

    off_lvl = 0

    warnings.simplefilter('ignore', category=VerifyWarning)
    
    header_table0   = fits.getheader(name,1)

    hdulist = fits.open(name)
    data_table = hdulist[1].data

    ax_a.plot((data_table['JD']-2459021.83900)*24,[x+off_lvl*0 for x in data_table['BBC03u']],'c-')
    ax_a.plot((data_table['JD']-2459021.83900)*24,[x+off_lvl*1 for x in data_table['BBC04l']],'y--')
    ax_a.plot((data_table['JD']-2459021.83900)*24,[x+off_lvl*2 for x in data_table['BBC04u']],'y-')
    ax_a.plot((data_table['JD']-2459021.83900)*24,[x+off_lvl*3 for x in data_table['BBC10u']],'m-')
    ax_a.plot((data_table['JD']-2459021.83900)*24,[x+off_lvl*4 for x in data_table['BBC11l']],'r--')
    ax_a.plot((data_table['JD']-2459021.83900)*24,[x+off_lvl*5 for x in data_table['BBC11u']],'r-')
    ax_a.plot((data_table['JD']-2459021.83900)*24,[x+off_lvl*6 for x in data_table['BBC12l']],'g--')
    ax_a.plot((data_table['JD']-2459021.83900)*24,[x+off_lvl*7 for x in data_table['BBC12u']],'g-') #since on-off, airmass has no effect
    ax_a.axvline(x=(2459021.77561-2459021.83900)*24, c='b', ls ='--')
    ax_a.axvline(x=0, c='b', ls ='--')
    ax_a.axvline(x=(2459021.89189-2459021.83900)*24, c='b', ls ='--')

    ### SHOULD USE PTING MODEL TO RECLALIBRATE TELESCOPE POSITION AND RECALCULATE SEP
    t_range = ts.tt(jd=data_table['JD'])
    alt_s,az_s,dis_s = (earth+taitam).at(t_range).observe(sun).apparent().altaz()
    alt_t = data_table['Elevation']
    az_t = data_table['Azimuth']

    sep = numpy.degrees(numpy.sqrt(2-2*(numpy.cos(numpy.radians(alt_t))*numpy.cos(alt_s.radians)*numpy.cos(numpy.radians(az_t)-az_s.radians)
                                        +numpy.sin(numpy.radians(alt_t))*numpy.sin(alt_s.radians))))*60

    s_f = [0]*len(sep)
    for i in range(len(sep)):
        if sep[i] <= b_major/2:
            s_f[i] = gaussbeam(sep[i],0)
        else:
            s_f[i] = 999999999
    
    ax_b.plot((data_table['JD']-2459021.83900)*24,sep,'c-')
    ax_b.axhline(y=b_major/2,ls='--')
    ax_b.axvline(x=(2459021.77561-2459021.83900)*24, c='b', ls ='--')
    ax_b.axvline(x=0, c='b', ls ='--')
    ax_b.axvline(x=(2459021.89189-2459021.83900)*24, c='b', ls ='--')

    ax_c.plot((data_table['JD']-2459021.83900)*24,[i/j+off_lvl*0 for i,j in zip(data_table['BBC03u'],s_f)],'c-')
    ax_c.plot((data_table['JD']-2459021.83900)*24,[i/j+off_lvl*1 for i,j in zip(data_table['BBC04l'],s_f)],'y--')
    ax_c.plot((data_table['JD']-2459021.83900)*24,[i/j+off_lvl*2 for i,j in zip(data_table['BBC04u'],s_f)],'y-')
    ax_c.plot((data_table['JD']-2459021.83900)*24,[i/j+off_lvl*3 for i,j in zip(data_table['BBC10u'],s_f)],'m-')
    ax_c.plot((data_table['JD']-2459021.83900)*24,[i/j+off_lvl*4 for i,j in zip(data_table['BBC11l'],s_f)],'r--')
    ax_c.plot((data_table['JD']-2459021.83900)*24,[i/j+off_lvl*5 for i,j in zip(data_table['BBC11u'],s_f)],'r-')
    ax_c.plot((data_table['JD']-2459021.83900)*24,[i/j+off_lvl*6 for i,j in zip(data_table['BBC12l'],s_f)],'g--')
    ax_c.plot((data_table['JD']-2459021.83900)*24,[i/j+off_lvl*7 for i,j in zip(data_table['BBC12u'],s_f)],'g-')
    ax_c.axvline(x=(2459021.77561-2459021.83900)*24, c='b', ls ='--')
    ax_c.axvline(x=0, c='b', ls ='--')
    ax_c.axvline(x=(2459021.89189-2459021.83900)*24, c='b', ls ='--')

##    ax_c.plot((data_table['JD']-2459021.83900)*24,[i/j/j+10000 for i,j in zip(data_table['BBC03u'],s_f)],'c-')
##    ax_c.plot((data_table['JD']-2459021.83900)*24,[i/j/j+10000 for i,j in zip(data_table['BBC04l'],s_f)],'y--')
##    ax_c.plot((data_table['JD']-2459021.83900)*24,[i/j/j+10000 for i,j in zip(data_table['BBC04u'],s_f)],'y-')
##    ax_c.plot((data_table['JD']-2459021.83900)*24,[i/j/j+10000 for i,j in zip(data_table['BBC10u'],s_f)],'m-')
##    ax_c.plot((data_table['JD']-2459021.83900)*24,[i/j/j+10000 for i,j in zip(data_table['BBC11l'],s_f)],'r--')
##    ax_c.plot((data_table['JD']-2459021.83900)*24,[i/j/j+10000 for i,j in zip(data_table['BBC11u'],s_f)],'r-')
##    ax_c.plot((data_table['JD']-2459021.83900)*24,[i/j/j+10000 for i,j in zip(data_table['BBC12l'],s_f)],'g--')
##    ax_c.plot((data_table['JD']-2459021.83900)*24,[i/j/j+10000 for i,j in zip(data_table['BBC12u'],s_f)],'g-')

def readbeam(name): #beam is much larger than sun, assume sun radio image is simple gaussian source
    global gaussbeam,sungauss,b_X,b_Y,b_major
    
    header_table0   = fits.getheader(name,1)
    mybeam = Beam.from_fits_header(header_table0)
    b_major = header_table0['BMAJ']*60

    gaussbeam = Gaussian2D(1, 0, 0, b_major/(2*numpy.sqrt(2*numpy.log(2))), b_major/(2*numpy.sqrt(2*numpy.log(2)))) #FWHM~2.35sigma

    b_x = numpy.arange(-250, 251)
    b_y = numpy.arange(-250, 251)
    b_X,b_Y = numpy.meshgrid(b_x,b_y)
    
    sungauss = gaussbeam(b_X,b_Y)

### plot ###
limx = [0,0.12]

ax0 = plt.subplot(4,2,1)
ax1 = plt.subplot(4,2,2)
ax2 = plt.subplot(4,2,(3,4))
ax3 = plt.subplot(4,2,(5,6))
ax4 = plt.subplot(4,2,(7,8))

#extract data
readbeam(pathlib.Path.cwd().joinpath('20200621fits',data21list[0]))

##for i in range(4,len(data21list)):
##    readdata(pathlib.Path.cwd().joinpath('20200621fits',data21list[i]),ax2,ax3,ax4)

#beam plot
beam_im = ax0.imshow(sungauss,extent=(-250,251,-250,251),alpha=0.75,zorder=0)
beam_ct = ax0.contour(sungauss,extent=(-250,251,-250,251),zorder=1)
ax0.clabel(beam_ct, inline=True,fmt='%.2f')
ax0.contour(sungauss,[0.5],extent=(-250,251,-250,251),colors='r',linestyles='--',zorder=2)
plt.colorbar(beam_im,ax=ax0)

ax0.set_title('beam')
ax0.set_xlabel('arcmins')
ax0.set_ylabel('arcmins')
ax0.set_aspect('equal')

#respond curve
ax1.plot(numpy.arange(0, 251),gaussbeam(numpy.arange(0, 251),[0]*251))
ax1.set_title('response curve')
ax1.set_xlabel('offset (arcmins)')
ax1.set_ylabel('normalized response')

#data plot
#ax2.set_xlim(limx[0],limx[1])
ax2.set_title('20200621 Tai Tam')
ax2.set_xlabel('delta hour')
ax2.set_ylabel('ADU counts (arbitrary unit)')

#sep plot
#ax3.set_xlim(limx[0],limx[1])
ax3.set_title('seperation')
ax3.set_xlabel('delta hour')
ax3.set_ylabel('sep (arcmins)')

#corrected
#ax4.set_xlim(limx[0],limx[1])
#ax4.set_ylim(0,100000)
ax4.set_title('compensated counts')
ax4.set_xlabel('delta hour')
ax4.set_ylabel('ADU counts (arbitrary unit)') #uneven result since pointing error lead to wrong correcting factors

plt.get_current_fig_manager().window.wm_geometry('+0+0')
plt.show()
