from fnmatch import fnmatch
from radio_beam import Beam
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
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
import scipy.optimize as optimize

delta_alt   = -35/60    #deg
delta_az    = 270/60

ephem   = load('de421.bsp') #1900-2050 only
sun     = ephem['sun']
earth   = ephem['earth']

taitam  = wgs84.latlon(22+14/60+33/3600, 114+13/60+24/3600)

ts      = load.timescale()

path21 = pathlib.Path.cwd().joinpath('20200621fits')

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
    global image_data, wcs
    warnings.simplefilter('ignore', category=VerifyWarning)

    image_data      = fits.getdata(name, ext=0)
    wcs             = WCS(fits.getheader(name, ext=0))
    wcs.wcs.crval   = [0,0]
    wcs.wcs.ctype   = [ 'XOFFSET' , 'YOFFSET' ]
    
def readdata(name,ax_a,ax_b,ax_c):
    global date_reduced,tot_reduced,BBC03_reduced

    off_lvl = 0

    warnings.simplefilter('ignore', category=VerifyWarning)
    
    header_table0   = fits.getheader(name,0)

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
    #ax_a.axvline(x=(2459021.77561-2459021.83900)*24, c='b', ls ='--')
    #ax_a.axvline(x=0, c='b', ls ='--')
    #ax_a.axvline(x=(2459021.89189-2459021.83900)*24, c='b', ls ='--')

    ### SHOULD USE PTING MODEL TO RECLALIBRATE TELESCOPE POSITION AND RECALCULATE SEP
    t_range = ts.tt(jd=data_table['JD'])
    alt_s,az_s,dis_s = (earth+taitam).at(t_range).observe(sun).apparent().altaz()
    alt_t = data_table['Elevation']+delta_alt
    az_t = data_table['Azimuth']+delta_az

    sep = numpy.degrees(numpy.sqrt(2-2*(numpy.cos(numpy.radians(alt_t))*numpy.cos(alt_s.radians)*numpy.cos(numpy.radians(az_t)-az_s.radians)
                                        +numpy.sin(numpy.radians(alt_t))*numpy.sin(alt_s.radians))))*60

    s_f = [0]*len(sep)
    for i in range(len(sep)):
        if sep[i] <= b_major/2:
            s_f[i] = gaussbeam(sep[i],0)
        else:
            s_f[i] = 999999999
    
    ax_b.plot((data_table['JD']-2459021.83900)*24,sep,'c-')
    ax_b.axhline(y=b_major,ls='--')
    ax_b.axhline(y=b_major/2,ls='--')
    #ax_b.axvline(x=(2459021.77561-2459021.83900)*24, c='b', ls ='--')
    #ax_b.axvline(x=0, c='b', ls ='--')
    #ax_b.axvline(x=(2459021.89189-2459021.83900)*24, c='b', ls ='--')

    ax_c.plot((data_table['JD']-2459021.83900)*24,[i/j+off_lvl*0 for i,j in zip(data_table['BBC03u'],s_f)],'c-')
    ax_c.plot((data_table['JD']-2459021.83900)*24,[i/j+off_lvl*1 for i,j in zip(data_table['BBC04l'],s_f)],'y--')
    ax_c.plot((data_table['JD']-2459021.83900)*24,[i/j+off_lvl*2 for i,j in zip(data_table['BBC04u'],s_f)],'y-')
    ax_c.plot((data_table['JD']-2459021.83900)*24,[i/j+off_lvl*3 for i,j in zip(data_table['BBC10u'],s_f)],'m-')
    ax_c.plot((data_table['JD']-2459021.83900)*24,[i/j+off_lvl*4 for i,j in zip(data_table['BBC11l'],s_f)],'r--')
    ax_c.plot((data_table['JD']-2459021.83900)*24,[i/j+off_lvl*5 for i,j in zip(data_table['BBC11u'],s_f)],'r-')
    ax_c.plot((data_table['JD']-2459021.83900)*24,[i/j+off_lvl*6 for i,j in zip(data_table['BBC12l'],s_f)],'g--')
    ax_c.plot((data_table['JD']-2459021.83900)*24,[i/j+off_lvl*7 for i,j in zip(data_table['BBC12u'],s_f)],'g-')

    FULLBAND = data_table['BBC03u']+data_table['BBC04l']+data_table['BBC04u']+data_table['BBC10u']+\
               data_table['BBC11l']+data_table['BBC11u']+data_table['BBC12l']+data_table['BBC12u']

    ax_a.plot((data_table['JD']-2459021.83900)*24,[x if x > 100 else numpy.nan for x in FULLBAND],'k-',alpha=0.5)

    date_reduced = list((data_table['JD']-2459021.83900)*24)
    tot_reduced = [i/j if i/j > 100 else numpy.nan for i,j in zip(FULLBAND,s_f)]
    BBC03_reduced = [i/j if i/j > 100 else numpy.nan for i,j in zip(data_table['BBC03u'],s_f)]
    ax_c.plot(date_reduced,tot_reduced,'k-',alpha=0.5)
   
    #ax_c.axvline(x=(2459021.77561-2459021.83900)*24, c='b', ls ='--')
    #ax_c.axvline(x=0, c='b', ls ='--')
    #ax_c.axvline(x=(2459021.89189-2459021.83900)*24, c='b', ls ='--')

    if __name__ == "__main__":
        ax1.plot(sep,[x/max(data_table['BBC03u']) for x in data_table['BBC03u']])

def data():
    return [date_reduced,tot_reduced,BBC03_reduced]

def readbeam(name): #beam is much larger than sun, assume sun radio image is simple gaussian source
    global gaussbeam,sungauss,b_X,b_Y,b_major
    
    header_table0   = fits.getheader(name,1)
    mybeam = Beam.from_fits_header(header_table0)
    #b_major = header_table0['BMAJ']*60
    b_major = numpy.average([166.6+183.8,164.7+181.1,165.4+180.8,
                             164.0+179.4,162.8+176.3,164.5+174.6,
                             168.6+173.2,166.7+175.5,166.6+176.1,
                             169.3+186.3])

    gaussbeam = Gaussian2D(1, 0, 0, b_major/(2*numpy.sqrt(2*numpy.log(2))), b_major/(2*numpy.sqrt(2*numpy.log(2)))) #FWHM~2.35sigma

    b_x = numpy.arange(-250, 251)
    b_y = numpy.arange(-250, 251)
    b_X,b_Y = numpy.meshgrid(b_x,b_y)
    
    sungauss = gaussbeam(b_X,b_Y)

if __name__ == "__main__":
    
    count = 0

    for i in [-35]: #alt
        for j in [270]: #az
            
            count = count + 1
            
            ### plot ###
            limx = [0,0.12]

            ax0 = plt.subplot(4,3,1)
            ax1 = plt.subplot(4,3,2)
            ax2 = plt.subplot(4,3,(4,6))
            ax3 = plt.subplot(4,3,(7,9))
            ax4 = plt.subplot(4,3,(10,12))

            #extract data
            obs_i       = 16      #<===================
            delta_alt   = i/60    #deg
            delta_az    = j/60    #deg

            readbeam(pathlib.Path.cwd().joinpath('20200621fits',data21list[obs_i]))
            readdata(pathlib.Path.cwd().joinpath('20200621fits',data21list[obs_i]),ax2,ax3,ax4)
            readimage(pathlib.Path.cwd().joinpath('20200621fits',image21list[obs_i]))

            ax5 = plt.subplot(4,3,3, projection=wcs)

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
            ax1.axvline(x=b_major/2, c='r', ls ='--')
            ax1.axhline(y=0.5, c='r', ls ='--')
            ax1.text(0.5,0.95,'response curve', ha='center',va='top', transform=ax1.transAxes)
            ax1.set_xlabel('offset (arcmins)')
            ax1.set_ylabel('normalized response')

            #map plot
            map_im = ax5.imshow(image_data, cmap='coolwarm')
            plt.colorbar(map_im,ax=ax5)
            ax5.coords[0].set_format_unit('arcminute')
            ax5.coords[1].set_format_unit('arcminute')
            ax5.plot((-300/60,300/60),(-delta_alt,-delta_alt), c='r', ls ='--', transform=ax5.get_transform('world'))
            ax5.plot((-delta_az,-delta_az),(-300/60,300/60), c='r', ls ='--', transform=ax5.get_transform('world'))
            ax5.invert_yaxis()
            ax5.set_title('image')

            ax5.contour(image_data,extent=(ax5.get_xlim()+tuple([ax5.get_ylim()[1],ax5.get_ylim()[0]])),zorder=1)

            ##def gaussfits(params):
            ##    A,x0,y0 = params
            ##    img = image_data
            ##    x = numpy.linspace(0, img.shape[1], img.shape[1])
            ##    y = numpy.linspace(0, img.shape[0], img.shape[0])
            ##    x, y = numpy.meshgrid(x, y)
            ##    g = Gaussian2D(1, 0, 0, b_major/(2*numpy.sqrt(2*numpy.log(2)))/60, b_major/(2*numpy.sqrt(2*numpy.log(2)))/60)(x,y)
            ##
            ##    return sum((img-g).ravel())
            ##
            ##initial_guess = [4000,3,3]
            ##result = optimize.minimize(gaussfits, initial_guess)

            #data plot
            #ax2.set_xlim(limx[0],limx[1])
            ax2.text(0.5,0.95,'20200621 Tai Tam', ha='center',va='top', transform=ax2.transAxes)
            #ax2.set_xlabel('delta hour')
            ax2.set_ylabel('ADU counts (arbitrary unit)')

            #sep plot
            #ax3.set_xlim(limx[0],limx[1])
            ax3.text(0.5,0.95,'seperation', ha='center',va='top', transform=ax3.transAxes)
            #ax3.set_xlabel('delta hour')
            ax3.set_ylabel('sep (arcmins)')

            #corrected
            #ax4.set_xlim(limx[0],limx[1])
            #ax4.set_ylim(0,100000)
            ax4.text(0.5,0.95,'compensated counts', ha='center',va='top', transform=ax4.transAxes)
            ax4.set_xlabel('delta hour')
            ax4.set_ylabel('ADU counts (arbitrary unit)') #uneven result since pointing error lead to wrong correcting factors


            fig = plt.gcf()
            fig.set_size_inches(18.5, 10.5)
            fig.suptitle(str(data21list[obs_i])+'\n'+str(image21list[obs_i])+'\n'+'delta_alt='+str(int(delta_alt*60))+'min delta_az='+str(int(delta_az*60))+'min')
            fig.subplots_adjust(hspace=0.4)
            plt.get_current_fig_manager().window.wm_geometry('+0+0')

            plt.savefig(str(count)+'_'+str(data21list[obs_i][9:14])+'_'+str(image21list[obs_i][9:14])+'_'+'delta_alt '+str(int(delta_alt*60))+' delta_az '+str(int(delta_az*60))+'.png', bbox_inches='tight')
            plt.clf()
            #plt.show()
