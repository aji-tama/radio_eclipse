from fnmatch import fnmatch
from radio_beam import Beam
from astropy import units as u
from astropy.io import fits
from astropy.time import Time
from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel
from astropy.modeling.models import Gaussian2D
from datetime import date, datetime
from pytz import UTC, timezone, common_timezones
from skyfield import almanac
from skyfield.api import load, Topos, utc
import numpy
import matplotlib.pyplot as plt
from matplotlib.pyplot import text
import matplotlib.ticker as plticker
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

import fitFITS_04
import fitFITS_05
import fitFITS_06
import fitFITS_07
import fitFITS_08
import fitFITS_09
import fitFITS_14
import fitFITS_15
import fitFITS_16
import fitFITS_17
import fitFITS_20
import fitFITS_21
import fitFITS_22
import fitFITS_23
import fitFITS_24
import fitFITS_25
import fitFITS_26
import fitFITS_27
import fitFITS_28

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

### plot ###
limx = [0,0.12]

ax2 = plt.subplot(3,1,1)
ax3 = plt.subplot(3,1,2, sharex=ax2)
ax4 = plt.subplot(3,1,3, sharex=ax2)
#ax5 = plt.subplot(4,1,4, sharex=ax2)

#data plot
#ax2.set_xlim(limx[0],limx[1])
ax2.axvline(x=(2459021.77561-2459021.83900)*24, c='b', ls ='--')
ax2.axvline(x=0, c='b', ls ='--')
ax2.axvline(x=(2459021.89189-2459021.83900)*24, c='b', ls ='--')
ax2.set_title('20200621 Tai Tam')
#ax2.set_xlabel('delta hour')
ax2.set_ylabel('ADU counts (arbitrary unit)')

#sep plot
#ax3.set_xlim(limx[0],limx[1])
ax3.set_title('seperation')
#ax3.set_xlabel('delta hour')
ax3.set_ylabel('sep (arcmins)')

#corrected
#ax4.set_xlim(limx[0],limx[1])
#ax4.set_ylim(0,100000)
ax4.axvline(x=(2459021.77561-2459021.83900)*24, c='b', ls ='--')
ax4.axvline(x=0, c='b', ls ='--')
ax4.axvline(x=(2459021.89189-2459021.83900)*24, c='b', ls ='--')
ax4.set_title('compensated counts')
ax4.set_xlabel('delta hour')
ax4.set_ylabel('ADU counts (arbitrary unit)') #uneven result since pointing error lead to wrong correcting factors

fitFITS_04.readbeam(pathlib.Path.cwd().joinpath('20200621fits',data21list[4]))
fitFITS_04.readdata(pathlib.Path.cwd().joinpath('20200621fits',data21list[4]),ax2,ax3,ax4)
fitFITS_05.readbeam(pathlib.Path.cwd().joinpath('20200621fits',data21list[5]))
fitFITS_05.readdata(pathlib.Path.cwd().joinpath('20200621fits',data21list[5]),ax2,ax3,ax4)
fitFITS_06.readbeam(pathlib.Path.cwd().joinpath('20200621fits',data21list[6]))
fitFITS_06.readdata(pathlib.Path.cwd().joinpath('20200621fits',data21list[6]),ax2,ax3,ax4)
fitFITS_07.readbeam(pathlib.Path.cwd().joinpath('20200621fits',data21list[7]))
fitFITS_07.readdata(pathlib.Path.cwd().joinpath('20200621fits',data21list[7]),ax2,ax3,ax4)
fitFITS_08.readbeam(pathlib.Path.cwd().joinpath('20200621fits',data21list[8]))
fitFITS_08.readdata(pathlib.Path.cwd().joinpath('20200621fits',data21list[8]),ax2,ax3,ax4)
fitFITS_09.readbeam(pathlib.Path.cwd().joinpath('20200621fits',data21list[9]))
fitFITS_09.readdata(pathlib.Path.cwd().joinpath('20200621fits',data21list[9]),ax2,ax3,ax4)
fitFITS_14.readbeam(pathlib.Path.cwd().joinpath('20200621fits',data21list[14]))
fitFITS_14.readdata(pathlib.Path.cwd().joinpath('20200621fits',data21list[14]),ax2,ax3,ax4)
fitFITS_15.readbeam(pathlib.Path.cwd().joinpath('20200621fits',data21list[15]))
fitFITS_15.readdata(pathlib.Path.cwd().joinpath('20200621fits',data21list[15]),ax2,ax3,ax4)
fitFITS_16.readbeam(pathlib.Path.cwd().joinpath('20200621fits',data21list[16]))
fitFITS_16.readdata(pathlib.Path.cwd().joinpath('20200621fits',data21list[16]),ax2,ax3,ax4)
fitFITS_17.readbeam(pathlib.Path.cwd().joinpath('20200621fits',data21list[17]))
fitFITS_17.readdata(pathlib.Path.cwd().joinpath('20200621fits',data21list[17]),ax2,ax3,ax4)
fitFITS_20.readbeam(pathlib.Path.cwd().joinpath('20200621fits',data21list[20]))
fitFITS_20.readdata(pathlib.Path.cwd().joinpath('20200621fits',data21list[20]),ax2,ax3,ax4)
fitFITS_21.readbeam(pathlib.Path.cwd().joinpath('20200621fits',data21list[21]))
fitFITS_21.readdata(pathlib.Path.cwd().joinpath('20200621fits',data21list[21]),ax2,ax3,ax4)
fitFITS_22.readbeam(pathlib.Path.cwd().joinpath('20200621fits',data21list[22]))
fitFITS_22.readdata(pathlib.Path.cwd().joinpath('20200621fits',data21list[22]),ax2,ax3,ax4)
fitFITS_23.readbeam(pathlib.Path.cwd().joinpath('20200621fits',data21list[23]))
fitFITS_23.readdata(pathlib.Path.cwd().joinpath('20200621fits',data21list[23]),ax2,ax3,ax4)
fitFITS_24.readbeam(pathlib.Path.cwd().joinpath('20200621fits',data21list[24]))
fitFITS_24.readdata(pathlib.Path.cwd().joinpath('20200621fits',data21list[24]),ax2,ax3,ax4)
fitFITS_25.readbeam(pathlib.Path.cwd().joinpath('20200621fits',data21list[25]))
fitFITS_25.readdata(pathlib.Path.cwd().joinpath('20200621fits',data21list[25]),ax2,ax3,ax4)
fitFITS_26.readbeam(pathlib.Path.cwd().joinpath('20200621fits',data21list[26]))
fitFITS_26.readdata(pathlib.Path.cwd().joinpath('20200621fits',data21list[26]),ax2,ax3,ax4)
fitFITS_27.readbeam(pathlib.Path.cwd().joinpath('20200621fits',data21list[27]))
fitFITS_27.readdata(pathlib.Path.cwd().joinpath('20200621fits',data21list[27]),ax2,ax3,ax4)
fitFITS_28.readbeam(pathlib.Path.cwd().joinpath('20200621fits',data21list[28]))
fitFITS_28.readdata(pathlib.Path.cwd().joinpath('20200621fits',data21list[28]),ax2,ax3,ax4)

#model

#####################################
# ephem setting
tz          = timezone('Asia/Hong_Kong')

ephem       = load('de421.bsp') #1900-2050 only
sun         = ephem['sun']
earthmoon   = ephem['earth_barycenter']
earth       = ephem['earth']
moon        = ephem['moon']
#####################################

#####################################
# location information
#Hokoon
hokoon      = (Topos(str(22+23/60+1/3600)+' N', str(114+6/60+29/3600)+' E'),\
               22+23/60+1/3600,114+6/60+29/3600,'22:23:01','N','114:06:29','E')

Obs         = earth + hokoon[0] #<= set your observatory

ts          = load.timescale()
#####################################

plt.figure(0)

##Data_20         = pandas.read_csv(pathlib.Path.cwd().joinpath('20200621.csv'))
##
##Data_20.UT      = pandas.to_datetime(Data_20.UT).dt.tz_localize('UTC') # change to skyfield time obj
##Data_20.UT      = Data_20.UT.map(lambda t: t.replace(year=2020, month=6, day=21))
##Data_20_UTC     = ts.from_datetimes(Data_20.UT)
Data_20_UTC     = ts.utc(2020, 6, 21, 2, range(60*8))
#Data_20['HKT']  = pandas.to_datetime(Data_20.UT).dt.tz_convert(tz)

#print((Data_20_UTC.ut1-2459021.83900)*24)


(alt_s_20,az_s_20,d_s_20) = Obs.at(Data_20_UTC).observe(sun).apparent().altaz() # sun moon parameters
(alt_m_20,az_m_20,d_m_20) = Obs.at(Data_20_UTC).observe(moon).apparent().altaz()
Data_20_ele  = alt_s_20.degrees
Data_20_sep  = Obs.at(Data_20_UTC).observe(sun).apparent().separation_from(Obs.at(Data_20_UTC).observe(moon).apparent()).degrees

rs_20           = numpy.degrees(numpy.arctan2(1392000/2,d_s_20.au*149700598.8024))
rm_20           = numpy.degrees(numpy.arctan2(3474.2/2,d_m_20.au*149700598.8024))
det_20          = 2*(numpy.power(rs_20,2)*numpy.power(Data_20_sep,2)
                     +numpy.power(rm_20,2)*numpy.power(Data_20_sep,2)
                     +numpy.power(rs_20,2)*numpy.power(rm_20,2))-numpy.power(rs_20,4)-numpy.power(rm_20,4)-numpy.power(Data_20_sep,4)
Data_20_det  = [det if det>0 else 0 for det in det_20]

Data_20_area = 100*(numpy.pi*numpy.power(rs_20,2)
                    -(numpy.sqrt(Data_20_det)*numpy.sqrt(numpy.power(rm_20,2)-Data_20_det/(4*numpy.power(Data_20_sep,2))))/(2*Data_20_sep)
                    -numpy.power(rm_20,2)*numpy.arcsin(numpy.sqrt(Data_20_det)/(2*rm_20*Data_20_sep))+numpy.sqrt(Data_20_det)
                    -(numpy.sqrt(Data_20_det)*numpy.sqrt(numpy.power(rs_20,2)-Data_20_det/(4*numpy.power(Data_20_sep,2))))/(2*Data_20_sep)
                    -numpy.power(rs_20,2)*numpy.arcsin(numpy.sqrt(Data_20_det)/(2*rs_20*Data_20_sep)))/(numpy.pi*numpy.power(rs_20,2))

plt.plot((Data_20_UTC.ut1-2459021.83900)*24,Data_20_area,'y-')

#fitting
fitFITS_20_data_1 = fitFITS_20.data()[1][:1650] #2040
fitFITS_20_data_0 = fitFITS_20.data()[0][:1650]

pre_data = fitFITS_04.data()[1]+fitFITS_05.data()[1]+fitFITS_06.data()[1]+fitFITS_07.data()[1]+\
           fitFITS_08.data()[1]+fitFITS_09.data()[1]+[numpy.nan]+fitFITS_14.data()[1]+fitFITS_15.data()[1]+\
           fitFITS_16.data()[1]+fitFITS_17.data()[1]+[numpy.nan]
all_data = fitFITS_04.data()[1]+fitFITS_05.data()[1]+fitFITS_06.data()[1]+fitFITS_07.data()[1]+\
           fitFITS_08.data()[1]+fitFITS_09.data()[1]+[numpy.nan]+fitFITS_14.data()[1]+fitFITS_15.data()[1]+\
           fitFITS_16.data()[1]+fitFITS_17.data()[1]+[numpy.nan]+fitFITS_20_data_1+[numpy.nan]+fitFITS_21.data()[1]+\
           fitFITS_22.data()[1]+fitFITS_23.data()[1]+fitFITS_24.data()[1]+fitFITS_25.data()[1]+\
           fitFITS_26.data()[1]+fitFITS_27.data()[1]+fitFITS_28.data()[1]
##pre_data3 = fitFITS_04.data()[2]+fitFITS_05.data()[2]+fitFITS_06.data()[2]+fitFITS_07.data()[2]+\
##            fitFITS_08.data()[2]+fitFITS_09.data()[2]+[numpy.nan]+fitFITS_14.data()[2]+fitFITS_15.data()[2]+\
##            fitFITS_16.data()[2]+fitFITS_17.data()[2]+[numpy.nan]
##all_data3 = fitFITS_04.data()[2]+fitFITS_05.data()[2]+fitFITS_06.data()[2]+fitFITS_07.data()[2]+\
##            fitFITS_08.data()[2]+fitFITS_09.data()[2]+[numpy.nan]+fitFITS_14.data()[2]+fitFITS_15.data()[2]+\
##            fitFITS_16.data()[2]+fitFITS_17.data()[2]+[numpy.nan]+fitFITS_20.data()[2]+[numpy.nan]+fitFITS_21.data()[2]+\
##            fitFITS_22.data()[2]+fitFITS_23.data()[2]+fitFITS_24.data()[2]+fitFITS_25.data()[2]+\
##            fitFITS_26.data()[2]+fitFITS_27.data()[2]+fitFITS_28.data()[2]
all_date = fitFITS_04.data()[0]+fitFITS_05.data()[0]+fitFITS_06.data()[0]+fitFITS_07.data()[0]+\
           fitFITS_08.data()[0]+fitFITS_09.data()[0]+[numpy.nan]+fitFITS_14.data()[0]+fitFITS_15.data()[0]+\
           fitFITS_16.data()[0]+fitFITS_17.data()[0]+[numpy.nan]+fitFITS_20_data_0+[numpy.nan]+fitFITS_21.data()[0]+\
           fitFITS_22.data()[0]+fitFITS_23.data()[0]+fitFITS_24.data()[0]+fitFITS_25.data()[0]+\
           fitFITS_26.data()[0]+fitFITS_27.data()[0]+fitFITS_28.data()[0]
N_lvl = [x for x in pre_data if x == x]
N_f = numpy.average(N_lvl)
ax4.axhline(y=N_f,ls='--')

plt.plot(all_date,all_data/N_f*100)
plt.axvline(x=(2459021.77561-2459021.83900)*24, c='b', ls ='--', alpha=0.5)
plt.axvline(x=0, c='cyan', ls ='--', alpha=0.25)
plt.axvline(x=(2459021.89189-2459021.83900)*24, c='b', ls ='--', alpha=0.5)
plt.annotate('eclipse', xy=(((2459021.89189+2459021.77561)/2-2459021.83900)*24, 10), xycoords='data',
             horizontalalignment='center', verticalalignment='center')
plt.annotate('', xy=((2459021.77561-2459021.83900)*24, 10),  xycoords='data',
             xytext=((2459021.77561-2459021.83900)*24+1, 10), textcoords='data',
             arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=5),
             horizontalalignment='left', verticalalignment='center')
plt.annotate('', xy=((2459021.89189-2459021.83900)*24, 10),  xycoords='data',
             xytext=((2459021.89189-2459021.83900)*24-1, 10), textcoords='data',
             arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=5),
             horizontalalignment='right', verticalalignment='center')
#plt.title('')
plt.xlabel('delta hour')
plt.ylabel('power (%)')
plt.xlim(-6,1.5)
#plt.gca().xaxis.set_major_locator(plticker.MultipleLocator(base=0.5))
plt.ylim(0,120)
plt.legend(('optical','radio 1406-1446 MHz'))

def l2u(x):
    return x+0.33900*24

def u2l(x):
    return x-0.33900*24

axu = plt.gca().secondary_xaxis('top', functions=(l2u, u2l))
fmt = lambda x,y: '0{:.0f}:00'.format(x)
axu.xaxis.set_major_formatter(plticker.FuncFormatter(fmt))
axu.set_xlabel('Time (UT)')

#ax5.plot(all_date,all_data3/numpy.average([x for x in pre_data3 if x == x])*100)

plt.get_current_fig_manager().window.wm_geometry('+0+0')
plt.show()
