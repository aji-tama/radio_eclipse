import pathlib
import numpy
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import pandas
from skyfield import almanac
from skyfield.api import load, Topos, utc
from pytz import UTC, timezone, common_timezones
from datetime import date, time, datetime, timedelta
from scipy.optimize import curve_fit

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

# control data
Data_ctrl       = pandas.read_csv(pathlib.Path.cwd().joinpath('20200626.csv'))

Data_ctrl.tot   = Data_ctrl.iloc[:,-54:-10].sum(axis=1)*0.0078125 #add from ch11 to ch54, 44 items

Data_ctrl.UT    = pandas.to_datetime(Data_ctrl.UT).dt.tz_localize('UTC')
Data_ctrl.UT    = Data_ctrl.UT.map(lambda t: t.replace(year=2020, month=6, day=26))
Data_ctrl_UTC   = ts.from_datetimes(Data_ctrl.UT)

(alt_ctrl,az_ctrl,d_ctrl) = Obs.at(Data_ctrl_UTC).observe(sun).apparent().altaz()
Data_ctrl.ele   = alt_ctrl.degrees

# eclipse 2019
Data_19         = pandas.read_csv(pathlib.Path.cwd().joinpath('20191226.csv'))

Data_19['tot']  = Data_19.iloc[:,-54:-10].sum(axis=1)*0.0078125 #add from ch11 to ch54, 44 items

Data_19.UT      = pandas.to_datetime(Data_19.UT).dt.tz_localize('UTC') # change to skyfield time obj
Data_19.UT      = Data_19.UT.map(lambda t: t.replace(year=2019, month=12, day=26))
Data_19_UTC     = ts.from_datetimes(Data_19.UT)
Data_19['HKT']  = pandas.to_datetime(Data_19.UT).dt.tz_convert(tz)

(alt_s_19,az_s_19,d_s_19) = Obs.at(Data_19_UTC).observe(sun).apparent().altaz() # sun moon parameters
(alt_m_19,az_m_19,d_m_19) = Obs.at(Data_19_UTC).observe(moon).apparent().altaz()
Data_19['ele']  = alt_s_19.degrees
Data_19['sep']  = Obs.at(Data_19_UTC).observe(sun).apparent().separation_from(Obs.at(Data_19_UTC).observe(moon).apparent()).degrees

rs_19           = numpy.degrees(numpy.arctan2(1392000/2,d_s_19.au*149700598.8024))
rm_19           = numpy.degrees(numpy.arctan2(3474.2/2,d_m_19.au*149700598.8024))
det_19          = 2*(numpy.power(rs_19,2)*numpy.power(Data_19.sep,2)
                     +numpy.power(rm_19,2)*numpy.power(Data_19.sep,2)
                     +numpy.power(rs_19,2)*numpy.power(rm_19,2))-numpy.power(rs_19,4)-numpy.power(rm_19,4)-numpy.power(Data_19.sep,4)
Data_19['det']  = [det if det>0 else 0 for det in det_19]

Data_19['area'] = 100*(numpy.pi*numpy.power(rs_19,2)
                       -(numpy.sqrt(Data_19.det)*numpy.sqrt(numpy.power(rm_19,2)-Data_19.det/(4*numpy.power(Data_19.sep,2))))/(2*Data_19.sep)
                       -numpy.power(rm_19,2)*numpy.arcsin(numpy.sqrt(Data_19.det)/(2*rm_19*Data_19.sep))+numpy.sqrt(Data_19.det)
                       -(numpy.sqrt(Data_19.det)*numpy.sqrt(numpy.power(rs_19,2)-Data_19.det/(4*numpy.power(Data_19.sep,2))))/(2*Data_19.sep)
                       -numpy.power(rs_19,2)*numpy.arcsin(numpy.sqrt(Data_19.det)/(2*rs_19*Data_19.sep)))/(numpy.pi*numpy.power(rs_19,2))

Data_19         = Data_19.drop(Data_19.index[range(6283,6308)]) # drop 4th scan
Data_19         = Data_19.drop(Data_19.index[range(5950,5975)]) # drop 3rd scan
Data_19         = Data_19.drop(Data_19.index[range(5713,5738)]) # drop 2nd scan
Data_19         = Data_19.drop(Data_19.index[range(5382,5407)]) # drop 1st scan
Data_19         = Data_19.drop(Data_19.index[range(0,1298)]) # drop 1st hour

Data_19_eclipse = Data_19[Data_19['det'] != 0]
Data_19_control = Data_19[Data_19['det'] == 0]
Data_19_ctrl_a  = Data_19_control[Data_19_control['HKT'] < datetime(2019,12,26,13,54,38,tzinfo=tz)] # before totality
Data_19_ctrl_b  = Data_19_control[Data_19_control['HKT'] > datetime(2019,12,26,13,54,38,tzinfo=tz)] # after totality

# eclipse 2020
Data_20         = pandas.read_csv(pathlib.Path.cwd().joinpath('20200621.csv'))

Data_20['tot']  = Data_20.iloc[:,-54:-10].sum(axis=1)*0.0078125 #add from ch11 to ch54, 44 items

Data_20.UT      = pandas.to_datetime(Data_20.UT).dt.tz_localize('UTC') # change to skyfield time obj
Data_20.UT      = Data_20.UT.map(lambda t: t.replace(year=2020, month=6, day=21))
Data_20_UTC     = ts.from_datetimes(Data_20.UT)
Data_20['HKT']  = pandas.to_datetime(Data_20.UT).dt.tz_convert(tz)

(alt_s_20,az_s_20,d_s_20) = Obs.at(Data_20_UTC).observe(sun).apparent().altaz() # sun moon parameters
(alt_m_20,az_m_20,d_m_20) = Obs.at(Data_20_UTC).observe(moon).apparent().altaz()
Data_20['ele']  = alt_s_20.degrees
Data_20['sep']  = Obs.at(Data_20_UTC).observe(sun).apparent().separation_from(Obs.at(Data_20_UTC).observe(moon).apparent()).degrees

rs_20           = numpy.degrees(numpy.arctan2(1392000/2,d_s_20.au*149700598.8024))
rm_20           = numpy.degrees(numpy.arctan2(3474.2/2,d_m_20.au*149700598.8024))
det_20          = 2*(numpy.power(rs_20,2)*numpy.power(Data_20.sep,2)
                     +numpy.power(rm_20,2)*numpy.power(Data_20.sep,2)
                     +numpy.power(rs_20,2)*numpy.power(rm_20,2))-numpy.power(rs_20,4)-numpy.power(rm_20,4)-numpy.power(Data_20.sep,4)
Data_20['det']  = [det if det>0 else 0 for det in det_20]

Data_20['area'] = 100*(numpy.pi*numpy.power(rs_20,2)
                       -(numpy.sqrt(Data_20.det)*numpy.sqrt(numpy.power(rm_20,2)-Data_20.det/(4*numpy.power(Data_20.sep,2))))/(2*Data_20.sep)
                       -numpy.power(rm_20,2)*numpy.arcsin(numpy.sqrt(Data_20.det)/(2*rm_20*Data_20.sep))+numpy.sqrt(Data_20.det)
                       -(numpy.sqrt(Data_20.det)*numpy.sqrt(numpy.power(rs_20,2)-Data_20.det/(4*numpy.power(Data_20.sep,2))))/(2*Data_20.sep)
                       -numpy.power(rs_20,2)*numpy.arcsin(numpy.sqrt(Data_20.det)/(2*rs_20*Data_20.sep)))/(numpy.pi*numpy.power(rs_20,2))

Data_20_eclipse = Data_20[Data_20['det'] != 0]
Data_20_control = Data_20[Data_20['det'] == 0]
Data_20_ctrl_a  = Data_20_control[Data_20_control['HKT'] < datetime(2020,6,21,16,8,10,tzinfo=tz)] # before totality
Data_20_ctrl_b  = Data_20_control[Data_20_control['HKT'] > datetime(2020,6,21,16,8,10,tzinfo=tz)] # after totality

# 2019 data seems not calibrated, scaled to match 2020 data
sf = Data_20_ctrl_b.tot.mean()/Data_19_ctrl_b.tot.mean()

# fit control data to obtain tot vs. ele model
fitresult = numpy.polyfit(alt_ctrl.degrees,Data_ctrl.tot, 6)
mymodel = numpy.poly1d(fitresult)
print(mymodel)

def fitbaseline(x,c):
    return (fitresult[0]*x**6+fitresult[1]*x**5+fitresult[2]*x**4+fitresult[3]*x**3+fitresult[4]*x**2+fitresult[5]*x+fitresult[6])*c

par_19, cov_19 = curve_fit(f=fitbaseline, xdata=Data_19_ctrl_b.ele, ydata=Data_19_ctrl_b.tot*sf, p0=0)
par_20, cov_20 = curve_fit(f=fitbaseline, xdata=Data_20_ctrl_b.ele, ydata=Data_20_ctrl_b.tot, p0=0)

Data_19['blsub'] = Data_19.tot*sf-(fitresult[0]*Data_19.ele**6+fitresult[1]*Data_19.ele**5+fitresult[2]*Data_19.ele**4+fitresult[3]*Data_19.ele**3+fitresult[4]*Data_19.ele**2+fitresult[5]*Data_19.ele+fitresult[6])*par_19
Data_20['blsub'] = Data_20.tot-(fitresult[0]*Data_20.ele**6+fitresult[1]*Data_20.ele**5+fitresult[2]*Data_20.ele**4+fitresult[3]*Data_20.ele**3+fitresult[4]*Data_20.ele**2+fitresult[5]*Data_20.ele+fitresult[6])*par_20

# plot
fig = plt.figure(figsize=(12,9))
formatter = DateFormatter('%H:%M')
formatter.set_tzinfo(tz)

ax0 = plt.subplot(2,2,1)
ax0.plot(alt_ctrl.degrees,Data_ctrl.tot+50)
ax0.plot(numpy.linspace(13, 58, 100), mymodel(numpy.linspace(13, 58, 100))+50)

ax0.set_title('raw power (no eclipse) vs. elevation')
ax0.set_xlabel('elevation (deg)')
ax0.set_ylabel('power (arbitrary unit)')
ax0.legend(('20200626',
            'fitted baseline'))

ax1 = plt.subplot(2,2,2)
ax1.plot(Data_19_eclipse.ele,Data_19_eclipse.tot*sf,'g-')
ax1.plot(numpy.linspace(13, 58, 100), mymodel(numpy.linspace(13, 58, 100))*par_19,'g--')

ax1.plot(Data_20_eclipse.ele,Data_20_eclipse.tot,'r-')
ax1.plot(numpy.linspace(13, 58, 100), mymodel(numpy.linspace(13, 58, 100))*par_20,'r--')

ax1.plot(Data_19_ctrl_a.ele,Data_19_ctrl_a.tot*sf,'k--',alpha=0.5)
ax1.plot(Data_19_ctrl_b.ele,Data_19_ctrl_b.tot*sf,'k--',alpha=0.5)
ax1.plot(Data_20_ctrl_a.ele,Data_20_ctrl_a.tot,'k--',alpha=0.5)
ax1.plot(Data_20_ctrl_b.ele,Data_20_ctrl_b.tot,'k--',alpha=0.5)

ax1.set_title('raw power (eclipses) vs. elevation')
ax1.set_xlabel('elevation (deg)')
ax1.set_ylabel('power (arbitrary unit)')
ax1.legend(('20191226',
            '20191226 (baseline)',
            '20200621',
            '20200621 (baseline)',
            'no eclipse'))

ax2 = plt.subplot(2,2,3)
ax2.plot(Data_19.HKT,Data_19.blsub/Data_19_ctrl_b.tot.mean()*100,'g-')
ax2.plot(Data_19.HKT,Data_19.area-100,'y-')
ax2.axvline(x=datetime(2019,12,26,4,16,41), c='g', linestyle ='--')
ax2.axvline(x=datetime(2019,12,26,5,54,38), c='g', linestyle ='-.')
ax2.axvline(x=datetime(2019,12,26,7,21,30), c='g', linestyle ='--')

ax2.set_ylim([-100,20])
ax2.xaxis.set_major_formatter(formatter)
ax2.set_title('power reduced % vs. time 20191226')
ax2.set_xlabel('HKT')
ax2.set_ylabel('power reduced %')
ax2.legend(('radio',
            'optical'))

ax3 = plt.subplot(2,2,4)
ax3.plot(Data_20.HKT,Data_20.blsub/Data_20_ctrl_b.tot.mean()*100,'r-')
ax3.plot(Data_20.HKT,Data_20.area-100,'y-')
ax3.axvline(x=datetime(2020,6,21,6,36,53), c='r', linestyle ='--')
ax3.axvline(x=datetime(2020,6,21,8,8,10), c='r', linestyle ='-.')
ax3.axvline(x=datetime(2020,6,21,9,24,19), c='r', linestyle ='--')

ax3.set_ylim([-100,20])
ax3.xaxis.set_major_formatter(formatter)
ax3.set_title('power reduced % vs. time 20200621')
ax3.set_xlabel('HKT')
ax3.set_ylabel('power reduced %')
ax3.legend(('radio',
            'optical'))

plt.get_current_fig_manager().window.wm_geometry('+0+0')
fig.tight_layout()
plt.show()
