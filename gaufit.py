import pathlib
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from numpy import pi, r_
import matplotlib.pyplot as plt
from scipy import optimize

def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-x)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-y)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                 data)
    p, success = optimize.leastsq(errorfunction, params)
    return p

if __name__ == "__main__":
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

    # read FITS img
    name            = pathlib.Path.cwd().joinpath('20200621fits',image21list[27])
    wcs             = WCS(fits.getheader(name, ext=0))
    wcs.wcs.crval   = [0,0]
    wcs.wcs.ctype   = [ 'XOFFSET' , 'YOFFSET' ]
    image_data      = fits.getdata(name, ext=0)

    #print(image_data.shape[0])
    #print(image_data.shape[1])

    ax0 = plt.subplot(1,1,1, projection=wcs)
    ax0.imshow(image_data, cmap='coolwarm')
    ax0.coords[0].set_format_unit('arcminute')
    ax0.coords[1].set_format_unit('arcminute')
    ax0.invert_yaxis()

    params = fitgaussian(image_data)
    fit = gaussian(*params)

    plt.contour(fit(*np.indices(image_data.shape)), cmap='coolwarm')
    (height, x, y, width_x, width_y) = params

    ll = 338.835

    ax0.plot((-ll*7/7/60,-ll*7/7/60),(ll/60,-ll/60), c='r', ls ='--', transform=ax0.get_transform('world'))
    #ax0.plot((-ll*5/7/60,-ll*5/7/60),(ll/60,-ll/60), c='r', ls ='--', transform=ax0.get_transform('world'))
    #ax0.plot((-ll*3/7/60,-ll*3/7/60),(ll/60,-ll/60), c='r', ls ='--', transform=ax0.get_transform('world'))
    #ax0.plot((-ll/7/60,-ll/7/60),(ll/60,-ll/60), c='r', ls ='--', transform=ax0.get_transform('world'))
    #ax0.plot((ll/7/60,ll/7/60),(ll/60,-ll/60), c='r', ls ='--', transform=ax0.get_transform('world'))
    #ax0.plot((ll*3/7/60,ll*3/7/60),(ll/60,-ll/60), c='r', ls ='--', transform=ax0.get_transform('world'))
    #ax0.plot((ll*5/7/60,ll*5/7/60),(ll/60,-ll/60), c='r', ls ='--', transform=ax0.get_transform('world'))
    ax0.plot((ll*7/7/60,ll*7/7/60),(ll/60,-ll/60), c='r', ls ='--', transform=ax0.get_transform('world'))
    ax0.plot((ll/60,-ll/60),(-ll*7/7/60,-ll*7/7/60), c='r', ls ='--', transform=ax0.get_transform('world'))
    ax0.plot((ll/60,-ll/60),(ll*7/7/60,ll*7/7/60), c='r', ls ='--', transform=ax0.get_transform('world'))

##    ax0.plot((ll/30/7*(6.5-x+.25-3),ll/30/7*(6.5-x+.25-3)),(ll/60,-ll/60), c='k', ls ='--', transform=ax0.get_transform('world'))
##    ax0.plot((ll/60,-ll/60),(ll/30/7*(-6.5+y-.25+3),ll/30/7*(-6.5+y-.25+3)), c='k', ls ='--', transform=ax0.get_transform('world'))
##    
##    ax0.plot((6.5-x+.25,6.5-x+.25),(-0.5,6.5), c='y', ls ='--')
##    ax0.plot((-0.5,6.5),(6.5-y+.25,6.5-y+.25), c='y', ls ='--')

##    ax0.plot((ll/30/7*(x-3),ll/30/7*(x-3)),(ll/60,-ll/60), c='k', ls ='--', transform=ax0.get_transform('world'))
##    ax0.plot((ll/60,-ll/60),(ll/30/7*(3-y),ll/30/7*(3-y)), c='k', ls ='--', transform=ax0.get_transform('world'))
##
##    ax0.plot((x,x),(-0.5,6.5), c='y', ls ='--')
##    ax0.plot((-0.5,6.5),(y,y), c='y', ls ='--')

##    ax0.plot((ll/30/7*((x-.25)-3),ll/30/7*((x-.25)-3)),(ll/60,-ll/60), c='k', ls ='--', transform=ax0.get_transform('world'))
##    ax0.plot((ll/60,-ll/60),(ll/30/7*(3-(y+.25)),ll/30/7*(3-(y+.25))), c='k', ls ='--', transform=ax0.get_transform('world'))
##
##    ax0.plot((x-.25,x-.25),(-0.5,6.5), c='y', ls ='--')
##    ax0.plot((-0.5,6.5),(y+.25,y+.25), c='y', ls ='--')

##    ax0.plot((ll/30/7*((x+.5)-3),ll/30/7*((x+.5)-3)),(ll/60,-ll/60), c='k', ls ='--', transform=ax0.get_transform('world'))
##    ax0.plot((ll/60,-ll/60),(ll/30/7*(3-(y-.5)),ll/30/7*(3-(y-.5))), c='k', ls ='--', transform=ax0.get_transform('world'))
##
##    ax0.plot((x+.5,x+.5),(-0.5,6.5), c='y', ls ='--')
##    ax0.plot((-0.5,6.5),(y-.5,y-.5), c='y', ls ='--')

    ax0.plot((ll/30/7*((x+.25)-3),ll/30/7*((x+.25)-3)),(ll/60,-ll/60), c='k', ls ='--', transform=ax0.get_transform('world'))
    ax0.plot((ll/60,-ll/60),(ll/30/7*(3-(y-.25)),ll/30/7*(3-(y-.25))), c='k', ls ='--', transform=ax0.get_transform('world'))

    ax0.plot((x+.25,x+.25),(-0.5,6.5), c='y', ls ='--')
    ax0.plot((-0.5,6.5),(y-.25,y-.25), c='y', ls ='--')

    circle1 = plt.Circle((6.5-x+.25, 6.5-y+.25), width_x, color='r', fill=False)
    circle2 = plt.Circle((x, y), width_x, color='r', fill=False)
    circle3 = plt.Circle((x+.25, y-.25), width_x, color='r', fill=False)
    circle4 = plt.Circle((x+.5, y-.5), width_x, color='r', fill=False)
    circle5 = plt.Circle((x+.25, y-.25), width_x, color='r', fill=False)
    ax0.add_patch(circle5)

##    plt.text(0.95, 0.05, """
##    x : %.1f
##    y : %.1f
##    width_x : %.1f
##    width_y : %.1f""" %(60*ll/30/7*(6.5-x+.25-3), 60*ll/30/7*(6.5-y+.25-3), 60*width_x*ll/30/7, 60*width_y*ll/30/7),
##            fontsize=16, horizontalalignment='right',
##            verticalalignment='bottom', transform=ax0.transAxes)

##    plt.text(0.95, 0.05, """
##    x : %.1f
##    y : %.1f
##    width_x : %.1f
##    width_y : %.1f""" %(ll/30/7*(x-3), ll/30/7*(3-y), width_x*ll/30/7, width_y*ll/30/7),
##            fontsize=16, horizontalalignment='right',
##            verticalalignment='bottom', transform=ax0.transAxes)

##    plt.text(0.95, 0.05, """
##    x : %.1f
##    y : %.1f
##    width_x : %.1f
##    width_y : %.1f""" %(60*ll/30/7*((x-.25)-3), 60*ll/30/7*(3-(y+.25)), 60*width_x*ll/30/7, 60*width_y*ll/30/7),
##            fontsize=16, horizontalalignment='right',
##            verticalalignment='bottom', transform=ax0.transAxes)

##    plt.text(0.95, 0.05, """
##    x : %.1f
##    y : %.1f
##    width_x : %.1f
##    width_y : %.1f""" %(60*ll/30/7*((x+.5)-3), 60*ll/30/7*(3-(y-.5)), 60*width_x*ll/30/7, 60*width_y*ll/30/7),
##            fontsize=16, horizontalalignment='right',
##            verticalalignment='bottom', transform=ax0.transAxes)

    plt.text(0.95, 0.05, """
    x : %.1f
    y : %.1f
    width_x : %.1f
    width_y : %.1f""" %(60*ll/30/7*((x+.25)-3), 60*ll/30/7*(3-(y-.25)), 60*width_x*ll/30/7, 60*width_y*ll/30/7),
            fontsize=16, horizontalalignment='right',
            verticalalignment='bottom', transform=ax0.transAxes)

    #print(ax0.get_xlim(),ax0.get_ylim())
    #print(ax0.transLimits.inverted().transform(ax0.get_xlim()))

    #axis_to_data = ax0.transAxes + ax0.transData.inverted()
    #print(axis_to_data.transform((-0.5, 6.5)))
    plt.show()
    
