import warnings
warnings.simplefilter('error',RuntimeWarning)
import matplotlib
matplotlib.use('Agg')
import multiprocessing
import yt
import numpy as np
from yt import derived_field
from yt.fields.api import ValidateParameter
from yt.fields.derived_field import \
    ValidateGridType, \
    ValidateParameter, \
    ValidateSpatial, \
    NeedsParameter
from yt.utilities.physical_constants import G, mass_hydrogen
from yt.funcs import \
    just_one
from yt.units import kpc, Myr, Gyr,pc,Msun,km,second,Mpc
from yt.data_objects.particle_filters import add_particle_filter

from yt.config import ytcfg

import math
import fastcluster
import scipy.cluster.hierarchy as sch
from mpi4py import MPI
import os,sys
from astropy.table import Table , Column ,vstack ,hstack
from common_functions import *

sl_left = slice(None, -2, None)
sl_right = slice(2, None, None)
div_fac = 2.0

sl_center = slice(1, -1, None)
ftype='gas'

vort_validators = [ValidateSpatial(1,
                        [(ftype, "velocity_x"),
                         (ftype, "velocity_y"),
                         (ftype, "velocity_z")])]

def mad_based_outlier(points, thresh=5):
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

def _Disk_H(field, data):
    center = data.get_field_parameter('center')
    z = data["z"] - center[2]
    return np.abs(z)
def _Global_Disk_Radius(field, data):
    center = 0.5*data.ds.arr(np.ones(3), "code_length")
    x = data["x"] - center[0]
    y = data["y"] - center[1]
    r = np.sqrt(x*x+y*y)
    return r
def _Disk_Radius(field, data):
    center = data.get_field_parameter('center')
    x = data["x"] - center[0]
    y = data["y"] - center[1]
    r = np.sqrt(x*x+y*y)
    return r
def _Disk_Z(field, data):
    center = data.get_field_parameter('center')
    z = data["z"] - center[2]
    return z
def _Disk_Z2(field, data):
    center = data.get_field_parameter('center')
    z = data["z"] - center[2]
    return z**2
def _VZ(field,data):
        if data.has_field_parameter("bulk_velocity"):
                bv = data.get_field_parameter("bulk_velocity").in_units("cm/s")
        else:
                bv = data.ds.arr(np.zeros(3), "cm/s")
        zv = data["gas","velocity_z"] - bv[2]
        return zv
def _VZ2(field,data):
        if data.has_field_parameter("bulk_velocity"):
                bv = data.get_field_parameter("bulk_velocity").in_units("cm/s")
        else:
                bv = data.ds.arr(np.zeros(3), "cm/s")
        zv = data["gas","velocity_z"] - bv[2]
        return zv**2
def _sound_speed(field, data):
    tr = data.ds.gamma * data["gas", "pressure"] / data["gas", "density"]
    return np.sqrt(tr)
def _sfr_mass(field,data):
    mass=data[('particle_mass')].in_units('Msun')
    age=np.array(data[('age')].in_units('Myr'))
    tdyn=np.array(data[('dynamical_time')].in_units('Myr'))

    T=(age)/tdyn
    T[np.where(T>100)]=100
    f=(1-fej*(1-(1+T)*np.exp(-T)))
    mass = mass/f
    tau1=((age-5)/tdyn)
    tau0=((age-10)/tdyn)
    tau0[tau0<0]=0
    masa=mass*((1+tau0)*np.exp(-tau0) - (1+tau1)*np.exp(-tau1))
    #masa[(age>10)|(age<5)]=0*Msun
    masa[(age<5)]=0*Msun
    return masa
def young_stars(pfilter, data):
    age = data.ds.current_time - data[pfilter.filtered_type, "creation_time"]
    filter = np.logical_and(age.in_units('Myr') <= 10.0, age.in_units('Myr') >= 5.0)
    return filter
def _vorticity_z(field, data):
    f  = (data[ftype, "velocity_y"][sl_right,sl_center,sl_center] -
          data[ftype, "velocity_y"][sl_left,sl_center,sl_center]) \
          / (div_fac*just_one(data["index", "dx"]))
    f -= (data[ftype, "velocity_x"][sl_center,sl_right,sl_center] -
          data[ftype, "velocity_x"][sl_center,sl_left,sl_center]) \
          / (div_fac*just_one(data["index", "dy"]))
    new_field = data.ds.arr(np.zeros_like(data[ftype, "velocity_z"],
                                          dtype=np.float64),
                            f.units)
    new_field[sl_center, sl_center, sl_center] = f
    return new_field
def _div_vel(field, data):
    f  = (data[ftype, "velocity_x"][sl_right,sl_center,sl_center] -
          data[ftype, "velocity_x"][sl_left,sl_center,sl_center]) \
          / (div_fac*just_one(data["index", "dx"]))
    f += (data[ftype, "velocity_y"][sl_center,sl_right,sl_center] -
          data[ftype, "velocity_y"][sl_center,sl_left,sl_center]) \
          / (div_fac*just_one(data["index", "dy"]))
    f += (data[ftype, "velocity_z"][sl_center,sl_center,sl_right] -
          data[ftype, "velocity_z"][sl_center,sl_center,sl_left]) \
          / (div_fac*just_one(data["index", "dz"]))
    new_field = data.ds.arr(np.zeros_like(data[ftype, "velocity_z"],
                                          dtype=np.float64),
                            f.units)
    new_field[sl_center, sl_center, sl_center] = f
    return new_field
def _vturb(field, data):
    fx  = data[ftype, "velocity_x"][sl_right,sl_center,sl_center]/6.0
    fx += data[ftype, "velocity_x"][sl_left,sl_center,sl_center]/6.0
    fx += data[ftype, "velocity_x"][sl_center,sl_right,sl_center]/6.0
    fx += data[ftype, "velocity_x"][sl_center,sl_left,sl_center]/6.0
    fx += data[ftype, "velocity_x"][sl_center,sl_center,sl_right]/6.0
    fx += data[ftype, "velocity_x"][sl_center,sl_center,sl_left]/6.0

    FX= (data[ftype, "velocity_x"][sl_center,sl_center,sl_center]-fx)

    fy  = data[ftype, "velocity_y"][sl_right,sl_center,sl_center]/6.0
    fy += data[ftype, "velocity_y"][sl_left,sl_center,sl_center]/6.0
    fy += data[ftype, "velocity_y"][sl_center,sl_right,sl_center]/6.0
    fy += data[ftype, "velocity_y"][sl_center,sl_left,sl_center]/6.0
    fy += data[ftype, "velocity_y"][sl_center,sl_center,sl_right]/6.0
    fy += data[ftype, "velocity_y"][sl_center,sl_center,sl_left]/6.0

    FY= (data[ftype, "velocity_y"][sl_center,sl_center,sl_center]-fy)

    fz  = data[ftype, "velocity_z"][sl_right,sl_center,sl_center]/6.0
    fz += data[ftype, "velocity_z"][sl_left,sl_center,sl_center]/6.0
    fz += data[ftype, "velocity_z"][sl_center,sl_right,sl_center]/6.0
    fz += data[ftype, "velocity_z"][sl_center,sl_left,sl_center]/6.0
    fz += data[ftype, "velocity_z"][sl_center,sl_center,sl_right]/6.0
    fz += data[ftype, "velocity_z"][sl_center,sl_center,sl_left]/6.0

    FZ= (data[ftype, "velocity_z"][sl_center,sl_center,sl_center]-fz)



    new_field = data.ds.arr(np.zeros_like(data[ftype, "velocity_z"],
                                          dtype=np.float64),
                            FX.units)
    new_field[sl_center, sl_center, sl_center] = np.sqrt(FX**2+FY**2+FZ**2)
    return new_field
def _sturb(field, data):
    fx  = data[ftype, "velocity_x"][sl_right,sl_center,sl_center]/7.0
    fx += data[ftype, "velocity_x"][sl_left,sl_center,sl_center]/7.0
    fx += data[ftype, "velocity_x"][sl_center,sl_right,sl_center]/7.0
    fx += data[ftype, "velocity_x"][sl_center,sl_left,sl_center]/7.0
    fx += data[ftype, "velocity_x"][sl_center,sl_center,sl_right]/7.0
    fx += data[ftype, "velocity_x"][sl_center,sl_center,sl_left]/7.0
    fx += data[ftype, "velocity_x"][sl_center,sl_center,sl_center]/7.0

    FX  = (data[ftype, "velocity_x"][sl_right,sl_center,sl_center]-fx)**2/7.0
    FX += (data[ftype, "velocity_x"][sl_left,sl_center,sl_center]-fx)**2/7.0
    FX += (data[ftype, "velocity_x"][sl_center,sl_right,sl_center]-fx)**2/7.0
    FX += (data[ftype, "velocity_x"][sl_center,sl_left,sl_center]-fx)**2/7.0
    FX += (data[ftype, "velocity_x"][sl_center,sl_center,sl_right]-fx)**2/7.0
    FX += (data[ftype, "velocity_x"][sl_center,sl_center,sl_left]-fx)**2/7.0
    FX += (data[ftype, "velocity_x"][sl_center,sl_center,sl_center]-fx)**2/7.0

    fy  = data[ftype, "velocity_y"][sl_right,sl_center,sl_center]/7.0
    fy += data[ftype, "velocity_y"][sl_left,sl_center,sl_center]/7.0
    fy += data[ftype, "velocity_y"][sl_center,sl_right,sl_center]/7.0
    fy += data[ftype, "velocity_y"][sl_center,sl_left,sl_center]/7.0
    fy += data[ftype, "velocity_y"][sl_center,sl_center,sl_right]/7.0
    fy += data[ftype, "velocity_y"][sl_center,sl_center,sl_left]/7.0
    fy += data[ftype, "velocity_y"][sl_center,sl_center,sl_center]/7.0

    FY  = (data[ftype, "velocity_y"][sl_right,sl_center,sl_center]-fy)**2/7.0
    FY += (data[ftype, "velocity_y"][sl_left,sl_center,sl_center]-fy)**2/7.0
    FY += (data[ftype, "velocity_y"][sl_center,sl_right,sl_center]-fy)**2/7.0
    FY += (data[ftype, "velocity_y"][sl_center,sl_left,sl_center]-fy)**2/7.0
    FY += (data[ftype, "velocity_y"][sl_center,sl_center,sl_right]-fy)**2/7.0
    FY += (data[ftype, "velocity_y"][sl_center,sl_center,sl_left]-fy)**2/7.0
    FY += (data[ftype, "velocity_y"][sl_center,sl_center,sl_center]-fy)**2/7.0

    fz  = data[ftype, "velocity_z"][sl_right,sl_center,sl_center]/7.0
    fz += data[ftype, "velocity_z"][sl_left,sl_center,sl_center]/7.0
    fz += data[ftype, "velocity_z"][sl_center,sl_right,sl_center]/7.0
    fz += data[ftype, "velocity_z"][sl_center,sl_left,sl_center]/7.0
    fz += data[ftype, "velocity_z"][sl_center,sl_center,sl_right]/7.0
    fz += data[ftype, "velocity_z"][sl_center,sl_center,sl_left]/7.0
    fz += data[ftype, "velocity_z"][sl_center,sl_center,sl_center]/7.0

    FZ  = (data[ftype, "velocity_z"][sl_right,sl_center,sl_center]-fz)**2/7.0
    FZ += (data[ftype, "velocity_z"][sl_left,sl_center,sl_center]-fz)**2/7.0
    FZ += (data[ftype, "velocity_z"][sl_center,sl_right,sl_center]-fz)**2/7.0
    FZ += (data[ftype, "velocity_z"][sl_center,sl_left,sl_center]-fz)**2/7.0
    FZ += (data[ftype, "velocity_z"][sl_center,sl_center,sl_right]-fz)**2/7.0
    FZ += (data[ftype, "velocity_z"][sl_center,sl_center,sl_left]-fz)**2/7.0
    FZ += (data[ftype, "velocity_z"][sl_center,sl_center,sl_center]-fz)**2/7.0

    new_field = data.ds.arr(np.zeros_like(data[ftype, "velocity_z"],
                                          dtype=np.float64),
                            np.sqrt(FX).units)
    new_field[sl_center, sl_center, sl_center] = np.sqrt(FX+FY+FZ)
    return new_field
def _Omega(field,data):
        if data.has_field_parameter("bulk_velocity"):
                bv = data.get_field_parameter("bulk_velocity").in_units("cm/s")
        else:
                bv = data.ds.arr(np.zeros(3), "cm/s")
        xv = data["gas","velocity_x"] - bv[0]
        yv = data["gas","velocity_y"] - bv[1]
        center = data.get_field_parameter('center')
        x_hat = data["x"] - center[0]
        y_hat = data["y"] - center[1]
        r = np.sqrt(x_hat*x_hat+y_hat*y_hat)
        x_hat /= r**2
        y_hat /= r**2

        return yv*x_hat-xv*y_hat
def _vc(field,data):
        if data.has_field_parameter("bulk_velocity"):
                bv = data.get_field_parameter("bulk_velocity").in_units("cm/s")
        else:
                bv = data.ds.arr(np.zeros(3), "cm/s")
        xv = data["gas","velocity_x"] - bv[0]
        yv = data["gas","velocity_y"] - bv[1]
        center = data.get_field_parameter('center')
        x_hat = data["x"] - center[0]
        y_hat = data["y"] - center[1]
        r = np.sqrt(x_hat*x_hat+y_hat*y_hat)
        x_hat /= r
        y_hat /= r

        return (yv*x_hat-xv*y_hat)
add_particle_filter("young_stars", function=young_stars, filtered_type='all', requires=["particle_type"])
yt.add_field("Disk_H",
             function=_Disk_H,
             units="pc",
             take_log=False,
             validators=[ValidateParameter('center')])
yt.add_field("Global_Disk_Radius",
             function=_Global_Disk_Radius,
             units="cm",
             take_log=False,
             validators=[ValidateParameter('center')])
yt.add_field("Disk_Radius",
             function=_Disk_Radius,
             units="cm",
             take_log=False,
             validators=[ValidateParameter('center')])
yt.add_field("Disk_Z",
             function=_Disk_Z,
             units="pc",
             take_log=False,
             validators=[ValidateParameter('center')])
yt.add_field("Disk_Z2",
             function=_Disk_Z2,
             units="pc**2",
             take_log=False,
             validators=[ValidateParameter('center')])
yt.add_field("VZ", function=_VZ,take_log=False, units=r"km/s",validators=[ValidateParameter('bulk_velocity')])
yt.add_field("VZ2", function=_VZ2,take_log=False, units=r"km**2/s**2",validators=[ValidateParameter('bulk_velocity')])
yt.add_field("sound_speed", function=_sound_speed, units=r"km/s")
yt.add_field("sfr_mass",function=_sfr_mass,
             units="Msun",
             take_log=False,
             validators=[ValidateParameter('center')],particle_type='True')
yt.add_field((ftype, "vorticity_z"),function=_vorticity_z,
                       units="1/yr",
                       validators=vort_validators,take_log=True)
yt.add_field((ftype, "div_vel"),function=_div_vel,
                       units="1/yr",
                       validators=vort_validators,take_log=True)
yt.add_field(("gas", "sturb"),function=_sturb,units="km/s",validators=vort_validators)
yt.add_field(("gas", "vturb"),function=_vturb,units="km/s",validators=vort_validators)
yt.add_field("Omega", function=_Omega,take_log=False, units=r"1/yr",validators=[ValidateParameter('bulk_velocity')])
yt.add_field("vc", function=_vc,take_log=False, units=r"km/s",validators=[ValidateParameter('bulk_velocity')])

def Velocity_profile(V,R,DR,Rmin,Rmax,function):

    Redges=np.arange(Rmin*pc,Rmax*pc,DR*pc)
    Rcen=0.5*(Redges[1:]+Redges[0:-1])
    N=len(Rcen)
    Vcen=np.zeros(N)
    for k in range(N):
        ring=(Redges[k]<R)&(Redges[k+1]>R)
        if len(ring[ring])<4:
            Vcen[k]=np.nan
        else:
            auxome=V[ring]
            Vcen[k]=function(auxome)

    correct=~np.isnan(Vcen)
    Rcen=Rcen[correct]
    Vcen=Vcen[correct]

    return Rcen,Vcen
def Vorticity_profile(V,R,DR,Rmin,Rmax,function):
    x,y = Velocity_profile(V,R,DR,Rmin,Rmax,function)
    U=np.copy(y)
    du=x[1:-1]*(U[2:]-U[:-2])/((x[2:]-x[:-2])*U[1:-1])
    du=np.insert(du,-1,du[-1])
    du=np.insert(du,0,du[0])
    y=y/x
    x=np.insert(x,0,0)
    y=np.insert(y,0,y[0]*2)
    y[1:]*=(1+du)

    return x,y

def Quantities(ds,xo,yo,scale,error,
        density,new_stellar,molecular_density,x_i,y_i,sigma_v,Hg,
        nu_g,VDiff, Rgal,sspeed,omega_s,xg,yg,mass,
        vxg,vyg,vzg,r_gal,star_formation_s,star_mass_s,star_age,
        star_vz_s,x_star,y_star,star_z_s,average_omega_s,average_vorticity_s,nu_star_s,omega_star_s,kappa_star_s,nu_dm_s,
        omega_dm_s,kappa_dm_s,Ofield,Kfield,Nfield,DX):

    #print(id)
    Result=[]

    circle_3d=np.sqrt((xo-xg)**2+(yo-yg)**2)<scale
    circle_2d=np.sqrt((x_i-xo)**2+(y_i-yo)**2)<scale
    circle_star=np.sqrt((x_star-xo)**2+(y_star-yo)**2)<scale

    gas_mass    =   mass[circle_3d]
    xdisk       =   xg[circle_3d]
    ydisk       =   yg[circle_3d]
    zdisk       =   zg[circle_3d]
    try:
        xdisk_cm    =   np.average(xdisk,weights=gas_mass)
    except ZeroDivisionError:
        print(xo,yo,scale)
    ydisk_cm    =   np.average(ydisk,weights=gas_mass)
    zdisk_cm    =   np.average(zdisk,weights=gas_mass)
    xdisk      -=   xdisk_cm
    ydisk      -=   ydisk_cm
    zdisk      -=   zdisk_cm
    del xdisk_cm,ydisk_cm,zdisk_cm
    vx          =   vxg[circle_3d]
    vy          =   vyg[circle_3d]
    vz          =   vzg[circle_3d]
    vx_cm       =   np.average(vx,weights=gas_mass)
    vy_cm       =   np.average(vy,weights=gas_mass)
    vz_cm       =   np.average(vz,weights=gas_mass)
    vx         -=   vx_cm
    vy         -=   vy_cm
    vz         -=   vz_cm
    k11         =   (0.5*gas_mass*vx**2).sum()
    k22         =   (0.5*gas_mass*vy**2).sum()
    k33         =   (0.5*gas_mass*vz**2).sum()
    k11.convert_to_units('Msun*pc**2/Myr**2')
    k22.convert_to_units('Msun*pc**2/Myr**2')
    k33.convert_to_units('Msun*pc**2/Myr**2')

    del vx_cm,vy_cm,vz_cm
    Iz          =   max((gas_mass*(xdisk**2+ydisk**2)).sum(),gas_mass.sum()*DX**2/6.0)
    Lz          =   (gas_mass*(xdisk*vy-ydisk*vx)).sum()
    rot         =   Lz/Iz
    Lx          =   (gas_mass*(ydisk*vz-zdisk*vy)).sum()
    Ly          =   (gas_mass*(zdisk*vx-xdisk*vz)).sum()
    angle       =   np.arccos(Lz/np.sqrt(Lx**2+Ly**2+Lz**2))
    del Lx,Ly

    aux_R       =   r_gal[circle_3d]
    OT          =   Ofield[circle_3d]
    KT          =   Kfield[circle_3d]
    NT          =   Nfield[circle_3d]

    v11         =   (gas_mass*(xdisk**2)*(3*OT-KT)).sum()
    v22         =   -(gas_mass*(ydisk**2)*OT).sum()
    v33         =   -(gas_mass*(zdisk**2)*NT).sum()
    v11.convert_to_units('Msun*pc**2/Myr**2')
    v22.convert_to_units('Msun*pc**2/Myr**2')
    v33.convert_to_units('Msun*pc**2/Myr**2')

    del xdisk,ydisk,vx,vy,vz,aux_R,OT,KT,NT

    gas_mass            =   gas_mass.sum()
    surface_density     =   density[circle_2d]
    new_stellar_density =   new_stellar[circle_2d]
    molecular_surface_density = molecular_density[circle_2d]
    frac                =   molecular_surface_density.sum()/surface_density.sum()
    molecular_gas_mass  =   frac*gas_mass


    sigma_vel   =   sigma_v[circle_2d]
    gas_H       =   Hg[circle_2d]
    Zg          =   gas_H*2*np.sqrt(3.0)/np.pi
    nu_gas      =   nu_g[circle_2d]
    v_diff      =   VDiff[circle_2d]
    galaxy_radius   =   Rgal[circle_2d]
    sound_speed         =   sspeed[circle_2d]
    average_omega       =   average_omega_s[circle_2d]
    average_vorticity   =   average_vorticity_s[circle_2d]
    omega               =   omega_s[circle_2d]
    nu_star     =   nu_star_s[circle_2d]
    omega_star  =   omega_star_s[circle_2d]
    kappa_star  =   kappa_star_s[circle_2d]

    nu_dm       =   nu_dm_s[circle_2d]
    omega_dm    =   omega_dm_s[circle_2d]
    kappa_dm    =   kappa_dm_s[circle_2d]

    star_formation  =   star_formation_s[circle_star].sum()
    star_edad       =   star_age[circle_star]
    star_mass       =   star_mass_s[circle_star]
    sfr_10 	    =   star_mass[star_edad<10*Myr].sum()/(10*Myr)
    sfr_01          =   star_mass[star_edad<1*Myr].sum()/(1*Myr) 
    
    sfr_10.convert_to_units('Msun/yr')
    sfr_01.convert_to_units('Msun/yr')

    number_stars    =   len(star_mass)
    #print('number of stars', number_stars)
    star_vz         =   star_vz_s[circle_star]
    star_vz2        =   star_vz[:]
    try:
        star_vz     =   np.average(star_vz,weights=star_mass)
        star_sigma  =   np.sqrt(np.average((star_vz2-star_vz)**2,weights=star_mass))
    except ZeroDivisionError:
        star_vz     =   0.0*km/second
        star_sigma  =   0.0*km/second

    star_z      =   star_z_s[circle_star]
    star_z2     =   star_z[:]
    try:
        star_z  =   np.average(star_z,weights=star_mass)
        star_H  =   np.sqrt(np.average((star_z2-star_z)**2,weights=star_mass))
    except ZeroDivisionError:
        star_z  =   0.0*pc
        star_H  =   0.0*pc
    Zs          =   max(star_H*2*np.sqrt(3.0)/np.pi,0.5*DX)
    star_mass   =   star_mass.sum()
    nu_new_stars=   np.sqrt(np.pi*G*new_stellar_density/Zs).in_units('1/Myr')
    if (number_stars < 10)or(Zs<1):
        nu_new_stars*=0.0
    nu_new_stars[np.isnan(nu_new_stars)]    =   0.0
    del star_vz,star_vz2,star_z,star_z2

    #print('nu_dm',nu_dm)
    #print('nu_new_stars',nu_new_stars)
    Result.append(scale)
    Result.append(gas_mass)
    Result.append(molecular_gas_mass)
    Result.append(star_mass)
    Result.append(star_formation)
    Result.append(sfr_10)
    Result.append(sfr_01)
    Result.append(np.average(surface_density))
    Result.append(np.average(surface_density,weights=surface_density))
    Result.append(np.mean(molecular_surface_density))
    Result.append(np.average(molecular_surface_density,weights=surface_density))
    Result.append(np.average(new_stellar_density))
    Result.append(np.average(new_stellar_density,weights=surface_density))
    Result.append(np.average(sigma_vel))
    Result.append(np.average(sigma_vel,weights=surface_density))
    Result.append(star_sigma)
    Result.append(np.average(v_diff))
    Result.append(np.average(v_diff,weights=surface_density))
    Result.append(np.average(sound_speed))
    Result.append(np.average(sound_speed,weights=surface_density))
    Result.append(np.average(gas_H))
    Result.append(np.average(gas_H,weights=surface_density))
    Result.append(np.average(Zg))
    Result.append(np.average(Zg,weights=surface_density))
    Result.append(Zs)
    Result.append(np.average(nu_gas))
    Result.append(np.average(nu_gas,weights=surface_density))
    Result.append(np.average(nu_star))
    Result.append(np.average(nu_star,weights=surface_density))
    Result.append(np.average(nu_new_stars))
    Result.append(np.average(nu_new_stars,weights=surface_density))
    Result.append(np.average(nu_dm))
    Result.append(np.average(nu_dm,weights=surface_density))
    Result.append(np.average(galaxy_radius))
    Result.append(np.average(omega))
    Result.append(np.average(omega,weights=surface_density))
    Result.append(np.average(average_omega))
    Result.append(np.average(average_omega,weights=surface_density))
    Result.append(np.average(average_vorticity))
    Result.append(np.average(average_vorticity,weights=surface_density))
    Result.append(np.average(omega_star))
    Result.append(np.average(omega_star,weights=surface_density))
    Result.append(np.average(omega_dm))
    Result.append(np.average(omega_dm,weights=surface_density))
    Result.append(np.average(kappa_star))
    Result.append(np.average(kappa_star,weights=surface_density))
    Result.append(np.average(kappa_dm))
    Result.append(np.average(kappa_dm,weights=surface_density))
    Result.append(Iz)
    Result.append(Lz)
    Result.append(rot)
    Result.append(np.average(G*surface_density/(3.0*sigma_vel**2)).in_units('1/pc'))
    Result.append(np.average(G*surface_density/v_diff**2).in_units('1/pc'))
    Result.append(np.average(G*surface_density/(3.0*sigma_vel**2+sound_speed**2)).in_units('1/pc'))
    Result.append(np.average(G*surface_density/(v_diff**2+sound_speed**2)).in_units('1/pc'))
    Result.append(np.average(G*surface_density/(3.0*sigma_vel**2),weights=surface_density).in_units('1/pc'))
    Result.append(np.average(G*surface_density/v_diff**2,weights=surface_density).in_units('1/pc'))
    Result.append(np.average(G*surface_density/(3.0*sigma_vel**2+sound_speed**2),weights=surface_density).in_units('1/pc'))
    Result.append(np.average(G*surface_density/(v_diff**2+sound_speed**2),weights=surface_density).in_units('1/pc'))
    Result.append(np.average(np.sqrt(3.0)*sigma_vel/sound_speed).in_units('dimensionless'))
    Result.append(np.average(v_diff/sound_speed).in_units('dimensionless'))
    Result.append(np.average(np.sqrt(3.0)*sigma_vel/sound_speed,weights=surface_density).in_units('dimensionless'))
    Result.append(np.average(v_diff/sound_speed,weights=surface_density).in_units('dimensionless'))
    Result.append(np.average(Zg/(np.sqrt(3.0)*sigma_vel)).in_units('Myr'))
    Result.append(np.average(Zg/v_diff).in_units('Myr'))
    Result.append(np.average(Zg/(3.0*sigma_vel**2+sound_speed**2)**0.5).in_units('Myr'))
    Result.append(np.average(Zg/(v_diff**2+sound_speed**2)**0.5).in_units('Myr'))
    Result.append(np.average(Zg/(np.sqrt(3.0)*sigma_vel),weights=surface_density).in_units('Myr'))
    Result.append(np.average(Zg/v_diff,weights=surface_density).in_units('Myr'))
    Result.append(np.average(Zg/(3.0*sigma_vel**2+sound_speed**2)**0.5,weights=surface_density).in_units('Myr'))
    Result.append(np.average(Zg/(v_diff**2+sound_speed**2)**0.5,weights=surface_density).in_units('Myr'))
    Result.append(np.average(1.0/nu_gas).in_units('Myr'))
    Result.append(np.average(1.0/(nu_gas**2+nu_new_stars**2+nu_star**2+nu_dm**2)**0.5).in_units('Myr'))
    Result.append(np.average(1.0/nu_gas,weights=surface_density).in_units('Myr'))
    Result.append(np.average(1.0/(nu_gas**2+nu_new_stars**2+nu_star**2+nu_dm**2)**0.5,weights=surface_density).in_units('Myr'))
    Result.append(v11)
    Result.append(v22)
    Result.append(v33)
    Result.append(k11)
    Result.append(k22)
    Result.append(k33)
    Result=np.array(Result)

    return Result

def result_table(Resultados):

    Row=[]
    for res in Resultados:
        Row.append([res])
    Columns=['scale','gas_mass','molecular_gas_mass','star_mass','star_formation','sfr_10Myr','sfr_1Myr','surface_density','surface_density_w','molecular_surface_density',  #7
        'molecular_surface_density_w','new_stellar_density','new_stellar_density_w','sigma_vel','sigma_vel_w','star_sigma','v_diff', #7 (14)
    'v_diff_w','sound_speed','sound_speed_w','gas_H','gas_H_w','Zg','Zg_w','Zs','nu_gas','nu_gas_w','nu_star', #13 (27)
    'nu_star_w','nu_new_stars','nu_new_stars_w','nu_dm','nu_dm_w','galaxy_radius','omega','omega_w',     #10 (37)
    'average_omega','average_omega_w','average_vorticity','average_vorticity_w','omega_star','omega_star_w',                       #6 (43)
    'omega_dm','omega_dm_w','kappa_star','kappa_star_w','kappa_dm','kappa_dm_w','Iz','Lz','rot','L1','L2','L4','L5','L1_w','L2_w',  #18(61)
    'L4_w','L5_w','Mach_1','Mach_2','Mach_1_w','Mach_2_w','tcross_1','tcross_2','tcross_4','tcross_5',     #14 (75)
    'tcross_1_w','tcross_2_w','tcross_4_w','tcross_5_w','t_gas','t_tot','t_gas_w','t_tot_w','v11', #17  (92)
    'v22','v33','k11','k22','k33']
    tab = Table(Row, names=Columns)
    return tab

def create_table(name_table):
    Columns=['scale','gas_mass','molecular_gas_mass','star_mass','star_formation','sfr_10Myr','sfr_1Myr','surface_density','surface_density_w','molecular_surface_density',  #7
        'molecular_surface_density_w','new_stellar_density','new_stellar_density_w','sigma_vel','sigma_vel_w','star_sigma','v_diff', #7 (14)
    'v_diff_w','sound_speed','sound_speed_w','gas_H','gas_H_w','Zg','Zg_w','Zs','nu_gas','nu_gas_w','nu_star', #13 (27)
    'nu_star_w','nu_new_stars','nu_new_stars_w','nu_dm','nu_dm_w','galaxy_radius','omega','omega_w',     #10 (37)
    'average_omega','average_omega_w','average_vorticity','average_vorticity_w','omega_star','omega_star_w',                       #6 (43)
    'omega_dm','omega_dm_w','kappa_star','kappa_star_w','kappa_dm','kappa_dm_w','Iz','Lz','rot','L1','L2','L4','L5','L1_w','L2_w',  #18(61)
    'L4_w','L5_w','Mach_1','Mach_2','Mach_1_w','Mach_2_w','tcross_1','tcross_2','tcross_4','tcross_5',     #14 (75)
    'tcross_1_w','tcross_2_w','tcross_4_w','tcross_5_w','t_gas','t_tot','t_gas_w','t_tot_w','v11', #17  (92)
    'v22','v33','k11','k22','k33']
    tabla=Table()
    for col in Columns:
        tabla[col]=Column()
    tabla.write(name_table,path='data',format='hdf5',serialize_meta=True)
    del tabla
    return 0

comm = MPI.COMM_WORLD
ncores = comm.Get_size()
id = comm.Get_rank()

cmdargs = sys.argv

data=cmdargs[-7]
C_DM=float(cmdargs[-6])
error=bool(int(cmdargs[-5]))
Lmin=float(cmdargs[-4])
Lmax=float(cmdargs[-3])
Nscales=int(cmdargs[-2])
Resolution=float(cmdargs[-1])
print(data)
simulation=data+"/G-"+data[-4:]

name_table='Files/'+data+'_multi_resolution'

if id==0:
    if os.path.isfile(name_table):
        print('table file exists')
    else:
        create_table(name_table)
        print('table file created')


ds=yt.load('../Sims/'+simulation)

eff=ds.parameters['StarMakerMassEfficiency']
nthres=ds.parameters['StarMakerOverDensityThreshold']
fej=ds.parameters['StarMassEjectionFraction']
Ms=ds.parameters['PointStellarMass']*ds.mass_unit.in_units('Msun')
a=ds.parameters['PointStellara']*ds.length_unit.in_units('pc')
b=ds.parameters['PointStellarb']*ds.length_unit.in_units('pc')
MDM=ds.parameters['PointSourceGravityConstant']*ds.mass_unit.in_units('Msun')*(np.log(C_DM+1.0)-C_DM/(C_DM+1.0))/(np.log(1.0+1.0)-1.0/(1.0+1.0))
rs=ds.parameters['PointSourceGravityCoreRadius']*ds.length_unit.in_units('pc')

grids=ds.refine_by**ds.index.max_level*ds.domain_dimensions[0]
dx=ds.length_unit.in_units('pc')/grids
Phi_0=G*MDM/rs**3/(np.log(1.0+C_DM)-C_DM/(1+C_DM))
Phi_0.convert_to_units('1/Myr**2')
dxs=1e-3*dx/rs
print(dxs)
"""
def DM_nu(R):
    return np.sqrt(G*MDM*(np.log(1.0+R/rs)-(R/rs)/(1.0+R/rs))/(np.log(1+C_DM)-C_DM/(1+C_DM))/(R+dx)**3).in_units('1/Myr')
def DM_omega(R):
    return np.sqrt(G*MDM*(np.log(1.0+R/rs)-(R/rs)/(1.0+R/rs))/(np.log(1+C_DM)-C_DM/(1+C_DM))/(R+dx)**3).in_units('1/Myr')
def DM_kappa(R):
    return np.sqrt(G*MDM/((R+dx)*(R+rs)**2)+DM_omega(R)**2).in_units('1/Myr')
"""
def DM_nu(R):
    x=R/rs
    return np.sqrt(Phi_0*(np.log(1+x+dxs)/(x+dxs)**3-1.0/((x+dxs)**2*(1+x+dxs)))).in_units('1/Myr')
def DM_omega(R):
    x=R/rs
    return np.sqrt(Phi_0*(np.log(1+x+dxs)/(x+dxs)**3-1.0/((x+dxs)**2*(1+x+dxs)))).in_units('1/Myr')
def DM_kappa(R):
    x=R/rs
    return np.sqrt(Phi_0*(np.log(1+x+dxs)/(x+dxs)**3-1.0/((x+dxs)**2*(1+x+dxs)) + 1.0/((x+dxs)*(1+x+dxs)**2))).in_units('1/Myr')

def Stellar_nu(R,error=False):
    if error:
        return np.sqrt(G*Ms/((a+b)**2+R**2)**1.5).in_units('1/Myr')
    return np.sqrt(G*Ms*(a+b)/(b*((a+b)**2+R**2)**1.5)).in_units('1/Myr')
def Stellar_omega(R):
    return np.sqrt(G*Ms/((a+b)**2+R**2)**1.5).in_units('1/Myr')
def Stellar_kappa(R):
    return np.sqrt(G*Ms*(4*(a+b)**2+R**2)/((a+b)**2+R**2)**2.5).in_units('1/Myr')

Critical=(a+b)/np.sqrt(2.0)
print('Critical',Critical)


########### Average_fields ###################################

L=float(ds.length_unit.in_units('kpc'))/8
Disk = ds.disk('c', [0., 0., 1.],(L, 'kpc'), (1, 'kpc'))
VC=Disk['vc'].in_units("pc/Myr")
R=Disk['Disk_Radius'].in_units("pc")
Masa=Disk['cell_mass'].in_units("Msun")

print('\n        Computing Vorticity and Omega fields        \n')
#### Average Omega Field ####
print(L*1000)
print('parsecs')
x,y=Velocity_profile(VC,R,500,0,L*1000,np.mean)
y=y/x
x=np.insert(x,0,0)
y=np.insert(y,0,2*y[0]-y[1])
function_omega=interp1d(x,y)
#### Average Vorticity Field ####
x,y=Vorticity_profile(VC,R,500,0,L*1000,np.mean)
function_vorticity=interp1d(x,y)
del VC,R,Masa,x,y

##############################################################
print('        Young Stars positions        \n')
dd=ds.all_data()

grids=ds.refine_by**ds.index.max_level*ds.domain_dimensions[0]
dx=ds.length_unit.in_units('pc')/grids
Np=int(50*kpc/dx)

disk_dd = dd.cut_region(["obj['Disk_H'].in_units('pc') < 5.0e3"])
proj = ds.proj('density', 2,data_source=disk_dd,weight_field='density')
promedio = proj.to_frb((50.0, 'kpc'), Np, center=[0.5,0.5,0.5])

x_i=promedio['x'].in_units('pc')
y_i=promedio['y'].in_units('pc')
d_i=promedio['dx'].in_units('pc')

high_res=d_i<1.5*dx
L_new=min(x_i[high_res].max()-x_i[high_res].min(),y_i[high_res].max()-y_i[high_res].min())
L_new=np.round(L_new/1000 -1*pc)*1000
Np=int(L_new*pc/dx)
L_new=L_new/1000.0

del x_i,y_i,d_i,high_res,proj,promedio

disk_dd = dd.cut_region(["obj['Disk_H'].in_units('pc') < 5.0e3"])
proj = ds.proj('density', 2,data_source=disk_dd)
integral = proj.to_frb((L_new, 'kpc'), Np, center=[0.5,0.5,0.5])
proj2 = ds.proj('density', 2,data_source=disk_dd,weight_field='density')
promedio = proj2.to_frb((L_new, 'kpc'), Np, center=[0.5,0.5,0.5])


##### Fields 2D #####

density=integral['density'].in_units('Msun/pc**2')
x_i=promedio['x'].in_units('pc')
y_i=promedio['y'].in_units('pc')
new_stellar_density=integral['all_density'].in_units('Msun/pc**2')


metals=promedio['metallicity']
temperature=promedio['temperature'].in_units('K')
ft=np.array(np.ones_like(temperature))
ft[temperature>1e4]=0
X=0.77*(1+3.1*metals**0.365)
Scomp=density*pc**2/Msun
Scomp.convert_to_units('dimensionless')
s=np.log(1+0.6*X)/(0.04*Scomp*metals)
delta=0.0712*(0.1/s+0.675)**(-2.8)
f=1.0-(1+(3*s/(4+4*delta))**(-5))**(-0.2)
f*=ft

molecular_density=f*density
velocity=promedio['VZ'].in_units('km/s')
velocity_squared=promedio['VZ2'].in_units('km**2/s**2')
zeta=promedio['Disk_Z'].in_units('pc')
zeta_squared=promedio['Disk_Z2'].in_units('pc**2')
sigma_vel=np.abs(velocity_squared-velocity**2)**0.5
gas_H=np.abs(zeta_squared-zeta**2)**0.5
Zg          =   gas_H*2*np.sqrt(3.0)/np.pi
nu_gas=np.sqrt(np.pi*G*density/Zg).in_units('1/Myr')
v_diff=promedio['vturb'].in_units('km/s')
galaxy_radius=promedio['Global_Disk_Radius'].in_units('pc')
print('extrema',galaxy_radius.min(),galaxy_radius.max())
sound_speed=promedio['sound_speed'].in_units('km/s')
omega=promedio['Omega'].in_units('1/Myr')

del velocity,velocity_squared,zeta,zeta_squared,metals,temperature,ft,f,Scomp,delta,s,X

#### Fields 3D #####
Disk = ds.disk('c', [0., 0., 1.],(0.707*L_new+0.0012*Lmax, 'kpc'), (3, 'kpc'))
print('Disk_radius',0.707*L_new+0.0012*Lmax,0.707*L_new,0.0012*Lmax)

xg=Disk['x'].in_units('pc')
yg=Disk['y'].in_units('pc')
zg=Disk['z'].in_units('pc')

print('extrema x', xg.min(),xg.max())

mass=Disk['cell_mass'].in_units('Msun')

vxg=Disk['velocity_x'].in_units('km/s')
vyg=Disk['velocity_y'].in_units('km/s')
vzg=Disk['velocity_z'].in_units('km/s')

r_gal=Disk['Global_Disk_Radius'].in_units('pc')

star_formation=Disk['sfr_mass']/(5*Myr)
star_formation.convert_to_units('Msun/yr')
star_mass=Disk['particle_mass'].in_units('Msun')
star_age=Disk['age'].in_units('Myr')
star_vz=Disk['particle_velocity_z'].in_units('km/s')
star_x=Disk['particle_position_x'].in_units('pc')
star_y=Disk['particle_position_y'].in_units('pc')
star_z=Disk['particle_position_z'].in_units('pc')
print('extrema xs',star_x.min(),star_x.max())

#### Derived fields #########

average_omega=function_omega(galaxy_radius.in_units('pc'))/Myr
average_vorticity=function_vorticity(galaxy_radius.in_units('pc'))/Myr
nu_star=Stellar_nu(galaxy_radius.in_units('pc'),error=error)
omega_star=Stellar_omega(galaxy_radius.in_units('pc'))
kappa_star=Stellar_kappa(galaxy_radius.in_units('pc'))
nu_dm=DM_nu(galaxy_radius.in_units('pc'))
omega_dm=DM_omega(galaxy_radius.in_units('pc'))
kappa_dm=DM_kappa(galaxy_radius.in_units('pc'))
Ofield=Stellar_omega(r_gal)**2+DM_omega(r_gal)**2
Kfield=Stellar_kappa(r_gal)**2+DM_kappa(r_gal)**2
Nfield=Stellar_nu(r_gal,error=error)**2+DM_nu(r_gal)**2
####################
threshold=np.array(density)>1
X_p,Y_p=x_i[threshold],y_i[threshold]

indices = np.arange(0,len(X_p),1,dtype=int)
np.random.shuffle(indices)
indices=indices[:10000]

X_p=X_p[indices]
Y_p=Y_p[indices]

M=np.zeros((len(X_p),2))
M[:,0]=X_p
M[:,1]=Y_p

scales=np.round(0.1*np.exp(np.linspace(np.log(Lmin),np.log(Lmax),Nscales)))*10


print('        Scale Loop        \n')

for scale in scales:
    print('        This scale : %f\t     \n' %scale)

    distance = sch.distance.pdist(M)   # vector of (100 choose 2) pairwise distances
    Link = fastcluster.linkage(distance, method='complete')
    ind = sch.fcluster(Link, scale, 'distance')

    xcm=[]
    ycm=[]

    for j in set(ind):
        xcm.append(np.average(X_p[ind == j]))
        ycm.append(np.average(Y_p[ind == j]))

    Ncm = len(xcm)
    xcm=np.array(xcm)
    ycm=np.array(ycm)

    if Ncm>1000:
        indices = np.arange(0,len(xcm),1,dtype=int)
        np.random.shuffle(indices)
        indices=indices[:1000]
        xcm=xcm[indices]
        ycm=ycm[indices]

    print('        Number of regions :%d\t     \n' %len(xcm))

    part=int(len(xcm)/ncores)+1

    xpart=xcm[id*part:(id+1)*part]
    ypart=ycm[id*part:(id+1)*part]
    if len(xpart)>0:
        for xo,yo in zip(xpart,ypart):
            Resultado=Quantities(ds,xo*pc,yo*pc,scale,error,
            density,new_stellar_density,molecular_density,x_i,y_i,sigma_vel,gas_H,
            nu_gas,v_diff,galaxy_radius,sound_speed,omega,xg,yg,mass,
            vxg,vyg,vzg,r_gal,star_formation,star_mass,star_age,
            star_vz,star_x,star_y,star_z,average_omega,average_vorticity,nu_star,omega_star,kappa_star,nu_dm,
            omega_dm,kappa_dm,Ofield,Kfield,Nfield,dx)

            new_tabla=result_table(Resultado)
            tabla=Table.read(name_table,path='data',format='hdf5')

            new_tabla=vstack([tabla,new_tabla])
            new_tabla.write(name_table,path='data',format='hdf5',serialize_meta=True,overwrite=True)
            del tabla
