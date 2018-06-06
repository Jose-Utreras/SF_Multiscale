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

def Quantities(ds,xo,yo,ls,Resolution,error):
    print(id)
    Result=[]
    Disk = ds.disk([xo,yo,0.5], [0., 0., 1.],(0.5*ls/1000.0, 'kpc'), (1, 'kpc'))
    gas_mass=Disk['cell_mass'].in_units('Msun')
    xdisk=Disk['x'].in_units('pc')
    ydisk=Disk['y'].in_units('pc')
    zdisk=Disk['z'].in_units('pc')
    xdisk_cm=np.average(xdisk,weights=gas_mass)
    ydisk_cm=np.average(ydisk,weights=gas_mass)
    zdisk_cm=np.average(zdisk,weights=gas_mass)
    xdisk-=xdisk_cm
    ydisk-=ydisk_cm
    zdisk-=zdisk_cm
    del xdisk_cm,ydisk_cm,zdisk_cm
    vx=Disk['velocity_x'].in_units('pc/Myr')
    vy=Disk['velocity_y'].in_units('pc/Myr')
    vz=Disk['velocity_z'].in_units('pc/Myr')
    vx_cm=np.average(vx,weights=gas_mass)
    vy_cm=np.average(vy,weights=gas_mass)
    vz_cm=np.average(vz,weights=gas_mass)
    vx-=vx_cm
    vy-=vy_cm
    vz-=vz_cm
    k11=(0.5*gas_mass*vx**2).sum()
    k22=(0.5*gas_mass*vy**2).sum()
    k33=(0.5*gas_mass*vz**2).sum()
    k11.convert_to_units('Msun*pc**2/Myr**2')
    k22.convert_to_units('Msun*pc**2/Myr**2')
    k33.convert_to_units('Msun*pc**2/Myr**2')

    del vx_cm,vy_cm,vz_cm
    Iz=(gas_mass*(xdisk**2+ydisk**2)).sum()
    Lz=(gas_mass*(xdisk*vy-ydisk*vx)).sum()
    rot=Lz/Iz
    Lx=(gas_mass*(ydisk*vz-zdisk*vy)).sum()
    Ly=(gas_mass*(zdisk*vx-xdisk*vz)).sum()
    angle=np.arccos(Lz/np.sqrt(Lx**2+Ly**2+Lz**2))
    del Lx,Ly

    aux_R=Disk['Global_Disk_Radius'].in_units('pc')
    OT=Stellar_omega(aux_R)**2+DM_omega(aux_R)**2
    KT=Stellar_kappa(aux_R)**2+DM_kappa(aux_R)**2
    NT=Stellar_nu(aux_R,error=error)**2+DM_nu(aux_R)**2

    v11=(gas_mass*(xdisk**2)*(3*OT-KT)).sum()
    v22=-(gas_mass*(ydisk**2)*OT).sum()
    v33=-(gas_mass*(zdisk**2)*NT).sum()
    v11.convert_to_units('Msun*pc**2/Myr**2')
    v22.convert_to_units('Msun*pc**2/Myr**2')
    v33.convert_to_units('Msun*pc**2/Myr**2')
    print('global radius',aux_R)
    del xdisk,ydisk,vx,vy,vz,aux_R,OT,KT,NT

    gas_mass=gas_mass.sum()
    proj = ds.proj('density', 2,data_source=Disk)
    proj2 = ds.proj('density', 2,data_source=Disk,weight_field='density')
    width=(ls,'pc')
    NN=int(ls/(Resolution))
    res=[NN,NN]
    integral = proj.to_frb(width, res, center=[xo,yo,0.5])
    promedio = proj2.to_frb(width, res, center=[xo,yo,0.5])
    Radius=promedio['Disk_Radius'].in_units('pc')
    bad_data=(Radius>0.5*ls*pc/1.1)|(np.isnan(Radius))
    good_data=~bad_data
    surface_density=integral['density'].in_units('Msun/pc**2')
    new_stellar_density=integral['all_density'].in_units('Msun/pc**2')
    tau=0.066*surface_density*pc*pc/Msun
    tau.convert_to_units('dimensionless')
    X=0.77*(1+3.1)
    s=np.log(1+0.6*X+0.01*X**2)/(0.6*tau)
    s[s>=2]=2
    f=1-0.75*s/(1+0.25*s)
    molecular_surface_density=f*surface_density
    frac=molecular_surface_density.sum()/surface_density.sum()
    molecular_gas_mass=frac*gas_mass
    del X, tau,s,f,frac

    velocity=promedio['VZ'].in_units('km/s')
    velocity_squared=promedio['VZ2'].in_units('km**2/s**2')
    zeta=promedio['Disk_Z'].in_units('pc')
    zeta_squared=promedio['Disk_Z2'].in_units('pc**2')
    sigma_vel=np.abs(velocity_squared-velocity**2)**0.5
    gas_H=np.abs(zeta_squared-zeta**2)**0.5
    Zg=gas_H*2*np.sqrt(3.0)/np.pi
    nu_gas=np.sqrt(np.pi*G*surface_density/Zg).in_units('1/Myr')

    del velocity, velocity_squared,zeta,zeta_squared

    v_diff=promedio['vturb'].in_units('km/s')
    v_turb=promedio['sturb'].in_units('km/s')
    galaxy_radius=promedio['Global_Disk_Radius'].in_units('pc')
    sound_speed=promedio['sound_speed'].in_units('km/s')
    average_omega=function_omega(galaxy_radius.in_units('pc'))/Myr
    average_vorticity=function_vorticity(galaxy_radius.in_units('pc'))/Myr
    omega=promedio['Omega'].in_units('1/Myr')
    vorticity=promedio['vorticity_z'].in_units('1/Myr')
    nu_star=Stellar_nu(galaxy_radius.in_units('pc'),error=error)
    omega_star=Stellar_omega(galaxy_radius.in_units('pc'))
    kappa_star=Stellar_kappa(galaxy_radius.in_units('pc'))

    nu_dm=DM_nu(galaxy_radius.in_units('pc'))
    omega_dm=DM_omega(galaxy_radius.in_units('pc'))
    kappa_dm=DM_kappa(galaxy_radius.in_units('pc'))

    star_formation=Disk['sfr_mass'].sum()/(5*Myr)
    star_formation.convert_to_units('Msun/yr')
    star_mass=Disk['particle_mass'].in_units('Msun')
    number_stars=len(star_mass)
    star_vz=Disk['particle_velocity_z'].in_units('km/s')
    star_vz2=star_vz[:]
    star_vz=np.average(star_vz,weights=star_mass)
    star_sigma=np.sqrt(np.average((star_vz2-star_vz)**2,weights=star_mass))

    star_z=Disk['particle_position_z'].in_units('pc')
    star_z2=star_z[:]
    star_z=np.average(star_z,weights=star_mass)
    star_H=np.sqrt(np.average((star_z2-star_z)**2,weights=star_mass))
    Zs=star_H*2*np.sqrt(3.0)/np.pi
    star_mass=star_mass.sum()
    nu_new_stars=np.sqrt(np.pi*G*new_stellar_density/Zs).in_units('1/Myr')
    if (number_stars < 10)or(Zs<1):
        nu_new_stars*=0.0
    del star_vz,star_vz2,star_z,star_z2

    surface_density=surface_density[good_data]
    molecular_surface_density=molecular_surface_density[good_data]
    new_stellar_density=new_stellar_density[good_data]
    nu_new_stars=nu_new_stars[good_data]
    sigma_vel=sigma_vel[good_data]
    v_diff=v_diff[good_data]
    v_turb=v_turb[good_data]
    sound_speed=sound_speed[good_data]
    gas_H=gas_H[good_data]
    Zg=Zg[good_data]
    nu_gas=nu_gas[good_data]
    nu_star=nu_star[good_data]
    nu_dm=nu_dm[good_data]
    galaxy_radius=galaxy_radius[good_data]
    omega=omega[good_data]
    average_omega=average_omega[good_data]
    vorticity=vorticity[good_data]
    average_vorticity=average_vorticity[good_data]
    omega_star=omega_star[good_data]
    omega_dm=omega_dm[good_data]
    kappa_star=kappa_star[good_data]
    kappa_dm=kappa_dm[good_data]
    print('nu_dm',nu_dm)
    print('nu_new_stars',nu_new_stars)
    Result.append(scale)
    Result.append(gas_mass)
    Result.append(molecular_gas_mass)
    Result.append(star_mass)
    Result.append(star_formation)
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
    Result.append(np.average(v_turb))
    Result.append(np.average(v_turb,weights=surface_density))
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
    Result.append(np.average(vorticity))
    Result.append(np.average(vorticity,weights=surface_density))
    Result.append(np.average(omega))
    Result.append(np.average(omega,weights=surface_density))
    Result.append(np.average(vorticity))
    Result.append(np.average(vorticity,weights=surface_density))
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
    Result.append(np.average(G*surface_density/v_turb**2).in_units('1/pc'))
    Result.append(np.average(G*surface_density/(3.0*sigma_vel**2+sound_speed**2)).in_units('1/pc'))
    Result.append(np.average(G*surface_density/(v_diff**2+sound_speed**2)).in_units('1/pc'))
    Result.append(np.average(G*surface_density/(v_turb**2+sound_speed**2)).in_units('1/pc'))
    Result.append(np.average(G*surface_density/(3.0*sigma_vel**2),weights=surface_density).in_units('1/pc'))
    Result.append(np.average(G*surface_density/v_diff**2,weights=surface_density).in_units('1/pc'))
    Result.append(np.average(G*surface_density/v_turb**2,weights=surface_density).in_units('1/pc'))
    Result.append(np.average(G*surface_density/(3.0*sigma_vel**2+sound_speed**2),weights=surface_density).in_units('1/pc'))
    Result.append(np.average(G*surface_density/(v_diff**2+sound_speed**2),weights=surface_density).in_units('1/pc'))
    Result.append(np.average(G*surface_density/(v_turb**2+sound_speed**2),weights=surface_density).in_units('1/pc'))
    Result.append(np.average(np.sqrt(3.0)*sigma_vel/sound_speed).in_units('dimensionless'))
    Result.append(np.average(v_diff/sound_speed).in_units('dimensionless'))
    Result.append(np.average(v_turb/sound_speed).in_units('dimensionless'))
    Result.append(np.average(np.sqrt(3.0)*sigma_vel/sound_speed,weights=surface_density).in_units('dimensionless'))
    Result.append(np.average(v_diff/sound_speed,weights=surface_density).in_units('dimensionless'))
    Result.append(np.average(v_turb/sound_speed,weights=surface_density).in_units('dimensionless'))
    Result.append(np.average(Zg/(np.sqrt(3.0)*sigma_vel)).in_units('Myr'))
    Result.append(np.average(Zg/v_diff).in_units('Myr'))
    Result.append(np.average(Zg/v_turb).in_units('Myr'))
    Result.append(np.average(Zg/(3.0*sigma_vel**2+sound_speed**2)**0.5).in_units('Myr'))
    Result.append(np.average(Zg/(v_diff**2+sound_speed**2)**0.5).in_units('Myr'))
    Result.append(np.average(Zg/(v_turb**2+sound_speed**2)**0.5).in_units('Myr'))
    Result.append(np.average(Zg/(np.sqrt(3.0)*sigma_vel),weights=surface_density).in_units('Myr'))
    Result.append(np.average(Zg/v_diff,weights=surface_density).in_units('Myr'))
    Result.append(np.average(Zg/v_turb,weights=surface_density).in_units('Myr'))
    Result.append(np.average(Zg/(3.0*sigma_vel**2+sound_speed**2)**0.5,weights=surface_density).in_units('Myr'))
    Result.append(np.average(Zg/(v_diff**2+sound_speed**2)**0.5,weights=surface_density).in_units('Myr'))
    Result.append(np.average(Zg/(v_turb**2+sound_speed**2)**0.5,weights=surface_density).in_units('Myr'))
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

ds=yt.load('../Circulation/Sims/'+simulation)

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
dxs=0.5*dx/rs
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
    return np.sqrt(Phi_0*(np.log(1+x)/(x+dxs)**3-1.0/((x+dxs)**2*(1+x)))).in_units('1/Myr')
def DM_omega(R):
    x=R/rs
    return np.sqrt(Phi_0*(np.log(1+x)/(x+dxs)**3-1.0/((x+dxs)**2*(1+x)))).in_units('1/Myr')
def DM_kappa(R):
    x=R/rs
    return np.sqrt(Phi_0*(np.log(1+x)/(x+dxs)**3-1.0/((x+dxs)**2*(1+x)) + 1.0/((x+dxs)*(1+x)**2))).in_units('1/Myr')

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

MASS=dd['sfr_mass'].in_units('Msun')
Observable=MASS>1

xs=np.array(dd['particle_position_x'].in_units('code_length'))
ys=np.array(dd['particle_position_y'].in_units('code_length'))
mass=np.array(dd['particle_mass'].in_units('Msun'))

xs=xs[Observable]
ys=ys[Observable]
mass=mass[Observable]

M=np.zeros((len(xs),2))
M[:,0]=xs
M[:,1]=ys

scales=np.exp(np.linspace(np.log(Lmin),np.log(Lmax),Nscales))

del MASS,Observable

print('        Scale Loop        \n')

for scale in scales:
    print('        This scale : %f\t     \n' %scale)
    l=1.1*scale/float(ds.length_unit.in_units('pc'))
    ls=1.1*scale

    distance = sch.distance.pdist(M)   # vector of (100 choose 2) pairwise distances
    Link = fastcluster.linkage(distance, method='complete')
    ind = sch.fcluster(Link, l, 'distance')

    xcm=[]
    ycm=[]

    for j in set(ind):
        temp_mass=mass[ind==j].sum()
        if temp_mass>10:
            xcm.append(np.average(xs[ind == j],weights=mass[ind == j]))
            ycm.append(np.average(ys[ind == j],weights=mass[ind == j]))

    del temp_mass

    print('        Number of regions :%d\t     \n' %len(xcm))

    part=int(len(xcm)/ncores)+1

    xpart=xcm[id*part:(id+1)*part]
    ypart=ycm[id*part:(id+1)*part]

    if len(xpart)>0:
        for xo,yo in zip(xpart,ypart):
            Resultado=Quantities(ds,xo,yo,ls,Resolution,error)
            print(Resultado)
"""
    name_table='Files/'+data+'_multi_resolution'
    if os.path_is_file(name_table):
        print('File exists')
    else:
        tabla=Table()
        tabla.write(name_table,format='hdf5',serialize_meta=True,overwrite=True)
        del tabla
"""
