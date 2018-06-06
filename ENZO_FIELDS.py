import matplotlib
matplotlib.use('Agg')
import yt
import numpy as np
from scipy.interpolate import interp1d
from yt.utilities.physical_constants import G
from yt.units import kpc,pc,km,second,yr,Myr
from yt.data_objects.particle_filters import add_particle_filter
from yt.fields.derived_field import \
    ValidateGridType, \
    ValidateParameter, \
    ValidateSpatial, \
    NeedsParameter
import h5py
from yt.funcs import \
    just_one
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

def _Disk_H(field, data):
    center = data.get_field_parameter('center')
    z = data["z"] - center[2]
    return np.abs(z)
def _Disk_Radius(field, data):
    center = data.get_field_parameter('center')
    x = data["x"] - center[0]
    y = data["y"] - center[1]
    r = np.sqrt(x*x+y*y)
    return r
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
def _wturb(field, data):

    td  = data[ftype, "density"][sl_right,sl_center,sl_center]
    td += data[ftype, "density"][sl_left,sl_center,sl_center]
    td += data[ftype, "density"][sl_center,sl_right,sl_center]
    td += data[ftype, "density"][sl_center,sl_left,sl_center]
    td += data[ftype, "density"][sl_center,sl_center,sl_right]
    td += data[ftype, "density"][sl_center,sl_center,sl_left]
    td += data[ftype, "density"][sl_center,sl_center,sl_center]

    fx  = data[ftype, "px"][sl_right,sl_center,sl_center]/td
    fx += data[ftype, "px"][sl_left,sl_center,sl_center]/td
    fx += data[ftype, "px"][sl_center,sl_right,sl_center]/td
    fx += data[ftype, "px"][sl_center,sl_left,sl_center]/td
    fx += data[ftype, "px"][sl_center,sl_center,sl_right]/td
    fx += data[ftype, "px"][sl_center,sl_center,sl_left]/td
    fx += data[ftype, "px"][sl_center,sl_center,sl_center]/td

    FX  = (data[ftype, "velocity_x"][sl_right,sl_center,sl_center] -fx)**2*data[ftype, "density"][sl_right,sl_center,sl_center]
    FX += (data[ftype, "velocity_x"][sl_left,sl_center,sl_center]  -fx)**2*data[ftype, "density"][sl_left,sl_center,sl_center]
    FX += (data[ftype, "velocity_x"][sl_center,sl_right,sl_center] -fx)**2*data[ftype, "density"][sl_center,sl_right,sl_center]
    FX += (data[ftype, "velocity_x"][sl_center,sl_left,sl_center]  -fx)**2*data[ftype, "density"][sl_center,sl_left,sl_center]
    FX += (data[ftype, "velocity_x"][sl_center,sl_center,sl_right] -fx)**2*data[ftype, "density"][sl_center,sl_center,sl_right]
    FX += (data[ftype, "velocity_x"][sl_center,sl_center,sl_left]  -fx)**2*data[ftype, "density"][sl_center,sl_center,sl_left]
    FX += (data[ftype, "velocity_x"][sl_center,sl_center,sl_center]-fx)**2*data[ftype, "density"][sl_center,sl_center,sl_center]

    fy  = data[ftype, "py"][sl_right,sl_center,sl_center]/td
    fy += data[ftype, "py"][sl_left,sl_center,sl_center]/td
    fy += data[ftype, "py"][sl_center,sl_right,sl_center]/td
    fy += data[ftype, "py"][sl_center,sl_left,sl_center]/td
    fy += data[ftype, "py"][sl_center,sl_center,sl_right]/td
    fy += data[ftype, "py"][sl_center,sl_center,sl_left]/td
    fy += data[ftype, "py"][sl_center,sl_center,sl_center]/td

    FY  = (data[ftype, "velocity_y"][sl_right,sl_center,sl_center] -fy)**2*data[ftype, "density"][sl_right,sl_center,sl_center]
    FY += (data[ftype, "velocity_y"][sl_left,sl_center,sl_center]  -fy)**2*data[ftype, "density"][sl_left,sl_center,sl_center]
    FY += (data[ftype, "velocity_y"][sl_center,sl_right,sl_center] -fy)**2*data[ftype, "density"][sl_center,sl_right,sl_center]
    FY += (data[ftype, "velocity_y"][sl_center,sl_left,sl_center]  -fy)**2*data[ftype, "density"][sl_center,sl_left,sl_center]
    FY += (data[ftype, "velocity_y"][sl_center,sl_center,sl_right] -fy)**2*data[ftype, "density"][sl_center,sl_center,sl_right]
    FY += (data[ftype, "velocity_y"][sl_center,sl_center,sl_left]  -fy)**2*data[ftype, "density"][sl_center,sl_center,sl_left]
    FY += (data[ftype, "velocity_y"][sl_center,sl_center,sl_center]-fy)**2*data[ftype, "density"][sl_center,sl_center,sl_center]

    fz  = data[ftype, "pz"][sl_right,sl_center,sl_center]/td
    fz += data[ftype, "pz"][sl_left,sl_center,sl_center]/td
    fz += data[ftype, "pz"][sl_center,sl_right,sl_center]/td
    fz += data[ftype, "pz"][sl_center,sl_left,sl_center]/td
    fz += data[ftype, "pz"][sl_center,sl_center,sl_right]/td
    fz += data[ftype, "pz"][sl_center,sl_center,sl_left]/td
    fz += data[ftype, "pz"][sl_center,sl_center,sl_center]/td

    FZ  = (data[ftype, "velocity_z"][sl_right,sl_center,sl_center] -fz)**2*data[ftype, "density"][sl_right,sl_center,sl_center]
    FZ += (data[ftype, "velocity_z"][sl_left,sl_center,sl_center]  -fz)**2*data[ftype, "density"][sl_left,sl_center,sl_center]
    FZ += (data[ftype, "velocity_z"][sl_center,sl_right,sl_center] -fz)**2*data[ftype, "density"][sl_center,sl_right,sl_center]
    FZ += (data[ftype, "velocity_z"][sl_center,sl_left,sl_center]  -fz)**2*data[ftype, "density"][sl_center,sl_left,sl_center]
    FZ += (data[ftype, "velocity_z"][sl_center,sl_center,sl_right] -fz)**2*data[ftype, "density"][sl_center,sl_center,sl_right]
    FZ += (data[ftype, "velocity_z"][sl_center,sl_center,sl_left]  -fz)**2*data[ftype, "density"][sl_center,sl_center,sl_left]
    FZ += (data[ftype, "velocity_z"][sl_center,sl_center,sl_center]-fz)**2*data[ftype, "density"][sl_center,sl_center,sl_center]

    FX/=td
    FY/=td
    FZ/=td
    new_field = data.ds.arr(np.zeros_like(data[ftype, "velocity_z"],
                                          dtype=np.float64),
                            np.sqrt(FX).units)
    new_field[sl_center, sl_center, sl_center] = np.sqrt(FX+FY+FZ)
    return new_field
def _vz_squared(field,data):
        if data.has_field_parameter("bulk_velocity"):
                bv = data.get_field_parameter("bulk_velocity").in_units("cm/s")
        else:
                bv = data.ds.arr(np.zeros(3), "cm/s")
        vz = data["gas","velocity_z"] - bv[2]

        return vz**2
def _vz(field,data):
        if data.has_field_parameter("bulk_velocity"):
                bv = data.get_field_parameter("bulk_velocity").in_units("cm/s")
        else:
                bv = data.ds.arr(np.zeros(3), "cm/s")
        vz = data["gas","velocity_z"] - bv[2]

        return vz
def _px(field,data):
        if data.has_field_parameter("bulk_velocity"):
                bv = data.get_field_parameter("bulk_velocity").in_units("cm/s")
        else:
                bv = data.ds.arr(np.zeros(3), "cm/s")
        vx = data["gas","velocity_x"] - bv[0]
        return vx*data["gas","density"]
def _py(field,data):
        if data.has_field_parameter("bulk_velocity"):
                bv = data.get_field_parameter("bulk_velocity").in_units("cm/s")
        else:
                bv = data.ds.arr(np.zeros(3), "cm/s")
        vx = data["gas","velocity_y"] - bv[0]
        return vx*data["gas","density"]
def _pz(field,data):
        if data.has_field_parameter("bulk_velocity"):
                bv = data.get_field_parameter("bulk_velocity").in_units("cm/s")
        else:
                bv = data.ds.arr(np.zeros(3), "cm/s")
        vx = data["gas","velocity_z"] - bv[0]
        return vx*data["gas","density"]
def _sound_speed_2(field, data):
    tr = data.ds.gamma * data["gas", "pressure"] / data["gas", "density"]
    return tr
def _sound_speed_rep(field, data):
    tr = data.ds.gamma * data["gas", "pressure"] / data["gas", "density"]
    return 1.0/np.sqrt(tr)
def _sound_speed_rep_2(field, data):
    tr = data.ds.gamma * data["gas", "pressure"] / data["gas", "density"]
    return 1.0/tr
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
def _rho_J(field, data):
    return 16.0*data['density']*G*data['dx']**2/(np.pi*data["sound_speed"]**2)
def young_stars(pfilter, data):
    age = data.ds.current_time - data[pfilter.filtered_type, "creation_time"]
    filter = np.logical_and(age.in_units('Myr') <= 10.0, age.in_units('Myr') >= 5.0)
    return filter
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
def _true_mass(field,data):
    mass=data[('particle_mass')].in_units('Msun')
    age=np.array(data[('age')].in_units('Myr'))
    tdyn=np.array(data[('dynamical_time')].in_units('Myr'))

    T=(age)/tdyn
    T[np.where(T>100)]=100
    f=(1-fej*(1-(1+T)*np.exp(-T)))
    mass = mass/f
    return mass

yt.add_field("Disk_H",function=_Disk_H,units="pc",take_log=False,validators=[ValidateParameter('center')])
yt.add_field("Disk_Radius",
             function=_Disk_Radius,
             units="cm",
             take_log=False,
             validators=[ValidateParameter('center')])
yt.add_field(("gas", "vz_squared"),function=_vz_squared,units="km**2/s**2")
yt.add_field(("gas", "vz"),function=_vz,units="km/s")
yt.add_field("sound_speed", function=_sound_speed, units=r"km/s")
yt.add_field("sound_speed_rep", function=_sound_speed_rep, units=r"s/km")
yt.add_field("sound_speed_2", function=_sound_speed_2, units=r"km**2/s**2")
yt.add_field("sound_speed_rep_2", function=_sound_speed_rep_2, units=r"s**2/km**2")
yt.add_field(("gas", "px"),function=_px,units="g/s/cm**2")
yt.add_field(("gas", "py"),function=_py,units="g/s/cm**2")
yt.add_field(("gas", "pz"),function=_pz,units="g/s/cm**2")
yt.add_field(("gas", "sturb"),function=_sturb,units="km/s",validators=vort_validators)
yt.add_field(("gas", "vturb"),function=_vturb,units="km/s",validators=vort_validators)
yt.add_field(("gas", "wturb"),function=_vturb,units="km/s",validators=vort_validators)
yt.add_field("Omega", function=_Omega,take_log=False, units=r"1/yr",validators=[ValidateParameter('bulk_velocity')])
yt.add_field("vc", function=_vc,take_log=False, units=r"km/s",validators=[ValidateParameter('bulk_velocity')])
yt.add_field("rho_J", function=_rho_J,take_log=False, units=r"dimensionless")
yt.add_field("sfr_mass",function=_sfr_mass,
             units="Msun",
             take_log=False,
             validators=[ValidateParameter('center')],particle_type='True')
yt.add_field("true_mass",function=_true_mass,
             units="Msun",
             take_log=False,
             validators=[ValidateParameter('center')],particle_type='True')
yt.add_particle_filter("young_stars", function=young_stars, filtered_type='all', requires=["particle_type"])
