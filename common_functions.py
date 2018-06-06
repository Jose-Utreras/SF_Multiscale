import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import fileinput
from scipy import fftpack
from scipy.interpolate import interp1d
from scipy.fftpack import fft, ifft
from scipy.optimize import curve_fit
from scipy.stats import kurtosis,skew
from photutils import CircularAperture
from photutils import aperture_photometry

def Heaviside(x):
    return np.piecewise(x,[x<0,x>=0], [lambda x: 0, lambda x: 1])

def distance(p,q):
    return np.sqrt((p[0]-q[0])**2+(p[1]-q[1])**2+(p[2]-q[2])**2)

def colorplot(number,n):
    dz=np.linspace(0,number,number+1)

    norm = plt.Normalize()
    colors = plt.cm.jet(norm(dz))
    return colors[n]

def gaussian(x, a, mu, sig):
    return a*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def gaussian_step(x, a, mu, sig,step):
    return step+a*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def radial_map(mapa):
    xmax,ymax=np.shape(mapa)

    X=np.array(list(np.reshape(range(xmax),(1,xmax)))*xmax)

    Y=X.T
    X=np.reshape(X,(xmax,xmax))-(xmax-1)/2
    Y=np.reshape(Y,(xmax,xmax))-(ymax-1)/2
    R=np.sqrt(X**2+Y**2)

    return R

def angle_map(mapa):
    xmax,ymax=np.shape(mapa)

    X=np.array(list(np.reshape(range(xmax),(1,xmax)))*xmax)

    Y=X.T
    X=np.reshape(X,(xmax,xmax))-(xmax-1)/2
    Y=np.reshape(Y,(xmax,xmax))-(ymax-1)/2
    return np.arctan2(Y,X)

def radial_map_N(N1,N2):
    xmax,ymax=N1,N2

    X=np.array(list(np.reshape(range(xmax),(1,xmax)))*xmax)

    Y=X.T
    X=np.reshape(X,(xmax,xmax))-(xmax-1)/2
    Y=np.reshape(Y,(xmax,xmax))-(ymax-1)/2
    R=np.sqrt(X**2+Y**2)

    return R

def change_word_infile(filename,text_to_search,replacement_text):
    with fileinput.FileInput(filename, inplace=True, backup='.bak') as file:
        for line in file:
            print(line.replace(text_to_search, replacement_text), end='')

def Fourier_2D(mapa,L):
    N=len(mapa)
    noise=np.random.normal(0,10,size=(N,N))
    dx=L/N
    k1 = fftpack.fftfreq(N,d=dx)

    kx,ky = np.meshgrid(k1, k1)
    KX = fftpack.fftshift( kx )
    KY = fftpack.fftshift( ky )
    K=np.sqrt(KX**2+KY**2)

    F1=fftpack.fft2( mapa )
    F1=fftpack.fftshift( F1 )

    return K,F1

def Fourier_map(mapa,k,Nbins):
    kedges=np.linspace(k.min(),k.max(),Nbins+1)
    kbins=0.5*(kedges[1:]+kedges[:-1])
    Profile=np.zeros(Nbins)
    for i in range(Nbins):
        ring=(kedges[i]<k)&(k<kedges[i+1])
        Profile[i]=np.median(mapa[ring])
    return kbins,Profile

def power_law_map(n,N,rc):

    Omega=np.ones((N,N))

    X=np.array(list(np.reshape(range(N),(1,N)))*N)

    Y=X.T
    X=np.reshape(X,(N,N))-0.5*(N-1)
    Y=np.reshape(Y,(N,N))-0.5*(N-1)
    R=np.sqrt(X**2+Y**2)
    R=2*R/N
    Omega/=(R+rc)**n
    Omega/=Omega[int(N/2)][-1]
    Vort=(2-n)*Omega

    return Vort

def Knumber(mapa,L):
    N=len(mapa)
    noise=np.random.normal(0,10,size=(N,N))
    dx=L/N
    k1 = fftpack.fftfreq(N,d=dx)

    kx,ky = np.meshgrid(k1, k1)
    KX = fftpack.fftshift( kx )
    KY = fftpack.fftshift( ky )
    K=np.sqrt(KX**2+KY**2)
    return K

def map_profile(mapa):
    N=len(mapa)
    R=2*radial_map_N(N,N)/N
    h=2*(np.percentile(mapa,75)-np.percentile(mapa,25))/N**(1.0/3.0)
    Nbins=np.percentile(mapa,99)-np.percentile(mapa,1)
    Nbins=4*int(Nbins/h)

    Redges=np.linspace(0,R.max(),Nbins+1)
    Rcen=0.5*(Redges[1:]+Redges[:-1])
    hist=np.zeros(Nbins)
    hstd=np.zeros(Nbins)
    weights=np.zeros(Nbins)

    for k in range(Nbins):
        ring=(Redges[k]<=R)&(R<Redges[k+1])
        hist[k]=np.mean(mapa[ring])
        hstd[k]=np.std(mapa[ring])
        weights[k]+=len(mapa[ring].ravel())


    hstd=np.insert(hstd,0,hstd[0])
    hist=np.insert(hist,0,2*hist[0]-hist[1])
    Rcen=np.insert(Rcen,0,0)
    weights=np.insert(weights,0,1)

    hist=np.insert(hist,len(hist),hist.min())
    Rcen=np.insert(Rcen,len(Rcen),np.sqrt(2)*1.5)
    hstd=np.insert(hstd,len(hstd),hstd[-1])
    weights=np.insert(weights,len(weights),1)

    n_seed=np.log(h.max()/hist[hist>0].min())/np.log(Rcen.max()/Rcen[Rcen>0].min())
    popt, pcov = curve_fit(two_functions, Rcen[(Rcen>0)&(hist>0)], hist[(Rcen>0)&(hist>0)],
        p0=[np.mean(hist),n_seed,n_seed ,np.mean(Rcen)])
    temp=two_functions(Rcen,*popt)
    bad_values=(np.isnan(temp))|(temp==np.inf)|(temp==-np.inf)
    temp[bad_values]=hist[bad_values]
    temp=(hist+temp)/2
    #temp[temp<0]=0
    return Rcen,temp,hstd,weights

def standard_deviation_from_map(mapa):
    N=len(mapa)
    R=2*radial_map_N(N,N)/N
    h=2*(np.percentile(mapa,75)-np.percentile(mapa,25))/N**(1.0/3.0)
    Nbins=np.percentile(mapa,99)-np.percentile(mapa,1)
    Nbins=int(Nbins/h)

    Redges=np.linspace(0,R.max(),Nbins+1)
    Rcen=0.5*(Redges[1:]+Redges[:-1])
    hstd=np.zeros(Nbins)
    weights=np.zeros(Nbins)
    wskew=np.zeros(Nbins)
    sskew=np.zeros(Nbins)

    for k in range(Nbins):
        ring=(Redges[k]<=R)&(R<Redges[k+1])
        hstd[k]=np.std(mapa[ring])
        weights[k]+=len(mapa[ring].ravel())
        nn=len(mapa[ring])
        wskew[k]=skew(mapa[ring])
        sskew[k]=np.sqrt(6*nn*(nn-1)/((nn-2)*(nn+1)*(nn+3)))


    return Rcen,hstd,weights,wskew,sskew

def get_distribution(mapa):
    new=mapa.ravel()
    h=np.percentile(new,75)-np.percentile(new,25)
    h*=2
    h/=len(new)**(1.0/3.0)
    Nbins=np.percentile(new,99.9)-np.percentile(new,0.1)
    Nbins=int(Nbins/h)

    h,bins=np.histogram(new,Nbins,normed=True)
    center=0.5*(bins[1:]+bins[:-1])

    center=center[h>0]
    h=h[h>0]
    """
    g=simpson_F(center,h)

    f1=interp1d(g/g[-1],center)

    thresh=f1(0.9999)

    new_h=h[center>thresh]
    new_c=center[center>thresh]

    lx=np.log(new_c)
    ly=np.log(new_h)

    m=np.sum((ly-ly[0])*(lx-lx[0]))/np.sum((lx-lx[0])**2)
    h[center>thresh]=new_h[0]*(new_c/new_c[0])**m
    """
    h=np.insert(h,0,0)
    center=np.insert(center,0,bins[0])

    h=np.insert(h,0,0)
    center=np.insert(center,0,-1e10)

    h=np.insert(h,len(h),0)
    center=np.insert(center,len(center),bins[-1])

    h=np.insert(h,len(h),0)
    center=np.insert(center,len(center),1e10)

    func=interp1d(center,h)
    return func, Nbins

def simpson_array(x,y):
    w=3*np.ones_like(y)
    w[::3]-=1
    w[0]=1
    w[-1]=1

    dx=3*(x[1]-x[0])/8
    suma=(y*w).sum()*dx
    return suma

def simpson_F(x,y):
    w=3*np.ones_like(y)
    w[::3]-=1
    w[0]=1
    w[-1]=1

    dx=3*(x[1]-x[0])/8
    suma=np.cumsum(y*w)*dx
    return suma

def map_from_profile(R,y,N):

    fun=interp1d(R,y)
    Radius=2*radial_map_N(N,N)/N
    return fun(Radius)

def two_functions(x,A,n1,n2,x0):
    B=A*x0**(-n1+n2)
    return np.piecewise(x,[x<x0,x>=x0],[lambda x: A/x**n1, lambda x: B/x**n2])

def square_function(x,a,b,c):
    return np.polyval([a,b,c],x)

def core_slope(x,a,b,c):
    return a/(b+x**2)**c

def symmetric_map(mapa):
    N=len(mapa)
    R=2*radial_map_N(N,N)/N
    h=2*(np.percentile(mapa,75)-np.percentile(mapa,25))/N**(1.0/3.0)
    Nbins=np.percentile(mapa,99.9)-np.percentile(mapa,0.1)
    Nbins=2*int(Nbins/h)
    Nbins=min(Nbins,int(N/3))
    Redges=np.linspace(0,R.max(),Nbins+1)
    Rcen=0.5*(Redges[1:]+Redges[:-1])

    A=np.zeros(Nbins)
    B=np.zeros(Nbins)
    C=np.zeros(Nbins)
    for k in range(Nbins):
        ring=(Redges[k]<=R)&(R<Redges[k+1])
        yaux=mapa[ring].ravel()
        raux=R[ring].ravel()
        yaux=yaux[raux.argsort()]
        raux=raux[raux.argsort()]
        ymed=np.median(yaux)
        smed=np.median(np.abs(yaux-ymed))*1.48
        correct=[(ymed-2*smed<yaux)&(yaux<ymed+2*smed)]
        popt, pcov = curve_fit(square_function, raux[correct], yaux[correct])
        A[k]=popt[0]
        B[k]=popt[1]
        C[k]=popt[2]

    xtest=np.linspace(0,1.6,100000)
    ytest=np.zeros_like(xtest)

    for k in range(Nbins+1):
        if k==0:
            x2=Rcen[k]
            kregion=xtest<x2
            a2,b2,c2=A[k],B[k],C[k]
            d2=x2-xtest[kregion]
            f2=a2*xtest[kregion]**2+b2*xtest[kregion]+c2
            ytest[kregion]=f2

        elif k==Nbins:
            x1=Rcen[k-1]
            kregion=x1<xtest
            a1,b1,c1=A[k-1],B[k-1],C[k-1]
            d1=xtest[kregion]-x1
            f1=a1*xtest[kregion]**2+b1*xtest[kregion]+c1
            ytest[kregion]=f1
        else:
            x1=Rcen[k-1]
            x2=Rcen[k]
            kregion=(x1<xtest)&(x2>xtest)
            a1,b1,c1=A[k-1],B[k-1],C[k-1]
            a2,b2,c2=A[k],B[k],C[k]

            d1=xtest[kregion]-x1
            d2=x2-xtest[kregion]
            D=d1+d2
            f1=a1*xtest[kregion]**2+b1*xtest[kregion]+c1
            f2=a2*xtest[kregion]**2+b2*xtest[kregion]+c2
            ytest[kregion]=(f1*d2+f2*d1)/D



    func=interp1d(xtest,ytest)
    return func(R)

def profile_from_circulation(mapa):
    N=len(mapa)
    radio=np.linspace(0.2,len(mapa)/2,int(len(mapa)*0.75))
    ap=[]
    ii=(N-1)/2.0
    for rad in radio:
        apertures = CircularAperture((ii,ii), rad)
        aux=aperture_photometry(mapa, apertures)
        ap.append(float(aux['aperture_sum']/(np.pi*2*rad**2)))
    ome=np.array(ap)
    ex_rad=np.linspace(radio.max()*1.01,1.35*radio.max(),int(len(mapa)*0.15))
    ex_ome=np.zeros_like(ex_rad)
    R=radial_map_N(N,N)
    angle=angle_map(mapa)
    r1=radio.max()
    for kk, r2 in enumerate(ex_rad):
        t2=np.arccos(r1/r2)
        phi=np.pi/2-2*t2
        q1=[(t2<angle)&(angle<0.5*np.pi-t2)]
        q2=[(t2+0.5*np.pi<angle)&(angle<np.pi-t2)]
        q3=[(t2-np.pi<angle)&(angle<-0.5*np.pi-t2)]
        q4=[(t2-0.5*np.pi<angle)&(angle<-t2)]
        rg=R>r2
        region=np.zeros_like(angle,dtype=bool)
        region[q1]=1
        region[q2]=1
        region[q3]=1
        region[q4]=1
        region[rg]=0
        factor=np.abs(2*phi*r2**2/len(mapa[region].ravel()))
        if factor >1e5:
            factor=1

        ex_ome[kk]=factor*mapa[region].sum()/(4*phi*r2**2)

    radio=np.array(list(radio)+list(ex_rad))
    ome=np.array(list(ome)+list(ex_ome))
    rad=radio/r1

    n=np.zeros_like(rad)
    n[1:-1]=(np.log(ome[2:])-np.log(ome[:-2]))/(np.log(rad[2:])-np.log(rad[:-2]))
    try:
        h1=np.log(rad[1])-np.log(rad[0])
        h2=np.log(rad[2])-np.log(rad[0])
        f0=np.log(ome[0])
        f1=np.log(ome[1])
        f2=np.log(ome[2])
        n[0]=(h2**2*(f1-f0)-h1**2*(f2-f0))/(h2*h1*(h2-h1))
    except:
        h2=np.log(rad[2])-np.log(rad[1])
        f1=np.log(ome[1])
        f2=np.log(ome[2])
        n[0]=(f2-f1)/h2

    high_radius=rad>1.2
    new_o=ome[high_radius]
    new_r=rad[high_radius]
    new_n=np.zeros_like(new_o)
    new_n[1:-1]=new_r[1:-1]*(new_o[2:]-new_o[:-2])/(new_r[2:]-new_r[:-2])/new_o[1:-1]
    new_n[0]=new_n[1]
    new_n[-1]=new_n[-2]
    n[high_radius]=new_n

    vort=(2+n)*ome
    #return rad,vort,ome,n
    rad=np.insert(rad,0,0)
    vort=np.insert(vort,0,2*vort[0]-vort[1])
    rad=np.insert(rad,len(rad),1.5)
    vort=np.insert(vort,len(vort),vort[-1])
    func=interp1d(rad,vort)
    R=2*radial_map_N(N,N)/N
    new_map=func(R)
    return new_map

def profile_from_circulation2(mapa):
    N=len(mapa)
    radio=np.linspace(0.1,np.sqrt(2)*len(mapa)/2,len(mapa))
    ap=[]
    ii=(N-1)/2.0
    for rad in radio:
        apertures = CircularAperture((ii,ii), rad)
        aux=aperture_photometry(mapa, apertures)
        ap.append(float(aux['aperture_sum']))
    circ=np.array(ap)

    ro=0.5*N
    func=interp1d(radio,circ)
    x=np.linspace(radio.min(),radio.max(),4000)
    y=func(x)
    dg=np.zeros_like(x)
    dg[1:-1]=(y[2:]-y[:-2])/(x[2:]-x[:-2])
    dg[1:-1]/=2*np.pi*x[1:-1]

    dg[x>ro]/=1-8*np.arccos(ro/x[x>ro])/(2*np.pi)
    dg[0]=2*dg[1]-dg[2]

    x=np.insert(x,0,0)
    dg=np.insert(dg,0,dg[0])

    x=np.insert(x,len(x),2*x[-1])
    dg=np.insert(dg,len(dg),dg[-1])

    R=radial_map_N(N,N)
    func2=interp1d(x,dg)
    new_map=func2(R)

    return new_map

def symmetric_smooth_profile(mapa,L,d):
    N=len(mapa)
    R=radial_map_N(N,N)*L/N
    Nsteps=int(d*N/L+0.5)

    Rarray=[]
    Varray=[]

    for ns in range(Nsteps):
        Redges=np.arange(-d/2+ns*d/Nsteps,R.max()-d/2+ns*d/Nsteps,d)
        Rcen=0.5*(Redges[1:]+Redges[:-1])
        Nbins=len(Rcen)
        for k in range(Nbins):
            ring=(Redges[k]<=R)&(R<Redges[k+1])
            Varray.append(np.mean(mapa[ring]))
            Rarray.append(Rcen[k])

    Rarray=np.array(Rarray)
    Varray=np.array(Varray)

    Varray=Varray[Rarray.argsort()]
    Rarray=Rarray[Rarray.argsort()]
    return Rarray,Varray

def best_factor(mapa1,mapa2):
    x2=np.std(mapa1)/np.std(mapa2)
    x1=0.9*x2
    x3=1.1*x2

    f1=np.std(mapa1-x1*mapa2)
    f2=np.std(mapa1-x2*mapa2)

    for k in range(20):
        f3=np.std(mapa1-x3*mapa2)
        df=(f3-f2)/(x3-x2)
        ddf=df-(f2-f1)/(x2-x1)
        ddf/=0.5*(x3-x1)

        step=-df/ddf

        x1=x2
        x2=x3
        x3=x3+step

        f1=f2
        f2=f3
    return x3

def correct_map(mapa1,mapa2):
    new_map=mapa2
    for k in range(5):
        factor=best_factor(mapa1,new_map)
        new_map=new_map*factor
        base=np.mean(mapa1)-np.mean(new_map)
        new_map+=base

    return new_map

def shift_image(image,sx,sy):
    N=len(image)
    new=np.zeros_like(image)
    new[0:N-sx,0:N-sy]=image[sx:,sy:]
    new[N-sx:,0:N-sy]=image[:sx,sy:]
    new[0:N-sx,N-sy:]=image[sx:,:sy]
    new[N-sx:,N-sy:]=image[:sx,:sy]

    return new

def simple_symmetric_map(mapa,func):
    N=len(mapa)
    R=2*radial_map_N(N,N)/N
    Nbins=int(0.5*N)
    Redges=np.linspace(0,R.max(),Nbins+1)
    Rcen=0.5*(Redges[1:]+Redges[:-1])
    new_map=np.zeros_like(mapa)
    for k in range(Nbins):
        ring=(Redges[k]<=R)&(R<Redges[k+1])
        yaux=mapa[ring].ravel()
        yaux=yaux[yaux.argsort()]
        new_map[ring]=func(yaux)
    return new_map

def clean_data(x):
    if len(x)==1:
        return x
    eta=2*0.6744897501
    med=np.median(x)
    p25,p75=np.percentile(x,[25,75])
    sig=4*(p75-p25)/eta
    x=x[(x<med+sig)&(x>med-sig)]
    return x

def two_lines(k,n1,n2,A,k0):
    B=n1*k0+A-n2*k0
    return np.piecewise(k,[k<k0,k>=k0],[lambda k: n1*k+A, lambda k: n2*k+B])

def two_segments(k,n1,n2,A,B,k0):
    return np.piecewise(k,[k<k0,k>=k0],[lambda k: n1*k+A, lambda k: n2*k+B])

def difference_factor(mapa1,mapa2):
    x2=np.std(mapa1)/np.std(mapa2)
    x1=0.9*x2
    x3=1.1*x2

    f1=np.std(mapa1-x1*mapa2)
    f2=np.std(mapa1-x2*mapa2)

    for k in range(20):
        f3=np.std(mapa1-x3*mapa2)
        df=(f3-f2)/(x3-x2)
        ddf=df-(f2-f1)/(x2-x1)
        ddf/=0.5*(x3-x1)

        step=-df/ddf

        x1=x2
        x2=x3
        x3=x3+step

        f1=f2
        f2=f3
    return x3

def Fourier_power_spectrum(mapa,k,Nbins):
    kedges=np.linspace(k.min(),k.max(),Nbins+1)
    dk=kedges[1]-kedges[0]
    kbins=0.5*(kedges[1:]+kedges[:-1])
    Profile=np.zeros(Nbins)
    for i in range(Nbins):
        ring=(kedges[i]<k)&(k<kedges[i+1])
        Profile[i]=np.sum(mapa[ring])*dk
    return kbins,Profile,dk

def two_functions_smooth(k,n1,n2,A,B):
    return A/(k**n1*(B+k**n2)+1.0e-10)

def make_profile(x,y,dx):

    nx=int((x.max()-x.min())/dx)
    dx=0.5*(x.max()-x.min())/nx

    x_edges=np.linspace(x.min()-dx,x.max()+dx,nx+2)
    x_center=0.5*(x_edges[1:]+x_edges[:-1])

    y_center=np.zeros_like(x_center)

    for k in range(nx+1):
        ring=(x_edges[k]<x)&(x<x_edges[k+1])
        y_center[k]=np.median(y[ring])

    return x_center , y_center
