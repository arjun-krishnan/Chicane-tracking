# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 14:29:01 2020

@author: SheRlocK
"""

import pandas as pd
import matplotlib.pyplot as plt
import scipy.constants as const
import numpy as np
import time
from scipy import interpolate

global v_vec,gg
gg=[]

class Particle:
    E0_e= (const.m_e*const.c**2)/const.e/(10**6)
    def __init__(self,E,x=0,y=0,z=0):
        self.E= E
        self.pos= [[x],[y],[z]]
        self.g= E/self.E0_e
        self.vz= const.c*np.sqrt(1-1/self.g**2)
        self.p=np.sqrt((const.e*self.E*1e6)**2-const.m_e**2*const.c**4)/const.c
        self.I1= 0
        
    def track(self,Bf):
        dt=1e-13
        dz=self.vz*dt
        self.dz=dz
        rn=np.array([self.pos[0][0],self.pos[1][0],self.pos[2][0]])
        k=0
        self.By=[] 
        p_ele=np.array([0,0,self.p])
        print("Tracking particle (E="+str(self.E)+" MeV)")
        while (rn[2]<Bf.z[-1]): 
            #B=np.array([Bf.Bxfunc((rn[0],rn[2])),Bf.Byfunc((rn[0],rn[2])),Bf.Bzfunc((rn[0],rn[2]))])
            B=np.array([0,Bf.Byfunc((0,rn[2])),0])
            self.By.append(B[1])
            p_vec=np.sqrt(np.sum(p_ele**2,axis=0))
            gamma_vec=np.sqrt((p_vec/const.m_e/const.c)**2+1)
            gg.append(np.copy(gamma_vec))
               
            dp_vec=-np.cross(p_ele,B)/const.m_e/gamma_vec*const.e*dt
            p_new=p_ele+dp_vec
        #    p_vec_new=np.sqrt(np.sum(p_new**2,axis=0))
            rn=rn+p_new/const.m_e/gamma_vec*dt
            for i in range(3):
                self.pos[i].append(rn[i])
            p_ele=np.copy(p_new)
            k+=1
        print("Tracking finished! \nNo.of iterations:  "+str(k)+"\n Runtime :  ",time.time()-t_i)
        self.nsteps=k
        self.I1,self.I2=[0],[0]
        for bb in self.By:
            self.I1.append(self.I1[-1]+(bb*self.dz))
        for bb in self.I1[1:]:
            self.I2.append(self.I2[-1]+(bb*self.dz))

        dxdz=np.gradient(self.pos[0],dz)
        self.pathlen=sum(np.sqrt(1+dxdz**2)*dz)
        
    def plot_track(self):    
        fig,ax=plt.subplots()
        ax.plot(self.pos[2],np.array(self.pos[0])*1000)
        ax.set_xlabel("Z(m)")
        ax.set_ylabel("X(mm)",color='blue')
        ax2=ax.twinx()
        ax2.plot(self.pos[2][:-1],self.By,'red')
        ax2.set_ylabel("By (T)",color='red')
        
        
class B_Field:
    head=["x","y","z","Bx","By","Bz"]
    def __init__(self,filename,x_mid=0,y_mid=0):
        self.fl= filename
        self.x_mid=x_mid
        self.y_mid=y_mid
        
        try:
            print(" Reading from file "+self.fl+".h5 ...")
            self.data=pd.read_hdf("../data/Field_data/"+filename+".h5")
        except:
            print(" File not found! \n Trying "+self.fl+".txt ...")
            self.data=pd.read_csv("../data/Field_data/"+filename+".txt",delim_whitespace=True,skiprows=2,names=self.head)
            self.data=self.data[self.data['y']==self.y_mid]
            #self.data.to_hdf(filename+".h5",key="B")       
            
        print("Reading Complete!")

        if self.data.empty:
            print("Empty DataFrame! Check y_mid value.")
            return
        
        self.B=([],[],[])
        xlist= list(self.data.x.unique())
        xlist.sort()
        for i in xlist:
            data2=self.data[(self.data['x']==i)]
            data2=data2.sort_values('z',ascending=True)
            self.z=np.array(list(data2['z']))
            self.B[0].append(list(data2['Bx']))
            self.B[1].append(list(data2['By']))
            self.B[2].append(list(data2['Bz']))
            
        zi=self.z[0]
        self.z=(self.z-zi)/1000    
        self.dz=(self.z[1]-self.z[0])
        self.dx=(xlist[1]-xlist[0])/1000
        self.ix= xlist.index(self.x_mid)
        self.x=(np.array(xlist)-self.x_mid)/1000
        
        self.Bxfunc=interpolate.RegularGridInterpolator((self.x,self.z),self.B[0])
        self.Byfunc=interpolate.RegularGridInterpolator((self.x,self.z),self.B[1])
        self.Bzfunc=interpolate.RegularGridInterpolator((self.x,self.z),self.B[2])
        
        self.Int_B=[]
        for i in range(3):
            self.Int_B.append(sum(np.array(self.B[i][self.ix])*self.dz))

    def plot_B(self):
        col=['blue','red','yellow']
        for i in range(3):
            plt.plot(self.z,self.B[i][self.x_pos],col[i])
        plt.show()
        
class Chicane:
    def __init__(self,p1,B):
        self.p1= p1
        self.B= B
        p1.track(B)
        self.x_max= max(p1.pos[0])
        self.x_min= min(p1.pos[0])
        self.IB= B.Int_B
        self.I1= p1.I1
        self.I2= p1.I2
        
    def R56_calc(self,dE=0.01):
        E2=self.p1.E+(self.p1.E*dE)
        p2=Particle(E2)
        p2.track(self.B)
        if(p1.nsteps>p2.nsteps):
            #self.R56= (self.p1.pathlen-p2.pathlen)-p2.dz/(dE)
            self.R56= (p2.pos[2][-1]-self.p1.pos[2][-2])/(dE)
        else:
            #self.R56= (self.p1.pathlen-p2.pathlen)/(dE)
            self.R56= (p2.pos[2][-1]-self.p1.pos[2][-1])/(dE)
        return(p2)
    
    def R51_calc(self,dx=1e-4):
        p3=Particle(E=1500,x=dx)
        p3.track(self.B)
        self.dz=p3.pos[2][-1]-self.p1.pos[2][-1]
        self.R51=self.dz/dx
        return(p3)
        
    def show_results(self):
        result="Length of the chicane  =  "+str(int(1000*(p1.pos[2][-1]-1.082)))+" mm \n R56  =  "+str(self.R56/1e-6)+"\t"+\
             "mu m \n R51  = "+str(self.R51)+"\t dz  = "+str(self.dz*1e9)+"  nm \n X_max  =   "+str(self.x_max)+\
            "\n X_min  =   "+str(self.x_min)+" m \n Field Integral : "+str(self.IB)+"\n I1 :  "+str(self.I1[-1])+"\n I2 :  "+str(self.I2[-1])+\
              "\n Additional path length : "+str(p1.pathlen-p1.pos[2][-1])
        print(result)
        f= open("../results/"+self.B.fl+"_result.txt","w+")
        f.write(result)
        f.close
        
        
        
t_i=time.time()    
p1=Particle(E=1492)
#Mag1=B_Field("Chicane1_476_72x2_chamfered")
Mag1=B_Field("Chicane2_403_54x2_chamfered")
chicane1=Chicane(p1,Mag1)
p2=chicane1.R56_calc(dE=7e-4)
p3=chicane1.R51_calc(dx=4.5e-4)
chicane1.show_results()

p1.plot_track()

#v_mag=np.sum(np.array(v_vec)**2,axis=1)**0.5