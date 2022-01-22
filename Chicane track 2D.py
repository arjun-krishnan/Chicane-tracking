
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
import os
from scipy import interpolate

global v_vec,gg

class B_Field:
    head=["x","y","z","Bx","By","Bz"]
    def __init__(self,filename,x_mid=0,y_mid=0):
     #   self.fl= filename
        self.x_mid=x_mid
        self.y_mid=y_mid
        
        try:
            print(" Reading from file "+filename+".txt ...")
            self.data=pd.read_csv("../Field_data/"+filename+".txt",delim_whitespace=True,skiprows=2,names=self.head)
            self.data=self.data[self.data['y']==self.y_mid]
            self.data.to_hdf("../Field_data/"+filename+".h5",key="B")
            os.remove("../Field_data/"+filename+".txt")
        except:
            print(" File doesn't exist! \n Trying "+filename+".h5 ...")
            self.data=pd.read_hdf("../Field_data/"+filename+".h5")
            
        print(" Reading Complete!")

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
        self.B=np.array(self.B)
        
        zi=self.z[0]
        self.z=(self.z-zi)/1000    
        self.dz=(self.z[1]-self.z[0])
        self.dx=(xlist[1]-xlist[0])/1000
        self.ix= (np.abs(np.array(xlist)-self.x_mid)).argmin()
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
    E0_e= (const.m_e*const.c**2)/const.e/(10**6)
    def __init__(self,Bf,E=1492):
        self.E= E
        self.B_field= Bf
        self.g= E/self.E0_e
        self.vz= const.c*np.sqrt(1-1/self.g**2)
        self.p=np.sqrt((const.e*self.E*1e6)**2-const.m_e**2*const.c**4)/const.c
        
    def calc_R56_R51(self,dE=7e-4,dx=5e-4):
        Bf=self.B_field
        dt=1e-14
        dz=self.vz*dt
        self.dz=dz
        
        k=0
        self.By=[] 
        E2=self.E+(self.E*dE)
        p22=np.sqrt((const.e*E2*1e6)**2-const.m_e**2*const.c**4)/const.c
        p_ele=np.array([[0,0,self.p],[0,0,p22],[0,0,self.p]])
        rn=np.array([[0,0,0],[0,0,0],[dx,0,0]]).T
        r_list=[rn]
        
        t_i=time.time()
        
        print("\n Tracking particle (E="+str(self.E)+" MeV)")
        
        while (np.mean(rn[2])<Bf.z[-1]): 
            B=np.array([Bf.Bxfunc((rn[0],rn[2])),Bf.Byfunc((rn[0],rn[2])),Bf.Bzfunc((rn[0],rn[2]))]).T
            self.By.append(B[0][1])
            p_vec=np.array([np.sqrt(np.sum(p_ele**2,axis=1))]).T
            gamma_vec=np.sqrt((p_vec/const.m_e/const.c)**2+1)

               
            dp_vec=-np.cross(p_ele,B)/const.m_e/gamma_vec*const.e*dt
            p_new=p_ele+dp_vec
            rn=rn+(p_new/const.m_e/gamma_vec*dt).T
            r_list.append(rn)
            p_ele=np.copy(p_new)
            k+=1
            
        print(" Tracking finished! \n No.of iterations:  "+str(k)+"\n Runtime :  ",time.time()-t_i)
        self.r_array= np.asarray(r_list).T
        
        z0,z1,z2= self.r_array[0][2],self.r_array[1][2],self.r_array[2][2]
        x0= self.r_array[0][0]
        
        self.R56= -(z0[-1]-z1[-1])/7e-4
        self.R51= (z0[-1]-z2[-1])/5e-4
        
        nsteps=k
        
        self.I1,self.I2=[0],[0]
        for bb in self.By:
            self.I1.append(self.I1[-1]+(bb*self.dz))
        for bb in self.I1[1:]:
            self.I2.append(self.I2[-1]+(bb*self.dz))
        
        result="\n\n Results: \n R56  =  "+str(self.R56/1e-6)+"\t"+"mu m \n R51  = "+str(self.R51)+"\t dz  = "+str((z0[-1]-z2[-1])*1e9)+"  nm "+\
            "\n X_max  =   "+str(max(x0))+"\n X_min  =   "+str(min(x0))+" m \n Field Integral : "+str(Bf.Int_B)+\
            "\n I1 :  "+str(self.I1[-1])+"\n I2 :  "+str(self.I2[-1])+"\n Additional path length : "+str((nsteps*self.dz-z0[-1])*1e6)+" mu m"
        print(result)
        
        
    def plot_track(self):    
        fig,ax=plt.subplots()
        ax.plot(self.r_array[0][2],self.r_array[0][0]*1000)
        ax.set_xlabel("Z(m)")
        ax.set_ylabel("X(mm)",color='blue')
        ax2=ax.twinx()
        ax2.plot(self.r_array[0][2][:-1],self.By,'red')
        ax2.set_ylabel("By (T)",color='red')
        


         
Mag1=B_Field("chicane2_400A_415A_12x3_250mm_250mm_full_test")
chicane1= Chicane(Mag1,E=1492)
chicane1.calc_R56_R51(dE=7e-4,dx=5e-4)
chicane1.plot_track()







'''
### To plot the good field region ###

Mag=B_Field("chicane2_400_200mm_100mm")
B=np.abs(Mag.B[1])
N=int(len(B)/2)
B_r=1-(B/B[N])
plt.figure()
plt.contourf(Mag.z*1000,Mag.x*1000,B,levels=100)
plt.colorbar(label="$B_y$ (T)")
plt.contourf(Mag.z*1000,Mag.x*1000,abs(B_r),levels=[0,0.0005],color='g',alpha=0.5)
plt.xlabel("Z (mm)")
plt.ylabel("X (mm)")

'''
