import pandas as pd
import numpy as np
import math
import os

def makeSuperCell(a,b,c,alpha,beta,gamma,xnumber,ynumber,znumber,P0):
    n=1
    P=[]
    row=P0.shape[0]
    column=P0.shape[1]
    for i in range(xnumber):
        for j in range(ynumber):
            for k in range(znumber):
                xplus=a*i+b*j*math.cos(gamma)+c*k*math.cos(beta)
                yplus=b*j*math.sin(gamma)+c*k*(math.cos(alpha)-math.cos(beta)*math.cos(gamma))/math.sin(gamma)
                zplus=c*k*math.sqrt(math.sin(beta)**2-((math.cos(alpha)-math.cos(beta)*math.cos(gamma))/math.sin(gamma))**2)
                Pnew=np.zeros((row,column))
                Pnew[:,0]=P0[:,0]+xplus
                Pnew[:,1]=P0[:,1]+yplus
                Pnew[:,2]=P0[:,2]+zplus
                n+=1
                P=np.append(P,Pnew).reshape(row*(n-1),column)
    return P

def GetPanel(p1,p2,p3):#ax+by+cz+d=0
    a = (p2[1]-p1[1])*(p3[2]-p1[2])-(p2[2]-p1[2])*(p3[1]-p1[1])
    b = (p2[2]-p1[2])*(p3[0]-p1[0])-(p2[0]-p1[0])*(p3[2]-p1[2])
    c = (p2[0]-p1[0])*(p3[1]-p1[1])-(p2[1]-p1[1])*(p3[0]-p1[0])
    d = 0-(a*p1[0]+b*p1[1]+c*p1[2])
    return a,b,c,d

def PBC(a,b,c,alpha,beta,gamma,P0):
    row=3
    column=P0.shape[1]
    Pnew=P0
    bdl=[0,0,0]
    bdr=[a,0,0]
    bul=[c*math.cos(beta),c*(math.cos(alpha)-math.cos(beta)*math.cos(gamma))/math.sin(gamma),c*math.sqrt(math.sin(beta)**2-((math.cos(alpha)-math.cos(beta)*math.cos(gamma))/math.sin(gamma))**2)]
    fdl=[b*math.cos(gamma),b*math.sin(gamma),0]
    fdr=[a+b*math.cos(gamma),b*math.sin(gamma),0]
    ful=[b*math.cos(gamma)+c*math.cos(beta),b*math.sin(gamma)+c*(math.cos(alpha)-math.cos(beta)*math.cos(gamma))/math.sin(gamma),c*math.sqrt(math.sin(beta)**2-((math.cos(alpha)-math.cos(beta)*math.cos(gamma))/math.sin(gamma))**2)]
    bur=[a+c*math.cos(beta),c*(math.cos(alpha)-math.cos(beta)*math.cos(gamma))/math.sin(gamma),c*math.sqrt(math.sin(beta)**2-((math.cos(alpha)-math.cos(beta)*math.cos(gamma))/math.sin(gamma))**2)]
    fur=[a+b*math.cos(gamma)+c*math.cos(beta),b*math.sin(gamma)+c*(math.cos(alpha)-math.cos(beta)*math.cos(gamma))/math.sin(gamma),c*math.sqrt(math.sin(beta)**2-((math.cos(alpha)-math.cos(beta)*math.cos(gamma))/math.sin(gamma))**2)]
    #front
    [af,bf,cf,df]=GetPanel(fdl,ful,fdr)
    #back
    [ab,bb,cb,db]=GetPanel(bdl,bul,bdr)
    #up
    [au,bu,cu,du]=GetPanel(ful,bul,fur)
    #down
    [ad,bd,cd,dd]=GetPanel(fdl,bdl,fdr)
    #left
    [al,bl,cl,dl]=GetPanel(bdl,fdl,ful)
    #right
    [ar,br,cr,dr]=GetPanel(fdr,fur,bdr)
    for i in range(row):
        if af*Pnew[i,0]+bf*Pnew[i,1]+cf*Pnew[i,2]+df>0:
            print('f')
            Pnew[i,0]= Pnew[i,0]-b*math.cos(gamma)
            Pnew[i,1]= Pnew[i,1]-b*math.sin(gamma)
        if ab*Pnew[i,0]+bb*Pnew[i,1]+cb*Pnew[i,2]+db<0:
            print('b')
            Pnew[i,0]= Pnew[i,0]+b*math.cos(gamma)
            Pnew[i,1]= Pnew[i,1]+b*math.sin(gamma)
        if au*Pnew[i,0]+bu*Pnew[i,1]+cu*Pnew[i,2]+du>0:
            print('u')
            Pnew[i,0] = Pnew[i,0]-c*math.cos(beta)
            Pnew[i,1] = Pnew[i,1]-c*(math.cos(alpha)-math.cos(beta)*math.cos(gamma))/math.sin(gamma)
            Pnew[i,2] = Pnew[i,2]-c*math.sqrt(math.sin(beta)**2-((math.cos(alpha)-math.cos(beta)*math.cos(gamma))/math.sin(gamma))**2)
        if ad*Pnew[i,0]+bd*Pnew[i,1]+cd*Pnew[i,2]+dd<0:
            print('d')
            Pnew[i,0] = Pnew[i,0]+c*math.cos(beta)
            Pnew[i,1] = Pnew[i,1]+c*(math.cos(alpha)-math.cos(beta)*math.cos(gamma))/math.sin(gamma)
            Pnew[i,2] = Pnew[i,2]+c*math.sqrt(math.sin(beta)**2-((math.cos(alpha)-math.cos(beta)*math.cos(gamma))/math.sin(gamma))**2)
        if al*Pnew[i,0]+bl*Pnew[i,1]+cl*Pnew[i,2]+dl<0:
            print('l')
            Pnew[i,0]=Pnew[i,0]+a
        if ar*Pnew[i,0]+br*Pnew[i,1]+cr*Pnew[i,2]+dr>0:
            Pnew[i,0]=Pnew[i,0]-a
    return Pnew

def ChangePosition(P1,P2,a,b,c,alpha,beta,gamma):
    distance=np.linalg.norm(np.array(P1) - np.array(P2))
    flag=0
    P2new=P2
    if P2[2]-P1[2]>0.5*c*math.sqrt(math.sin(beta)**2-((math.cos(alpha)-math.cos(beta)*math.cos(gamma))/math.sin(gamma))**2):
        P2new[0] = P2new[0]-c*math.cos(beta)
        P2new[1] = P2new[1]-c*(math.cos(alpha)-math.cos(beta)*math.cos(gamma))/math.sin(gamma)
        P2new[2] = P2new[2]-c*math.sqrt(math.sin(beta)**2-((math.cos(alpha)-math.cos(beta)*math.cos(gamma))/math.sin(gamma))**2)
        flag=1
    elif P2[2]-P1[2]<-0.5*c*math.sqrt(math.sin(beta)**2-((math.cos(alpha)-math.cos(beta)*math.cos(gamma))/math.sin(gamma))**2):
        P2new[0] = P2new[0]+c*math.cos(beta)
        P2new[1] = P2new[1]+c*(math.cos(alpha)-math.cos(beta)*math.cos(gamma))/math.sin(gamma)
        P2new[2] = P2new[2]+c*math.sqrt(math.sin(beta)**2-((math.cos(alpha)-math.cos(beta)*math.cos(gamma))/math.sin(gamma))**2)
        flag=1
    if P2new[1]-P1[1]>0.5*b*math.sin(gamma):
        P2new[0]=P2new[0]-b*math.cos(gamma)
        P2new[1]=P2new[1]-b*math.sin(gamma)
        flag=1
    elif P2new[1]-P1[1]<-0.5*b*math.sin(gamma):
        P2new[0]=P2new[0]+b*math.cos(gamma)
        P2new[1]=P2new[1]+b*math.sin(gamma)
        flag=1
    if P2new[0]-P1[0]>0.5*a:
        P2new[0]=P2new[0]-a
        flag=1
    elif P2new[0]-P1[0]<-0.5*a:
        P2new[0]=P2new[0]+a
        flag=1
    if flag==1:
        distance=np.linalg.norm(np.array(P1) - np.array(P2new))
    return distance

#read MOF list
MofList=pd.read_csv('mof_list.txt')

#cell parameters and number of non-h atoms
CellPara=pd.read_csv('XYZs/master.txt',sep=r'\s{1,}')

#weighting scheme
Weighting=pd.read_csv('weighting_scheme.txt',sep=r'\s{1,}')

descriptors = pd.DataFrame(np.zeros((471,133)))
descriptors.columns=['Name','e2','e2.25','e2.5','e2.75','e3','e3.25','e3.5','e3.75','e4','e4.25','e4.5','e4.75','e5','e5.25','e5.5','e5.75','e6','e6.25','e6.5','e6.75','e7','e7.25','e7.5','e7.75','e8','e8.25','e8.5','e8.75','e9','e9.25','e9.5','e9.75','e10','p2','p2.25','p2.5','p2.75','p3','p3.25','p3.5','p3.75','p4','p4.25','p4.5','p4.75','p5','p5.25','p5.5','p5.75','p6','p6.25','p6.5','p6.75','p7','p7.25','p7.5','p7.75','p8','p8.25','p8.5','p8.75','p9','p9.25','p9.5','p9.75','p10','v2','v2.25','v2.5','v2.75','v3','v3.25','v3.5','v3.75','v4','v4.25','v4.5','v4.75','v5','v5.25','v5.5','v5.75','v6','v6.25','v6.5','v6.75','v7','v7.25','v7.5','v7.75','v8','v8.25','v8.5','v8.75','v9','v9.25','v9.5','v9.75','v10','m2','m2.25','m2.5','m2.75','m3','m3.25','m3.5','m3.75','m4','m4.25','m4.5','m4.75','m5','m5.25','m5.5','m5.75','m6','m6.25','m6.5','m6.75','m7','m7.25','m7.5','m7.75','m8','m8.25','m8.5','m8.75','m9','m9.25','m9.5','m9.75','m10']

for i in range(471):
    #read the coordinates of the unit cell for each mof
    path=MofList.iloc[i,0]+str('.xyz')
    coor = pd.read_csv(os.path.join('XYZs',path), sep=r'\s{1,}', header=1)
    coor.columns = ["Element", "x", "y", "z","Na"]
    coor_cleaned=coor.drop(columns=['Na'])
    
    count=0
    
    value_e=np.zeros([33])
    value_p=np.zeros([33])
    value_v=np.zeros([33])
    value_m=np.zeros([33])
    
    #read the cell parameters
    a=CellPara.iloc[i,3]
    b=CellPara.iloc[i,4]
    c=CellPara.iloc[i,5]
    alpha=CellPara.iloc[i,6]
    alpha=alpha/180*math.pi
    beta=CellPara.iloc[i,7]
    beta=beta/180*math.pi
    gamma=CellPara.iloc[i,8]
    gamma=gamma/180*math.pi
    
    NonHNumber=CellPara.iloc[i,-1]
    
    #run PBC to make sure all are in the unit cell
    P0=np.array(coor_cleaned.iloc[:,1:4])
    P_cleaned=PBC(a,b,c,alpha,beta,gamma,P0)
    
    #settings for the APRDF
    B=100
    R=np.linspace(2,10,33)
    
    #Calculate all the distances of atom pairs
    for j in range(coor_cleaned.shape[0]-1):
        p1=np.array(coor_cleaned.iloc[j,1:4])
        p1_weighting=Weighting[Weighting['Element'].str.match(coor_cleaned.iloc[j,0])]
        for k in range(j+1,coor_cleaned.shape[0]):
            p2=np.array(coor_cleaned.iloc[k,1:4])
            count+=1
            p2_weighting=Weighting[Weighting['Element'].str.match(coor_cleaned.iloc[k,0])]
            d=ChangePosition(p1,p2,a,b,c,alpha,beta,gamma)
            #different Weighting scheme
            #different R: 2-10A with 0.25A increment
            
            #electronegativity
            w1_e=p1_weighting.iloc[0,1]
            w2_e=p2_weighting.iloc[0,1]
            for l in range(R.size):
                value_e[l]+=w1_e*w2_e*math.exp(-B*(d-R[l])**2)
            #polarizability
            w1_p=p1_weighting.iloc[0,2]
            w2_p=p2_weighting.iloc[0,2]
            for l in range(R.size):
                value_p[l]+=w1_p*w2_p*math.exp(-B*(d-R[l])**2)
            #vdWaalsVolume
            w1_v=p1_weighting.iloc[0,3]
            w2_v=p2_weighting.iloc[0,3]
            for l in range(R.size):
                value_v[l]+=w1_v*w2_v*math.exp(-B*(d-R[l])**2)
                
            #mass
            w1_m=p1_weighting.iloc[0,4]
            w2_m=p2_weighting.iloc[0,4]
            for l in range(R.size):
                value_m[l]+=w1_m*w2_m*math.exp(-B*(d-R[l])**2)
    
    descriptors.iloc[i,0]=MofList.iloc[i,0]
    descriptors.iloc[i,1:34]=value_e/NonHNumber
    descriptors.iloc[i,34:67]=value_p/NonHNumber
    descriptors.iloc[i,67:100]=value_v/NonHNumber
    descriptors.iloc[i,100:133]=value_m/NonHNumber
    print(count) 
        
descriptors.to_csv('result')