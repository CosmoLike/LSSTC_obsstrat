import sys
#sys.path.append('/home/teifler/CosmoLike/WFIRST_forecasts/')

from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from cosmolike_libs import * 

write_datav = lib.write_vector_wrapper
write_datav.argtypes = [ctypes.c_char_p,InputCosmologyParams, InputNuisanceParams]

write_datav = lib.write_vector_wrapper
write_datav.argtypes = [ctypes.c_char_p,InputCosmologyParams, InputNuisanceParams]


def get_fisher_matrix(FM_params, invcov,flag, step_width = 1.0):
    print("\n\n--------------------------------------------")
    print("Calculating Fisher Matrix")
    print("--------------------------------------------\n")
    print("Step Size = %.2f" % (step_width))
    ##First, set up containers for the data we need
    ndata = invcov.shape[0]
    npar = len(FM_params)
    derivs = np.zeros((npar,ndata))
    FM = np.zeros((npar,npar))
    diag_prior_Fisher = np.zeros(npar)
    file1 = "FM_datav2"+flag
    ##Now load necessary params
    cosmo_fid = InputCosmologyParams().fiducial()
    cosmo_sigma = InputCosmologyParams().fiducial_sigma()
    cosmo_prior= InputCosmologyParams().prior_Fisher()

    if flag=='LSST_Y1':
        nuisance_fid = InputNuisanceParams().fiducial_Y1()                             
    if flag=='LSST_Y3':
        nuisance_fid = InputNuisanceParams().fiducial_Y3()                             
    if flag=='LSST_Y6':
        nuisance_fid = InputNuisanceParams().fiducial_Y6()                             
    if flag=='LSST_Y10':    
        nuisance_fid = InputNuisanceParams().fiducial_Y10()

    nuisance_sigma = InputNuisanceParams().fiducial_sigma()
    nuisance_prior = InputNuisanceParams.prior_Fisher()
    #for x in FM_params: print(getattr(cosmo_fid, x))
    ##Find Fisher Matrix
    for n,p in enumerate(FM_params):
        ##Get values on the grid for each param
        try:
            tomo = int(p[-1])+1
        except ValueError:
            tomo = 0

        if (p in InputCosmologyParams().names()):
            cosmo_var = InputCosmologyParams().fiducial()
            diag_prior_Fisher[n] = 1./(getattr(cosmo_prior, p))**2.
            p0 = getattr(cosmo_fid,p)
            dp = getattr(cosmo_sigma,p)*step_width
            print("FM: evaluting derivative for parameter %s=%e with prior %e (%e)"%(p,p0,diag_prior_Fisher[n], 1./np.sqrt(diag_prior_Fisher[n])))
            #print(p0, dp)

            setattr(cosmo_var, p, p0-2.*dp)
            write_datav(file1,cosmo_var,nuisance_fid)
            dv_mm = np.genfromtxt(file1.decode('UTF-8'))[:,1]

            setattr(cosmo_var, p, p0-dp)
            write_datav(file1,cosmo_var,nuisance_fid)
            dv_m = np.genfromtxt(file1.decode('UTF-8'))[:,1]

            setattr(cosmo_var, p, p0+dp)
            write_datav(file1,cosmo_var,nuisance_fid)
            dv_p = np.genfromtxt(file1.decode('UTF-8'))[:,1]

            setattr(cosmo_var, p, p0+2.*dp)
            write_datav(file1,cosmo_var,nuisance_fid)
            dv_pp = np.genfromtxt(file1.decode('UTF-8'))[:,1]   
        elif (tomo):
            pshort = p[:-2]
            i = int(p[-1])
            np_var = nuisance_fid
            p0 = getattr(nuisance_fid,pshort)[i]
            diag_prior_Fisher[n] = 1./(getattr(nuisance_prior, pshort)[i])**2.

            print("FM: evaluting derivative for parameter %s[%d]=%e with prior %e (%e)"%(p,i,p0,diag_prior_Fisher[n], 1./np.sqrt(diag_prior_Fisher[n])))
            dp = getattr(nuisance_sigma,pshort)[i]*step_width*10.

            getattr(np_var, pshort)[i]= p0-2.*dp
            write_datav(file1,cosmo_fid,np_var)
            dv_mm = np.genfromtxt(file1)[:,1]

            getattr(np_var, pshort)[i]= p0-dp
            write_datav(file1,cosmo_fid,np_var)
            dv_m = np.genfromtxt(file1)[:,1]

            getattr(np_var, pshort)[i]= p0+dp
            write_datav(file1,cosmo_fid,np_var)
            dv_p = np.genfromtxt(file1)[:,1]

            getattr(np_var, pshort)[i]= p0+2.*dp
            write_datav(file1,cosmo_fid,np_var)
            dv_pp = np.genfromtxt(file1)[:,1]
        else:
            np_var = nuisance_fid
            p0 = getattr(nuisance_fid,p)
            diag_prior_Fisher[n] = 1./(getattr(nuisance_prior, p))**2.
            print("FM: evaluting derivative for parameter %s=%e with prior %e (%e)"%(p,p0,diag_prior_Fisher[n], 1./np.sqrt(diag_prior_Fisher[n])))
            dp = getattr(nuisance_sigma,p)*step_width*10.

            setattr(np_var, p, p0-2.*dp)
            write_datav(file1,cosmo_fid,np_var)
            dv_mm = np.genfromtxt(file1)[:,1]

            setattr(np_var, p, p0-dp)
            write_datav(file1,cosmo_fid,np_var)
            dv_m = np.genfromtxt(file1)[:,1]

            setattr(np_var, p, p0+dp)
            write_datav(file1,cosmo_fid,np_var)
            dv_p = np.genfromtxt(file1)[:,1]

            setattr(np_var, p, p0+2.*dp)
            write_datav(file1,cosmo_fid,np_var)
            dv_pp = np.genfromtxt(file1)[:,1]


        #five point method for the first derivative
        derivs[n,:] = (-dv_pp +8.*dv_p -8.*dv_m+dv_mm)/(12.*dp)
        if (np.sum(np.abs(derivs[n,:]))==0):
            print("derivate is zero\nEXIT!\n")
            exit(1)
    #Matrix multiply the 
    for i in range(0,npar):
        for j in range(0,npar):
            FM[i,j] = np.dot(np.dot(invcov,derivs[i,:]),derivs[j,:])
    FM += np.diagflat(diag_prior_Fisher)
    return FM,derivs

def FM_analyze(FM,FM_params):
    n_cosmo = min(7,len(FM_params))
    #print FM
    FMinv= LA.inv(FM)
    covDE=FMinv[np.ix_([3,4],[3,4])]
    # for n in range(0,len(FM_params)):
    #     print("sigma(%s) = %e (no sys: %e, umarg: %e)" % (FM_params[n],np.sqrt(FMinv[n,n]),np.sqrt(FMinv_cosmo[n,n]),1./np.sqrt(FM[n,n])))
    #     print("sigma(%s) = %e (umarg: %e)" % (FM_params[n],np.sqrt(FMinv[n,n]),1./np.sqrt(FM[n,n])))
    ind_w0 = FM_params.index("w0") if "w0" in FM_params else -1 
    ind_wa = FM_params.index("wa") if "wa" in FM_params else -1 
    if ((ind_w0 > -1) & (ind_wa > -1)):
        FOM = 1./np.sqrt(FMinv[ind_w0,ind_w0]*FMinv[ind_wa,ind_wa] -FMinv[ind_w0,ind_wa]*FMinv[ind_wa,ind_w0])
        FOM2 = np.power(LA.det(LA.inv(covDE)),0.5)
        print("FoM = %e %e %d %d" %(FOM,FOM2,ind_wa,ind_w0))
    return FOM
 
def read_invcov(cov_filename):
    covfile = np.loadtxt(cov_filename.decode('UTF-8'))
    ndata = int(np.max(covfile[:,0]))+1
    invcov = np.zeros((ndata,ndata))
    for i in range(0,covfile.shape[0]):
        invcov[int(covfile[i,0]),int(covfile[i,1])]=covfile[i,2]
    return invcov

def init(file_source_z,file_lens_z,cov_file,Ntomo_lens,survey):
    initcosmo()
    initfisherprecision()
    initbins(20,20.0,15000.0,3000.0,21.0,5,int(Ntomo_lens))
    initsurvey(survey)
    initgalaxies(file_source_z,file_lens_z,"gaussian","gaussian","SRD")
    initclusters()
    initia("NLA_HF","GAMA")
    initpriors("none","none","none","none")
    initprobes("3x2pt_clusterN_clusterWL")
    invcov = read_invcov(cov_file)
    return invcov   

#######################
file_source_z = "./zdistris/"+sys.argv[6]
file_lens_z = "./zdistris/"+sys.argv[7]
datav_fid = "./datav/3x2pt_clusterN_clusterWL_"+sys.argv[1]+"_area"+sys.argv[2]+"_ng"+sys.argv[3]+"_nl"+sys.argv[4]
cov_file = "./cov/"+sys.argv[1]+"_area"+sys.argv[2]+"_ng"+sys.argv[3]+"_nl"+sys.argv[4]+"_3x2pt_clusterN_clusterWL_inv"

# print file_source_z 
# print file_lens_z 
# print datav_fid 
# print cov_file 

invcov = init(file_source_z,file_lens_z,cov_file,sys.argv[5],sys.argv[1])

if(sys.argv[1]=='LSST_Y1'):
    MORPRIOR=np.zeros((3,3))
    MORPRIOR=[[1302.77777778,1319.88833456,456.13561437],[1319.88833456,1725.21929159,590.65347498],[456.13561437,590.65347498,228.8004921]] 

if(sys.argv[1]=='LSST_Y3'):
    MORPRIOR=np.zeros((3,3))
    MORPRIOR=[[1302.77777778,1319.88833456,456.13561437],[1319.88833456,1725.21929159,590.65347498],[456.13561437,590.65347498,228.8004921]] 

if(sys.argv[1]=='LSST_Y6'):
    MORPRIOR=np.zeros((3,3))
    MORPRIOR=[[2622.22222222,2839.59646683,893.79987381],[2839.59646683,3454.43840371,1097.45127586],[893.79987381,1097.45127586,424.44951605]]  

if(sys.argv[1]=='LSST_Y10'):
    MORPRIOR=np.zeros((3,3))
    MORPRIOR=[[2622.22222222,2839.59646683,893.79987381],[2839.59646683,3454.43840371,1097.45127586],[893.79987381,1097.45127586,424.44951605]] 
   
flag=sys.argv[1]

FM_params= sample_cosmology_2pt_cluster_SRD(int(sys.argv[5]),MG=False)

f = open('Fishers/FoM.txt'+sys.argv[1],'a')
h = open('Fishers/Fisher_all.txt'+sys.argv[1],'a')
g = open('Fishers/Fisher_cosmo.txt'+sys.argv[1],'a')

print FM_params
for i in range(0,1):
    step_width=0.5+i*0.25
    #step_width=1.0
    FM,derivs = get_fisher_matrix(FM_params,invcov,flag,step_width = step_width)
    Priormat=np.zeros((len(FM_params),len(FM_params))) 
    PlanckBossJlaH0=np.zeros((7,7))

    if(sys.argv[1]=='LSST_Y1'):
        Priormat[19:22,19:22]=MORPRIOR
    if(sys.argv[1]=='LSST_Y3'):
        Priormat[21:24,21:24]=MORPRIOR
    if(sys.argv[1]=='LSST_Y6'):
        Priormat[23:26,23:26]=MORPRIOR
    if(sys.argv[1]=='LSST_Y10'):
        Priormat[24:27,24:27]=MORPRIOR
   
      
    FM=FM+Priormat
    FoM=FM_analyze(FM,FM_params)
    f.write('\n' + 'mode= %s %s %s %s'%(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]))
    f.write('\n' + 'FoM(excl. S3(no w0wa) prior)=%e'%(FoM))
    
    h.write('\n' + 'mode= %s %s %s %s'%(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]))
    for i in range(0,FM.shape[0]):
        h.write('[')
        for j in range(0,FM.shape[0]-1):
            h.write('%e, '%(FM[i,j]))
        j = FM.shape[0] -1
        h.write('%e]\n'%(FM[i,j]))
    h.write('\n')
    h.write('\n')
    iF = LA.inv(FM)
    cFM = LA.inv(iF[0:7,0:7])
    
    g.write('\n' + 'mode= %s %s %s %s'%(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]))
    for i in range(0,7):
        g.write('[')
        for j in range(0,6):
            g.write('%e, '%(cFM[i,j]))
        j = 6
        g.write('%e],\n'%(cFM[i,j]))
    g.write('\n')
    g.write('\n')

    PlanckBossJlaH0=np.array([[  8.52889541e+03,   9.34266060e-03,   2.21513407e+00,
              1.85008170e+01,   6.12185662e+00,   1.17720010e+02,
              5.30670581e+01],
           [  9.34266060e-03,   3.34475078e+01,   3.79135616e-01,
             -8.02192934e-02,   5.88736315e-02,  -8.69202865e+01,
             -7.08029577e+00],
           [  2.21513407e+00,   3.79135616e-01,   2.48739187e+02,
             -7.46022981e+00,  -2.69922217e+00,  -1.19693897e+02,
              4.78085262e+00],
           [  1.85008170e+01,  -8.02192934e-02,  -7.46022981e+00,
              8.42937127e+01,   1.68856256e+01,  -4.89063661e+01,
             -8.77194357e+00],
           [  6.12185662e+00,   5.88736315e-02,  -2.69922217e+00,
              1.68856256e+01,   9.55489400e+00,   5.27704214e+00,
             -1.38597499e+00],
           [  1.17720010e+02,  -8.69202865e+01,  -1.19693897e+02,
             -4.89063661e+01,   5.27704214e+00,   7.17777988e+04,
              3.08491105e+03],
           [  5.30670581e+01,  -7.08029577e+00,   4.78085262e+00,
             -8.77194357e+00,  -1.38597499e+00,   3.08491105e+03,
              5.32751808e+02]])

    Priormat[0:7,0:7]=PlanckBossJlaH0

    FM=FM+Priormat
    FoM2=FM_analyze(FM,FM_params)
    f.write('\n' + 'FoM(incl. S3 prior)=%e\n'%(FoM2))

f.close()
g.close()
h.close()
