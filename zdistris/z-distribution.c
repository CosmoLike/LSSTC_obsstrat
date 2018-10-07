#include <math.h>
#include <stdlib.h>
#if !defined(__APPLE__)
#include <malloc.h>
#endif
#include <stdio.h>
#include <assert.h>

#define NR_END 1
#define FREE_ARG char*

double *sm2_vector(long nl, long nh);
void sm2_error(char *s);
void sm2_polint(double xa[], double ya[], int n, double x, double *y, double *dy);
void sm2_free_vector(double *v, long nl, long nh);
double sm2_trapzd(double (*func)(double), double a, double b, int n, double *s);

double pf_LSST(double z);
double int_for_zdistr(double z);
void obstrat_zcalc();

double *sm2_vector(long nl, long nh)
/* allocate a double vector with subscript range v[nl..nh] */
{
	double *v;

	v=(double *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(double)));
	if (!v) sm2_error("allocation failure in double vector()");
	return v-nl+NR_END;
}

void sm2_error(char *s)
{
	printf("error:%s\n ",s);
	exit(1);
}


void sm2_polint(double xa[], double ya[], int n, double x, double *y, double *dy)
{
	int i,m,ns=1;
	double den,dif,dift,ho,hp,w;
	double *c,*d;

	dif=fabs(x-xa[1]);
	c=sm2_vector(1,n);
	d=sm2_vector(1,n);
	for (i=1;i<=n;i++) {
		if ( (dift=fabs(x-xa[i])) < dif) {
			ns=i;
			dif=dift;
		}
		c[i]=ya[i];
		d[i]=ya[i];
	}
	*y=ya[ns--];
	for (m=1;m<n;m++) {
		for (i=1;i<=n-m;i++) {
			ho=xa[i]-x;
			hp=xa[i+m]-x;
			w=c[i+1]-d[i];
			if ( (den=ho-hp) == 0.0)
				sm2_error("Error in routine sm2_polint");
			den=w/den;
			d[i]=hp*den;
			c[i]=ho*den;
		}
		*y += (*dy=(2*ns < (n-m) ? c[ns+1] : d[ns--]));
	}
	sm2_free_vector(d,1,n);
	sm2_free_vector(c,1,n);
}

void sm2_free_vector(double *v, long nl, long nh)
		/* free a double vector allocated with vector() */
{
	free((FREE_ARG) (v+nl-NR_END));
}

#define FUNC(x) ((*func)(x))

double sm2_trapzd(double (*func)(double), double a, double b, int n, double *s)
{
	double x,tnm,sum,del;
	int it,j;

	if (n == 1) {
		return (*s=0.5*(b-a)*(FUNC(a)+FUNC(b)));
	} else {
		for (it=1,j=1;j<n-1;j++) it <<= 1;
		tnm=it;
		del=(b-a)/tnm;
		x=a+0.5*del;
		for (sum=0.0,j=1;j<=it;j++,x+=del) {
			sum += FUNC(x);
		}
		*s=0.5*(*s+(b-a)*sum/tnm);
		return *s;
	}
}
#undef FUNC


/* ============================================================ *
 * qromb.c              *
 * Romberg Integration. Uses trapzd. NR p. 140      *
 * ============================================================ */

#define EPS 1.0e-6
#define JMAX 35
#define JMAXP (JMAX+1)
#define K 5

double sm2_qromb(double (*func)(double), double a, double b)
{
  double ss,dss;
  double s[JMAXP],h[JMAXP+1], strap;
  int j;

  h[1]=1.0;
  for (j=1;j<=JMAX;j++) {
    s[j]=sm2_trapzd(func,a,b,j,&strap);
    if (j >= K) {
      sm2_polint(&h[j-K],&s[j-K],K,0.0,&ss,&dss);
      if (fabs(dss) <= EPS*fabs(ss)) return ss;
    }
    h[j+1]=0.25*h[j];
  }
  sm2_error("Too many steps in routine sm2_qromb");
  return 0.0;
}
#undef EPS
#undef JMAX
#undef JMAXP
#undef K

#define EPS 1.0e-7
#define JMAX 35
#define JMAXP (JMAX+1)
#define K 5



typedef struct {
     double z0;		            
     double alpha; 
     double zdistrpar_zmin;   
     double zdistrpar_zmax;
     char REDSHIFT_FILE[200];
}redshiftpara;


redshiftpara redshift = {
  1.0,
  0.0,
  0.0,
  3.0,
  ""
};


double int_for_zdistr_LSST(double z)
{
  double zz=z/redshift.z0;
  return pow(z,2.0)*exp(-pow(zz,redshift.alpha));
}



double pf_LSST(double z)
{
  
  double x, f, norm;
  //First, compute the normalization
  norm = 1.0/(sm2_qromb(int_for_zdistr_LSST,redshift.zdistrpar_zmin,redshift.zdistrpar_zmax));

  x = z/redshift.z0;
  f = pow(x,redshift.alpha);
  f=exp(-f);
  return norm*pow(z,2.0)*f;
}



void obstrat_zcalc()
{
  int i,j;
  double z,dz;
  int Nstep=300;
  double res;
  FILE *F;
  char filename[200];
  
  double WLz0[12]={0.194, 0.190, 0.186, 0.186, 0.183, 0.179,0.181, 0.179, 0.176, 0.176, 0.174, 0.171};
  double WLalpha[12]={0.883, 0.862, 0.841,0.841, 0.821, 0.800,0.814, 0.800, 0.786,0.786, 0.772, 0.759};
  
  double LSSz0[12]={0.259, 0.261, 0.264,0.264, 0.268, 0.274,0.270, 0.274, 0.278,0.278, 0.283, 0.288};
  double LSSalpha[12]={0.952, 0.937, 0.925,0.925, 0.915, 0.907,0.912, 0.907, 0.903,0.903, 0.900, 0.898};

  
  for(j=0;j<12;j++){  
    sprintf(filename,"WL_zdistri_model%d_z0=%le_alpha=%le",j,WLz0[j],WLalpha[j]);
    F=fopen(filename,"w");
    
    redshift.z0      = WLz0[j];
    redshift.alpha   = WLalpha[j];
    redshift.zdistrpar_zmin = 0.000001;
    redshift.zdistrpar_zmax = 3.5;
    dz=(redshift.zdistrpar_zmax-redshift.zdistrpar_zmin)/((Nstep)*1.0);
    
    for(i=0;i<Nstep;i++){
      z=0.0+(i+0.5)*dz;
      res=pf_LSST(z);
      fprintf(F,"%le %le %le %le\n",0.0+i*dz,z,0.0+(i+1)*dz,res);   
    }
    fclose(F);
  }

  for(j=0;j<12;j++){  
    sprintf(filename,"LSS_zdistri_model%d_z0=%le_alpha=%le",j,LSSz0[j],LSSalpha[j]);
    F=fopen(filename,"w");
    
    redshift.z0        = LSSz0[j];
    redshift.alpha   =   LSSalpha[j];
    redshift.zdistrpar_zmin = 0.000001;
    redshift.zdistrpar_zmax = 3.5;
    dz=(redshift.zdistrpar_zmax-redshift.zdistrpar_zmin)/((Nstep)*1.0);
    
    for(i=0;i<Nstep;i++){
      z=0.0+(i+0.5)*dz;
      res=pf_LSST(z);
      fprintf(F,"%le %le %le %le\n",0.0+i*dz,z,0.0+(i+1)*dz,res);   
    }
    fclose(F);
  }
}



int main(void)
{
  obstrat_zcalc();
  return 0;
}
