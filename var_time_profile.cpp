/***************************************************************************
* Program: var_time_profile.c
* Generates calculations for any temporal-emission profile and spectrum by
* approximating it as a series of rectangular pulses, modulated by the appropriate
* profile.
* Usage: ashwin$ gcc -Wall -g -o var_time_profile -lm var_time_profile.c
*        ashwin$ ./var_time_profile
*                         Calculates:
*      Quantity                                             Filename
* time-resolved spectra in logscale                      time_res_spectra.dat
* Light curves in 4 different energy bands as well as    light_curves.lc
* total light curve
* Flux curves at definite energies                       f_eps_t.dat
* Peak-flux and Epeak vs. time                           flux_epk_time.dat
*
***************************************************************************/
# include <stdio.h>
# include <math.h>
# include <stdlib.h>
# include <time.h>
//#include <omp.h>
#include <iostream>

using namespace std;

#define C 2.9979e10
#define MP 1.50e-3
#define ME 8.187e-7
#define PI acos(-1.)
#define RE 2.8179e-13
#define SIGMA_T 6.6524e-25
#define HPLANCK 6.626e-27
#define BCR 4.413e13
#define Q 4.8013e-10
#define SQ(x) ((x)*(x))

#define N_eps 200 // Number of energy points with logarithmic resolution
#define N_band 4  //Number of energy bands in which light curves are extracted
#define N_shell 20 //Number of shell propagation delays
#define N_seg 100 // Number of time segments for each shell propagation delay
#define N_theta 100   // Number of angular points over which the angular integration is performed. I have used logarithmic resolution to increase the speed of integration
#define theta_min 1.e-3  // Minimum value of theta (on-axis)
#define theta_max PI/2.  // Maximum value of theta (off-axis)
#define step_seg 0.010   //The time resolution of the light curve for each value of r0(or t0)

/* These are the Cosmological Parameters for a flat universe using WMAP values*/
#define H_0 2.268e-18
#define O_M 0.27
#define O_L 0.73
#define N_lum 10000


#define det_area 1.0e+12 //Detector Area for a conversion from observed flux to photon counts

/*The following two functions, "max_of()" and "min_of()" have been written to find the minimum and maximum of any two variables*/
double max_of(double x1, double x2)
{
  if(x1 > x2)return(x1);
  else return(x2);
}

double min_of(double x1, double x2)
{
    if(x1 < x2)return(x1);
    else return(x2);
}

/*The following function calculates beta for different values of Gamma*/
double beta(double g)
{
 	if (g < 1.0001) return (sqrt(2.*(g - 1.)));
	else if (g > 1.e4) return (1.-(1./(2.*g*g)));
	else return (sqrt(1. - 1./(g*g)));
}

/*The following function sets the functional form of the energy spectrum*/
double sed_function(int spec_flag, double delta, double eps, double epk0pm, double z)
{
    double x=(1.+z)*eps/(delta*epk0pm);
    double Ep = epk0pm;
    double E = (1.+z)*eps/delta;
    double E_ref = 100./511.;
    
    if(spec_flag == 0)    //Perfect broken power-law with nuFnu indices ax and bx
    {
        double ax = 4./3.;
        double bx = -1./2.;
        if(x > 1.)return(pow(x,bx));
        else return(pow(x,ax));
    }
    
    else if(spec_flag == 1)  //Band function with photon indices a and b
    {
        double a = -0.866;
        double b = -2.66;
        double E_break = (a - b)*Ep/(2+a) ;
        if(E <  E_break)return(x*x*pow((E/E_ref),a)*exp(-E*(2.0 + a)/Ep));
        else return(x*x*pow((a - b)*Ep/(E_ref *(2.0 + a)),a - b)*exp(b - a)*pow((E/E_ref),b));
    }
    
    else if(spec_flag == 2) //Assymmetric log-parabolic function with photon indices a and b
    {
        double Eplp = 30./511.;
        double alp = 0.;
        double blp = 0.5;
        double S_p = 1.;
        return(S_p * pow(10.0, alp*log10(E/Eplp) - blp * pow(log10(E/Eplp),2.0)));
    }
    
    
    else if(spec_flag == 3) //Comptonized-Epeak function with photon indices a and b
    {
        double Epc = epk0pm;
        double alpha = -0.4553;
        return(x*x*pow((Epc*x/E_ref),alpha)*exp(-x*(2.0 + alpha)));
    }
    else return(0.);
}

/*The following function sets the temporal emission profile*/
double time_profile(int time_profile_flag, double t0, double t1)
{
    if(time_profile_flag == 0)   /* square profile */
    {
     return(1.0);
    }
    
    if(time_profile_flag == 1)  /* Power-law decay profile */
    {
        double alpha_t = 5.;
        return(pow(t0/t1,-alpha_t));
    }

    
    if(time_profile_flag == 2) return(exp(t0/t1));  /* Exponential profile */
    
    if(time_profile_flag == 3)    /* Gaussian intrinsic profile with mean and sigma */
    {
         double mean = 1.0;
         double sigma = 1.0;
         return(exp(- SQ((t0-mean)/sigma)));
    }
    else return(0.);
}

/* A brute force Luminosity distance calculator for a flat cosmology with WMAP parameter values for a given 'z' */
double lum_dist(double z)
    {
        double zp = 0., dzp = 0., dlum = 0.;
        for(int i = 0; i < N_lum; i ++)
        {
            zp = i*z/N_lum;
            dzp = z/N_lum;
            dlum += ((1.+z)*C/H_0) * (dzp/sqrt(O_M*pow(1.+zp,3.0)+O_L));
        }
        return(dlum);
    }

  double clk[100]={0}; /* Added */

/************************************************************************/
/************************************************************************/
int main()
{
      //clk[0] = 1000*omp_get_wtime();

    clock_t begin = clock();    // This function is used to calculate the total CPU time of the simulation
    
    FILE *time_res_spectra, *flux_epk_time, *f_eps_t, *theta_pars, *shell_pars;
	time_res_spectra = fopen("time_res_spectra.dat","w");      //saves a certain number of time resolved spectra (currently every 50th spectrum)
	flux_epk_time = fopen("flux_epk_time.dat","w");            //saves the peak-flux and Epeak for every time-resolved spectrum
    f_eps_t = fopen("f_eps_t.dat","w");                        //saves the flux vs. time for a specific energy (currently E = 88 keV)
    theta_pars = fopen("theta_pars.dat","w");                  //saves values of parameters that are a function of theta or mu
    shell_pars = fopen("shell_pars.dat","w");                  //saves shell parameters as a function of t0 or r0
    
    int time_profile_flag = 1;  // choose the temporal-emission profile
    int spec_flag = 0;          // choose the spectral function

	double eps_max[N_seg] = {0.}, en[N_eps] = {0.}, max[N_seg] = {0.};
    double r0a,r1a,r0b,r1b,rl,ru,term,factor;
    int i_eps = 0, i_time = 0, i_shell = 0, n_shift = 0, i_theta = 0;

    /*Define eSe[N_eps][N_shell][N_seg]: A 3-D flux array that stores the values of vFv flux for each energy, shell position and time-segment respectively */
    double ***eSe;
	if (NULL == (eSe = (double ***)malloc(N_eps*sizeof(double**)))){ puts("eSe malloc failure: line 185");exit(1);}
	for (i_eps = 0; i_eps < N_eps; i_eps ++){
		if (NULL == (eSe[i_eps] = (double **) malloc(N_shell*sizeof(double*)))){puts("eSe malloc failure: line 187");exit(1);}
		for(i_shell = 0; i_shell < N_shell; i_shell ++){	if (NULL == (eSe[i_eps][i_shell] = (double*)  malloc(N_seg*sizeof(double)))){puts("eSe malloc failure: line 188");exit(1);};/*cout << N_eps*sizeof(double**)*1+N_shell*sizeof(double*)*N_eps+N_seg*sizeof(double)*N_eps*N_shell <<endl;*/}
	}
    
    /*Define eSe_total[N_eps][N_seg]: A 2-D flux array that stores the values of vFv flux for each energy and time-segment respectively after integrating over all shell positions*/
    double **eSe_total;
    eSe_total = (double **)malloc(N_eps*sizeof(double*));
    for(i_eps = 0; i_eps < N_eps; i_eps ++) eSe_total[i_eps] =(double *) malloc(N_seg*sizeof(double));
    
   double G0 = 0., bG = 0., z = 0., etat = 0., etadelta = 0., etar = 0.;
   double theta = 0., mu = 0., dmu = 0., factor_theta = 0., d_theta = 0., theta_prev = 0.;
   double epk0 = 0., epk0pm = 0., eps = 0., epsmin = 0., epsmax = 0., epsfactor = 0.;
   double r0 = 0., deltarpm = 0., deltar = 0.;
   double delta = 0., coef = 0., uprime = 0.;
   double time = 0., dtprime = 0., tz = 0., tinitial = 0., tinitial_0 = 0., t0 = 0., t1 = 0., del_t0 = 0.;

    z = 1.0;  //redshift
    G0 = 300.; // Bulk Lorentz factor
   epk0 = 250./511.; //Epeak,0 in the observer frame
   epsmin = 0.1/511.; // Minimum energy for observations
   epsmax = 985./511.; // Maximum energy for observations
   etat = 1.0;         // Blastwave illumination parameter
   etadelta = 1.0;     // Blastwave width parameter
   etar = 1.0;         // Blastwave radius parameter
   uprime = 1.0; /* internal energy density (ergs/cm^3)*/
   epsfactor = (log10(epsmax) - log10(epsmin))/N_eps; //sets the logarithmic energy resolution
   t1 = 0.1;//0.1;    // Initial value of variability timescale. Also sets the scaling for the power-law profile
   t0 = t1;           // The variability timescale for each shell position
   del_t0 = t1;       // Size of the time increments
  
   
	double dlum = lum_dist(z)/1.0e+28; /* Luminosity distance in units of 1e28 cm */
    
   bG=beta(G0);
   r0=etar*2.*SQ(G0)*C*t1/(1.+z);  			/*  Blast wave location */
   deltarpm=etadelta*2.*G0*C*t1/(1.+z);   /*  Blast wave width */
   deltar=deltarpm/G0;                    /* Blastwave width in the observer frame */
   epk0pm=(1.+z)*epk0/(2.*G0);
    factor_theta = (log10(theta_max) - log10(theta_min))/N_theta; //sets the logarithmic angular resolution
   dtprime=etat*2.*G0*t1/(1.+z);   			/* Controls blast wave duration */
   tinitial_0=(1.+z)*((r0*(1.-bG)/(bG*C))-(deltar/(bG*C))); //tinitial for the very first radial position
   coef=C*uprime/(6.*SQ(dlum)*deltarpm);
   
/* Integration over energy */
for(i_eps = 0; i_eps < N_eps; i_eps ++)
{
    eps = epsmin*pow(10.0,epsfactor*i_eps);
    //delta_eps = eps - eps_prev;
	en[i_eps] = epsmin*pow(10.0,epsfactor*i_eps);
    t0 = t1;  //The initial value of the variability time t0

      //clk[1] = 1000*omp_get_wtime();



    for(i_shell = 0; i_shell < N_shell; i_shell ++)
    {
         //if(i_eps == 100)fprintf(shell_pars,"%d %e %e %e",i_shell, t0, r0, time_profile(time_profile_flag,t0,t1));
        coef = (C*uprime/(6.*SQ(dlum)*deltarpm)) * time_profile(time_profile_flag,t0,t1);
        r0=etar*2.*SQ(G0)*C*t0/(1.+z); // calculate r0 = r0(t0)
        tinitial=(1.+z)*((r0*(1.-bG)/(bG*C))-(deltar/(bG*C))); // calculate tinitial = tinitial(t0)
        n_shift = (int)(del_t0/step_seg);  //shift the light curve by a time corresponding to the new tinitial

        for(i_time = 0; i_time < i_shell*n_shift; i_time ++) {
		if (  (i_eps+1)*(i_shell+1)*(i_time+1)  > N_eps*N_shell*N_seg) {
			cout <<i_time << " " << (i_eps+1)*(i_shell+1)*(i_time+1) << "||"<<N_eps*N_shell*N_seg<<endl;
		}	
	}

//        for(i_time = 0; i_time < i_shell*n_shift; i_time ++)eSe[i_eps][i_shell][i_time] = 0.;
//#pragma omp parallel for
        for(i_time = i_shell*n_shift; i_time < N_seg; i_time ++)
        {
            time = tinitial+step_seg * (i_time-i_shell*n_shift);
            tz=time/(1.+z);
        
         // Integration over mu
             factor=0.0;
            theta_prev = 0.;
            for(i_theta = 0; i_theta < N_theta; i_theta ++)
            {
                  theta = theta_min * pow(10.,factor_theta*i_theta);
                  if(i_theta == 0)d_theta = 0.;
                  else d_theta = theta - theta_prev;
                  mu = cos(theta);
                  dmu = sin(theta)*d_theta;
                  delta=1./(G0*(1.-bG*mu));
                  r0a = ((r0/bG)-C*tz)/mu;
                  r1a = (bG*C*tz)/(1.-bG*mu);
                  r0b = ((r0/bG)-C*tz+C*G0*dtprime)/mu;
                  r1b = (bG*C*tz+deltar)/(1.-bG*mu);
                  rl = max_of(r0a,r1a);
                  ru = min_of(r0b,r1b);
                  if(ru <= rl) term=0.0;
                  else term=pow(delta,3.)*(pow(ru,3.)-pow(rl,3.))*sed_function(spec_flag, delta, eps, epk0pm, z);
				factor += dmu*term/SQ(1.e28);
                //if(i_eps == 100 && i_shell == 0 && i_time == 50)fprintf(theta_pars,"%d %e %e %e %e %e %e %e %e %e %e %e %e %e %e\n",i_theta, theta, d_theta, mu, dmu, delta, r0a, r1a, r0b, r1b, ru, rl, ru-rl, term, factor);
                theta_prev = theta;
			} /*close mu integration loop */
            
			eSe[i_eps][i_shell][i_time] += coef*factor;
            //if(i_eps == 100 && i_shell < 10)fprintf(f_eps_t,"%e %e\n",tinitial+i_time*step_seg,eSe[i_eps][i_shell][i_time]);
        } /* close time loop */
        
        t0 += del_t0; //Increment t0 for the next shell propagation delay
    } /* close N_shell loop */
    
    en[i_eps] = 511.0*en[i_eps];
} /* close energy loop */
      //clk[2] = 1000*omp_get_wtime();
for(i_eps = 0; i_eps < N_eps; i_eps ++)
{
//#pragma omp parallel for
  for(i_time = 0; i_time < N_seg; i_time ++)
  {
#pragma omp parallel for
     for(i_shell = 0; i_shell < N_shell; i_shell ++)
     {
         eSe_total[i_eps][i_time] += eSe[i_eps][i_shell][i_time];
     }
  }
}

/* Extract a vFv spectrum in logscale for every 50th time segment */
for(i_eps = 0; i_eps < N_eps; i_eps ++)
{
  for(i_time = 0; i_time < N_seg; i_time ++)
  {
	//if(i_time % 50 == 0)fprintf(time_res_spectra,"%e %e\n", log10(en[i_eps]), log10(eSe_total[i_eps][i_time]));
  }
}

/* Extract Peak-flux and Epeak as functions of time */
for(i_time = 0; i_time < N_seg; i_time ++)
{
  time = tinitial_0 + step_seg*i_time;
//#pragma omp parallel for
  for(i_eps = 0; i_eps < N_eps; i_eps ++)
  {
    if(eSe_total[i_eps][i_time] >= max[i_time])
    {
	  eps_max[i_time] = en[i_eps];
      max[i_time] = eSe_total[i_eps][i_time];
    }
  }
//fprintf(flux_epk_time,"%e %e %e\n", log10(time), log10(eps_max[i_time]), log10(max[i_time]));
}

    
/* Extract light curves in different energy bands as well as the total light curve */
    FILE *lc;
    lc = fopen("light_curves.lc","w");
    int i_band = 0;
    double en_start[N_band] = {20.,50.,100.,300.};
    double en_stop[N_band] = {50.,100.,300.,1000.};
    double lc_counts[N_band][N_seg] = {{0.},{0.}};
    double lc_total[N_seg] = {0.};
    
      //clk[3] = 1000*omp_get_wtime();

for(i_time = 0; i_time < N_seg; i_time ++)
{
    time = tinitial_0 + step_seg*i_time;
        for(i_eps = 0; i_eps < N_eps; i_eps ++)
        {
            for(i_band = 0; i_band < N_band; i_band ++)
            {
                if(en[i_eps] >= en_start[i_band] && en[i_eps] < en_stop[i_band]) lc_counts[i_band][i_time] += det_area*eSe_total[i_eps][i_time];
            }
            lc_total[i_time] += det_area*eSe_total[i_eps][i_time];
        }
      // fprintf(lc,"%e %e %e %e %e %e\n", time, lc_counts[0][i_time], lc_counts[1][i_time], lc_counts[2][i_time], lc_counts[3][i_time], lc_total[i_time]);
   }
      //clk[4] = 1000*omp_get_wtime();


free(**eSe);
free(*eSe);
free(eSe);
free(*eSe_total);
free(eSe_total);
fclose(time_res_spectra);
fclose(flux_epk_time);
fclose(f_eps_t);
fclose(lc);
fclose(theta_pars);
fclose(shell_pars);
clock_t end = clock();
printf("Elapsed: %f seconds\n", (double)(end - begin)/CLOCKS_PER_SEC);

      //clk[99] = 1000*omp_get_wtime();
        printf("\n Loop 2,1:\t%f [ms]\n Loop 4,3:\t%f [ms]"
                      /*  "       \n Iter 6,5:\t%f [ms]"
                        "       \n Visc 8,7:\t%f [ms]"
                        "       \n Mnext 9,8:\t%f [ms]"
                        "       \n Aval 10,9:\t%f [ms]"
                        "       \n Mnext 12,11:\t%f [ms]"
                        "       \n SDev 13,12:\t%f [ms]"
                        "       \n Loop 10,9:\t%f [ms]" */
                        "       \n TOTAL: \t%f [sec]\n",
                        clk[2]-clk[1],
                        clk[4]-clk[3],
                /*        //clk[6]-clk[5],
                        clk[8]-clk[7],
                        clk[9]-clk[8],
                        clk[10]-clk[9],
                        clk[12]-clk[11],
                        clk[13]-clk[12],*/
                   /*     clk8-clk7,
                        clk9-clk8,
                        clk10-clk9, */
                        (clk[99]-clk[0])/1000.);


return(0);
}






