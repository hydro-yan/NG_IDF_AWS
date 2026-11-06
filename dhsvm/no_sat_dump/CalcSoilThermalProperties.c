#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "data2.h"
#include "functions2.h"
#include "constants.h"

/*******************************************************************************/
/*				  CalcPhaseChange()                                            */
/*                                                                             */
/* Use energy balance approach to update the phase of water in each soil layer */
/* resulting from the updated soil temperature calculations. The phase change  */
/* may also change the soil temperature. The approach below is similar to that */
/* used in the DHSVM ground snowpack model.                                    */
/*                                                                             */
/*******************************************************************************/
void CalcPhaseChange(TEMPTABLE* S, int Dt) {

	int i;
	float Tf = 0, IceMass, IceMassStart, DeltaIce, LiqMass, LiqMassStart, DeltaLiq, He, Hkg, dz;

	for (i = 1; i < S->NumLayers + 1; i++) {

		//dz = (S->Zn[i + 1] - S->Zn[i - 1]) / 2.0;  /* layer thickness m^3/m^2 */
		dz = S->Zb[i] - S->Zb[i-1];  /* layer thickness m^3/m^2 */


		if ((*S).Ts[i] > Tf && (*S).SoilIce[i] > 0.0) {    /* melting   He and Hkg > 0   ************/
			He = -1.0 * (S->Ch[i] * dz / Dt) * (Tf - (*S).Ts[i]);  /* excess energy (W/m^2) */
			Hkg = He * Dt / LF;      /* potential kg of ice that can be melted */

			IceMassStart = (*S).SoilIce[i] * ICE_DENSITY * dz;    /* kg/m^2  */

			if (Hkg > IceMassStart) {  /* all ice melted */
				IceMass = 0.0;
				Hkg -= IceMassStart;   /* some energy left to raise soil layer above Tf */
				(*S).Ts[i] = Tf + ((Dt / (S->Ch[i] * dz)) * (Hkg * LF / Dt));
			}
			else {
				IceMass = IceMassStart - Hkg;;
			}

			LiqMass = (*S).SoilMoist[i] * WATER_DENSITY * dz;    /* kg/m^2  */
			LiqMass += IceMassStart - IceMass;

			(*S).SoilIce[i] = IceMass / (ICE_DENSITY * dz);
			(*S).SoilMoist[i] = LiqMass / (WATER_DENSITY * dz);
		}

		/* freezing  He and Hkg < 0 */
		if ((*S).Ts[i] < Tf && (*S).SoilMoist[i] > 0) {
			He = -1.0 * (S->Ch[i] * dz / Dt) * (Tf - (*S).Ts[i]);  /* neg value - energy deficit (W/m^2) */
			Hkg = He * Dt / LF;      /* potential kg of ice that can be added */


			LiqMassStart = (*S).SoilMoist[i] * WATER_DENSITY * dz;    /* kg/m^2  */
			DeltaLiq = LiqMassStart + Hkg;                        /* make some ice  */

			if (DeltaLiq < 0.0) {  /* all liquid frozen */
				LiqMass = 0.0;
				Hkg += LiqMassStart;   /* some energy left  to lower soil layer temp below Tf */
				(*S).Ts[i] = Tf + ((Dt / (S->Ch[i] * dz)) * (Hkg * LF / Dt));
			}
			else {
				LiqMass = DeltaLiq;
			}

			IceMass = (*S).SoilIce[i] * ICE_DENSITY * dz;    /* kg/m^2  */
			IceMass += LiqMassStart - LiqMass;

			(*S).SoilIce[i] = IceMass / (ICE_DENSITY * dz);
			(*S).SoilMoist[i] = LiqMass / (WATER_DENSITY * dz);
		}

	}

}

/*******************************************************************************/
/*				  TotalMassH2O()                                               */
/*                                                                             */
/* Calculate total mass of ice and liquid water in the soil column             */
/*                                                                             */
/*******************************************************************************/
void TotalMassH2O(TEMPTABLE* S, int Dt, float* TotalMassIce, float* TotalMassLiq) {

	int i;
	float dz;

	*TotalMassIce = 0.0;
	*TotalMassLiq = 0.0;
	for (i = 1; i < S->NumLayers + 1; i++) {
		dz = (S->Zn[i + 1] - S->Zn[i - 1]) / 2.0;  /* layer thickness m^3/m^2 */
		*TotalMassIce += (*S).SoilIce[i] * ICE_DENSITY * dz;    /* kg/m^2  */
		*TotalMassLiq += (*S).SoilMoist[i] * WATER_DENSITY * dz;    /* kg/m^2  */
	}
}



/*******************************************************************************/
/*				  CalcCh()                                                     */
/*                                                                             */
/*    CLM eqn 6.87                                                             */
/*                                                                             */
/*******************************************************************************/
float CalcCh(float ChMin, float SoilPorosity, float SoilIce, float SoilMoist) {
	float Ch;
	Ch = ChMin * (1.0 - SoilPorosity) + SoilIce * CH_ICE + SoilMoist * CH_WATER;
	return Ch;
}

/********************************************************************************/
/* CalcKhSol()                                                                  */
/*                                                                              */
/* Based on CLM5 eqn 6.76 and 6.78                                                      */
/********************************************************************************/
float CalKhSol(float SoilMoist, float SoilIce, float SoilPorosity, float KhDry, float KhMin,
	float SoilTemp)
{
	float RelSat, KhSol, Ke, FracLiqIce, SatCond;

	RelSat = (SoilMoist + SoilIce) / SoilPorosity;  /* CLM5 eqn 6.84  */

	/* based on CLM5 eqn. 6.76  */
	if (RelSat < 0.0000001) {
		KhSol = KhDry;
	}
	else {   /* CLM5 eqn 6.83 */
		if (SoilTemp > 0)
			Ke = log10(RelSat) + 1;
		else
			Ke = RelSat;
		FracLiqIce = SoilMoist / (SoilMoist + SoilIce);

		/* CLM5 eqn 6.78     */
		SatCond = pow(KhMin, 1.0 - SoilPorosity) * pow(KhH2O, (FracLiqIce * SoilPorosity)) * pow(KhIce, SoilPorosity * (1.0 - FracLiqIce));

		KhSol = Ke * SatCond + (1 - Ke) * KhDry;  
	}

	return KhSol;
}

