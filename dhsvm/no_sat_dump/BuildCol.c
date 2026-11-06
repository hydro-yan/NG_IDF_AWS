#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "data.h"
#include "functions.h"
#include "constants.h"

/***********************************************************************************/
/*				  BuildCol()                                                       */
/*                                                                                 */
/* Allocate the required memory and soil temperature layer/node depth information  */
/* The same soil layer temperature depths are used in all cells. The rooting zone  */
/* depths and soil properties are updated from each cell in MapPixToCol() below    */
/*                                                                                 */
/***********************************************************************************/
void BuildCol(TEMPTABLE *Stemp, float Zb[]) {
	
  
	int NumLayers = MAX_LAYERS - 2;  /* number of soil/bedrock layers  */
	
	float Alpha = 0.6;

	(*Stemp).NumLayers = NumLayers;
	(*Stemp).Alpha = Alpha;
	(*Stemp).Emissivity = 0.96;

	(*Stemp).Zn = (float*)calloc(MAX_LAYERS, sizeof(float));
	(*Stemp).Zb = (float*)calloc(MAX_LAYERS, sizeof(float));
	(*Stemp).Dz = (float*)calloc(MAX_LAYERS, sizeof(float));

	(*Stemp).SoilPorosity = (float*)calloc(MAX_LAYERS, sizeof(float));
	(*Stemp).SoilMoist = (float*)calloc(MAX_LAYERS, sizeof(float));
	(*Stemp).SoilIce = (float*)calloc(MAX_LAYERS, sizeof(float));

	(*Stemp).KhSol = (float*)calloc(MAX_LAYERS, sizeof(float));
	(*Stemp).KhDry = (float*)calloc(MAX_LAYERS, sizeof(float));
	(*Stemp).KhMin = (float*)calloc(MAX_LAYERS, sizeof(float));
	(*Stemp).Ch = (float*)calloc(MAX_LAYERS, sizeof(float));
	(*Stemp).ChMin = (float*)calloc(MAX_LAYERS, sizeof(float));
	(*Stemp).Ts = (float*)calloc(MAX_LAYERS, sizeof(float));

	 

	/* assign soil layer depth info */
	int i;
	(*Stemp).Zb[0] = 0.0;
	(*Stemp).Zn[0] = 0.0;
	(*Stemp).Dz[0] = 0.0;

	for (i = 1; i < MAX_LAYERS; i++) {
		(*Stemp).Zb[i] = Zb[i];
		(*Stemp).Zn[i] = (Zb[i-1] + Zb[i]) / 2.0;
		(*Stemp).Dz[i] = Zb[i] - Zb[i-1];

		//printf("%d %f %f %f\n", i, (*Stemp).Zn[i], (*Stemp).Dz[i], (*Stemp).Zb[i]);
	}

}


/*******************************************************************************/
/*				  MapPixToCol()                                                */
/* Maps cell information to soil column before soil temp calcs using the column */
/* information                                                                 */
/*                                                                             */
/*******************************************************************************/
void MapPixToCol(TEMPPIX* TempMap, VEGTABLE* VType, SOILTABLE* SType, float DepthToBedrock, TEMPTABLE* Stemp) {
	int i, j;
	float Zcrit, SoilBd, KhDry, Frac, Porosity;
	int Rz = 0;


	/*  NOTE: Frac is the fraction of the soil layer in the assigned rooting zone (Rz) (where Zb is) and the remainder is in Rz - 1 */

	(*Stemp).Emissivity = SType->Emissivity;
	for (i = 1; i < Stemp->NumLayers+1; i++) {
		Rz = TempMap->Rz[i];
		Frac = TempMap->Frac[i];

		if (Stemp->Zb[i] < DepthToBedrock) {  /* layer is in soil column */

			if (Frac == 1) { /* full layer in the same soil/root zone  ***************/
			
				SoilBd = 2700 * (1 - SType->Porosity[Rz]);
				KhDry = (0.135 * SoilBd + 64.7) / (2700 - 0.947 * SoilBd);
				Stemp->KhDry[i] = KhDry;
				Stemp->KhMin[i] = SType->KhSol[Rz];
				Stemp->ChMin[i] = SType->Ch[Rz];
				Stemp->SoilPorosity[i] = SType->Porosity[Rz];
			}
			else {  /*  layer splits soil/root zone  ********************************/
				Porosity = ((1.0 - Frac) * SType->Porosity[Rz - 1]) + (Frac * SType->Porosity[Rz]);
				SoilBd = 2700 * Porosity;
				KhDry = (0.135 * SoilBd + 64.7) / (2700 - 0.947 * SoilBd);
				Stemp->KhDry[i] = KhDry;

				Stemp->KhMin[i] = ((1.0 - Frac) * SType->KhSol[Rz - 1]) + (Frac * SType->KhSol[Rz]);
				Stemp->ChMin[i] = ((1.0 - Frac) * SType->Ch[Rz - 1]) + (Frac * SType->Ch[Rz]);
				Stemp->SoilPorosity[i] = Porosity;
			}

			/* these map directly by layer           */
			Stemp->SoilMoist[i] = TempMap->SoilMoist[i];
			Stemp->SoilIce[i] = TempMap->SoilIce[i];
			Stemp->Ts[i] = TempMap->Ts[i];

			/* calculate    */
			Stemp->Ch[i] = CalcCh(Stemp->ChMin[i], Stemp->SoilPorosity[i], Stemp->SoilIce[i], Stemp->SoilMoist[i]);

			Stemp->KhSol[i] = CalKhSol(Stemp->SoilMoist[i], Stemp->SoilIce[i], Stemp->SoilPorosity[i],
				Stemp->KhDry[i], Stemp->KhMin[i], Stemp->Ts[i]);
		}
		else {  /* node in bedrock  *************/

			if (Frac == 1 ) {  /* full layer in bedrock   */
				Stemp->KhDry[i] = 0;
				Stemp->KhMin[i] = 0;
				Stemp->ChMin[i] = 0;
				Stemp->SoilPorosity[i] = 0;
				Stemp->SoilMoist[i] = 0;
				Stemp->SoilIce[i] = 0;
				Stemp->Ch[i] = CH_BED;
				Stemp->KhSol[i] = KhBed;
				Stemp->Ts[i] = TempMap->Ts[i];
			}
			else {   /* partial layer in bedrock */
				Porosity = ((1.0 - Frac) * SType->Porosity[Rz - 1]);
				SoilBd = 2700 * Porosity;
				KhDry = (0.135 * SoilBd + 64.7) / (2700 - 0.947 * SoilBd);
				Stemp->KhDry[i] = KhDry;
				Stemp->KhMin[i] = (1.0 - Frac) * SType->KhSol[Rz - 1];
				Stemp->ChMin[i] = (1.0 - Frac) * SType->Ch[Rz - 1];
				Stemp->SoilPorosity[i] = Porosity;

				Stemp->SoilMoist[i] = (1.0 - Frac) * TempMap->SoilMoist[i];
				Stemp->SoilIce[i] = (1.0 - Frac) * TempMap->SoilIce[i];
				Stemp->Ts[i] = TempMap->Ts[i];

				/* calculate    */
				Stemp->Ch[i] = CalcCh(Stemp->KhMin[i], Stemp->SoilPorosity[i], Stemp->SoilIce[i], Stemp->SoilMoist[i]);

				Stemp->KhSol[i] = CalKhSol(Stemp->SoilMoist[i], Stemp->SoilIce[i], Stemp->SoilPorosity[i],
					Stemp->KhDry[i], Stemp->KhMin[i], Stemp->Ts[i]);

				Stemp->Ch[i] = ((1.0 - Frac) * Stemp->Ch[i]) + (Frac * CH_BED);
				Stemp->KhSol[i] = ((1.0 - Frac) * Stemp->KhSol[i]) + (Frac * KhBed);

			}
		}

		/*printf("%d %f  %d %f %f %f %f %f %f %f %f %f %f\n", i, Stemp->Zb[i], Rz, Frac, Stemp->Zn[i], Stemp->KhDry[i], Stemp->KhMin[i],
			Stemp->SoilPorosity[i], Stemp->SoilMoist[i], Stemp->SoilIce[i], Stemp->Ts[i], Stemp->Ch[i], Stemp->KhSol[i]); */


		float test = Stemp->KhSol[i];
	}

	Stemp->Ts[Stemp->NumLayers + 1] = TempMap->Ts[Stemp->NumLayers + 1];

}

/*******************************************************************************/
/*				  MapColToPix()                                                */
/* Map cell dependent information                                              */
/*                                                                             */
/*******************************************************************************/
void MapColToPix(TEMPPIX* TempMap, TEMPTABLE* Stemp) {
	int i;
	float test;

	for (i = 1; i < (*Stemp).NumLayers + 2; i++) {  
		TempMap->Ts[i] = Stemp->Ts[i];
		TempMap->SoilMoist[i] = Stemp->SoilMoist[i];
		TempMap->SoilIce[i] = Stemp->SoilIce[i];
	}

}