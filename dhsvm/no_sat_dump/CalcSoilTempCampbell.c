#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "data.h"
#include "functions.h"
#include "constants.h"


/**************************************************************************************/
/*				  CalcSoilTemp()                                                      */
/*                                                                                    */
/* Calculate soil temperature using a Crank-Nicholson type approach as described in:  */
/* G.S. Campbell, 1985, Soil Physics with Basic, Transport Models for Soil - Plant    */
/* Systems
/*                                                                                    */
/**************************************************************************************/
void CalcSoilTemp(TEMPTABLE *S, int Dt, float UpperCond, float UpperTemp, float Solar){
  
    /* read from config file  */

	/*********************************************************************************/
	/* NOTE: WE CONVERT TEMPS TO KELVIN, DO THE CALCS, THEN CONVERT BACK TO CELSIUS  */
	/*********************************************************************************/


	/* active soil nodes are at Zn[1] through Zn[S->NumLayers] */
 
	int i, Num;
	float Adj,Cp;
	float *A, *B, *C, *D, *K;
	float t, T3;;

	/* setup tri-diagonal matrix */
	Num = S->NumLayers + 2; /* includes upper and lower boundary nodes */
	A = (float*)calloc(Num, sizeof(float));
	B = (float*)calloc(Num, sizeof(float));
	C = (float*)calloc(Num, sizeof(float));
	D = (float*)calloc(Num, sizeof(float));
	K = (float*)calloc(Num, sizeof(float));
	
	/* precalculate conductance K (K[0] is atmospheric cond) and volumetric heat capacity */
	// K[0] = UpperCond;
	S->Ts[0] = UpperTemp;

	/* work in Kelvin  */
	for (i = 0; i < S->NumLayers + 2; i++) {
		S->Ts[i] += 273.15;
	}

	for (i = 1; i < S->NumLayers + 1; i++) {
		K[i] = S->KhSol[i] / (S->Zn[i + 1] - S->Zn[i]);  /* Campbell eqn 4.8 */
	}
	if (NO_FLUX_BC) {
		K[S->NumLayers] = 0.0;  /* test  */
	}


	/* setup tri-diagonal matrix */  
	
	Adj = 1.0 - S->Alpha;

	/* top layer */
	i = 1;

	T3 = S->Ts[i - 1] * S->Ts[i - 1] * S->Ts[i - 1];
	K[i-1] = UpperCond + 4 * S->Emissivity * STEFAN * T3; /* upper boundary condition */

	/* Campbell eqns 4.12 - 4.15 */
	Cp = S->Ch[i] * (S->Zn[i + 1] - S->Zn[i - 1]) / (2 * Dt);
	A[i] = 0.0;
	B[i] = S->Alpha * (K[i] + K[i - 1]) + Cp;
	C[i] = -1.0 * S->Alpha * K[i];
	D[i] = (Adj * K[i - 1] * S->Ts[i - 1]) + ((Cp - Adj * (K[i] + K[i - 1])) * S->Ts[i]) + (Adj * K[i] * S->Ts[i + 1]);
    
	T3 = S->Ts[i - 1] * S->Ts[i - 1] * S->Ts[i - 1];
	t = S->Emissivity * STEFAN * T3 * S->Ts[i - 1];
	D[i] += (S->Alpha * K[i-1] * S->Ts[i - 1]) + (Solar - S->Emissivity * STEFAN * T3 * S->Ts[i - 1]);  /* adj. for upper BC Ts[i - 1] is air/snowpack temp */

	//printf(" %d %f %f %f %f %f %f\n", i, K[i], Cp, A[i], B[i], C[i], D[i]);

	/* interior layers */
	for (i = 2; i < S->NumLayers + 1; i++) {
		Cp = S->Ch[i] * (S->Zn[i + 1] - S->Zn[i - 1]) / (2 * Dt);
		A[i] = -1.0 * S->Alpha * K[i-1];
		B[i] = S->Alpha * (K[i] + K[i - 1]) + Cp;
		C[i] = -1.0 * S->Alpha * K[i];
		D[i] = (Adj * K[i - 1] * S->Ts[i - 1]) + ((Cp - Adj * (K[i] + K[i - 1])) * S->Ts[i]) + (Adj * K[i] * S->Ts[i + 1]);

		//printf(" %d %f %f %f %f %f %f\n", i, K[i], Cp, A[i], B[i], C[i], D[i]);

	}
	D[S->NumLayers] += S->Alpha * K[S->NumLayers] * S->Ts[S->NumLayers + 1];  /* adjust for lower boundary condition */

	/* solve with Thomas Algorithm  */
	for (i = 1; i < S->NumLayers; i++) {
		C[i] = C[i] / B[i];
		D[i] = D[i] / B[i];
		B[i + 1] = B[i + 1] - A[i + 1] * C[i];
		D[i + 1] = D[i + 1] - A[i + 1] * D[i];
	}
	
	/* new temperature at lowest layer  */
	S->Ts[S->NumLayers] = D[S->NumLayers] / B[S->NumLayers];

	//printf("%d %f %f %f\n", S->NumLayers, S->Ts[S->NumLayers], D[S->NumLayers], B[S->NumLayers]);
	//printf("\n%d %f %f %f %f\n", S->NumLayers, S->Ts[S->NumLayers], D[S->NumLayers], C[S->NumLayers], S->Ts[S->NumLayers + 1]);

	/* temperature for interior layers      */
	for (i = (S->NumLayers - 1); i << 1 >  0; i--) {
		S->Ts[i] = D[i] - C[i] * S->Ts[i + 1];
		//printf("%d %f %f %f %f\n", i, S->Ts[i], D[i], C[i], S->Ts[i + 1]);
		//float t = D[i] - C[i] * S->Ts[i + 1];
	}

	/* Celsius for the rest of the model  */
/* work in Kelvin  */
	for (i = 0; i < S->NumLayers + 1; i++) S->Ts[i] -= 273.15;
	
	/* cleanup   */
	free(A);
	free(B);
	free(C);
	free(D);
	free(K);

	A = NULL;
	B = NULL;
	C = NULL;
	D = NULL;
	K = NULL;

}

