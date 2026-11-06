#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "data.h"
#include "functions.h"
#include "constants.h"

/*****************************************************************************************/
/*				  InitTempMap()                                                          */
/*                                                                                       */
/* Assign each soil temperature layer in the cell to a rooting zone (Rz) or bedrock.     */
/* "Frac" is the fraction of the soil temp layer in the assigned rooting zone. If the    */
/* entire layer is in a single RX, Frac = 1.0.  If the tamp layer is in two RZ, Frac     */
/* is the fraction of the layer in the assigned RZ and (1.0 - Frac) is the fraction in   */
/* rooting zone (Rz - 1). Set initial soil layer moisture, ice content, and temperature. */
/*                                                                                       */
/*****************************************************************************************/
void InitTempMap(TEMPPIX*** TempMap, MAPSIZE *Map, TOPOPIX ***TopoMap,  int NumLayers, float SoilIce,
	             VEGPIX*** VegMap, VEGTABLE** VType, SOILPIX*** SoilMap, SOILTABLE** SType, float Zb[]) {
	
	const char *Routine = "InitTempMap";
	int i, x, y, Vid, Sid;
	float Frac, Zcrit, DepthToBedrock; /* Frac is the fraction of the soil layer in the assigned Rz */
	//float *Zb;  /* bottom of individual soil layer */
	int Rz;
	//float SoilIce = 0.0;

	float test = Zb[2];

	/* allocate memory  */
	if (!(*TempMap = (TEMPPIX **)calloc(Map->NY, sizeof(TEMPPIX *))))
		ReportError((char *)Routine, 1);
	for (y = 0; y < Map->NY; y++) {
		(*TempMap)[y] = (TEMPPIX *)calloc(Map->NX, sizeof(TEMPPIX));
	}

	for (y = 0; y < Map->NY; y++) {
		for (x = 0; x < Map->NX; x++) {
			if (INBASIN((*TopoMap)[y][x].Mask)) {

			/* memory for NumLayers + 1 */
			(*TempMap)[y][x].SoilMoist = (float*)calloc((NumLayers + 2), sizeof(float));
			(*TempMap)[y][x].SoilIce = (float*)calloc((NumLayers + 2), sizeof(float));
			(*TempMap)[y][x].Ts = (float*)calloc((NumLayers + 2), sizeof(float));
			(*TempMap)[y][x].Rz = (unsigned char*)calloc((NumLayers + 2), sizeof(unsigned char));
			(*TempMap)[y][x].Frac = (float*)calloc((NumLayers + 2), sizeof(float));
			

			/* write to TempMap                        */
			for (i = 1; i < NumLayers + 2; i++) {
				/* assign soil moisture of the top soil layer to all soil temp layers as initial condition; 
				A more realistic assignment would be to assign soil moisture based on the fraction of the soil
				 temp layer in each rooting zone and the soil moisture in each rooting zone, 
				can be modified to assign different moisture to different layers if needed */
				
				(*TempMap)[y][x].SoilMoist[i] = (*SoilMap)[y][x].Moist[0]; 
				(*TempMap)[y][x].SoilIce[i] = SoilIce;  // initialy set as  0
				(*TempMap)[y][x].Ts[i] = 0; //(*SoilMap)[y][x].TSurf;
			}

			/* assign rooting zone (Rz) and Frac to each layer  */
			Rz = 0;
			Vid = (*VegMap)[y][x].Veg - 1;
			Sid = (*SoilMap)[y][x].Soil - 1;
			Zcrit = (*VType)[Vid].RootDepth[Rz];
			DepthToBedrock = (*SoilMap)[y][x].Depth;

		    for (i = 1; i < NumLayers+1; i++) {

			    Frac = -1;

			    if (Zb[i] < DepthToBedrock) {

				    if (Zb[i] > Zcrit && Rz < (*VType)[Vid].NSoilLayers - 1) {
					   Frac = (Zb[i] - Zcrit) / (Zb[i] - Zb[i - 1]); /* frac in root zone at Zb  */
					   Rz++;
					   Zcrit += (*VType)[Vid].RootDepth[Rz];
				    }

				    if (Frac < 0.0) { /* full layer in the same soil/root zone  ***************/

					   (*TempMap)[y][x].Rz[i] = Rz;
					   (*TempMap)[y][x].Frac[i] = 1.0;
				    }
				    else {  /*  layer splits soil/root zone  ********************************/
					    (*TempMap)[y][x].Rz[i] = Rz;
					    (*TempMap)[y][x].Frac[i] = Frac;
				    }

			    }
				else {  /* node in bedrock  *************/
					if (Zb[i - 1] < DepthToBedrock) {  /* layer partially in bedrock  */
						(*TempMap)[y][x].Rz[i] = (*VType)[Vid].NSoilLayers;
						(*TempMap)[y][x].Frac[i] = (Zb[i] - DepthToBedrock) / (Zb[i] - Zb[i - 1]); /* frac in bedrock  */
					}
					else {   /* full layer in bedrock */
						(*TempMap)[y][x].Rz[i] = (*VType)[Vid].NSoilLayers;
						(*TempMap)[y][x].Frac[i] = 1.0;
					}
				}

				//printf("Zb: %.4f Rz: %d Frac: %.4f \n", Zb[i], (*TempMap)[y][x].Rz[i], (*TempMap)[y][x].Frac[i]);
			}  /* end layer loop */
			}	
		}
	}
	
	//free(Zn);
}