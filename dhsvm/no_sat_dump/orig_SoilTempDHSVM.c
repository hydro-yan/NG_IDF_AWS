/*
 * SUMMARY:      SoilTempDHSVM.c - Bridge between DHSVM and soil temperature module
 * USAGE:        Part of DHSVM with soil temperature modeling extension
 *
 * AUTHOR:       Zhuoran Duan, zhuoran.duan@pnnl.gov;
 *               Mark Wigmosta, 
 * ORG:          PNNL
 * ORIG-DATE:    Apr-2026
 * DESCRIPTION:  Bridge/wrapper module that integrates the validated soil temperature
 *               module with DHSVM. Provides minimal-modification integration via
 *               modular wrapper functions.
 * DESCRIP-END.
 * FUNCTIONS:    InitSoilTempDHSVM(), UpdateSoilTempPerCell(), FreeSoilTempDHSVM(),
 *               UpdateSoilMapTemp(), CalcSurfaceConductance()
 * COMMENTS:
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "constants.h"
#include "settings.h"
#include "data.h"
#include "functions.h"
#include "Calendar.h"
#include "snow.h"

/* Import soil temperature module headers */
/* Adjust these paths based on build system configuration */


/* ========================================================================== */
/*              INITIALIZATION FUNCTIONS                                     */
/* ========================================================================== */

/*****************************************************************************
  InitSoilTempDHSVM() - Initialize soil temperature model for DHSVM
  
  Allocates memory and initializes all structures needed for soil temperature
  calculations across the entire grid domain.
*****************************************************************************/
// int InitSoilTempDHSVM(MAPSIZE *Map, SOILTABLE *SType, SOILPIX **SoilMap,
//                       VEGPIX **VegMap, VEGTABLE *VType, TEMPPIX **TempMap, 
//                       TEMPTABLE *Stemp, float *Zb, int NumLayers)
// {
//     int x, y, status;
//     float SoilIce = 0;  /* Initial ice content (m³/m³), can be set to 0 or a small value; could also be made an input parameter if needed */
//     float Alpha = 0.6;  /* Default alpha value for soil temperature calculations, can be adjusted or made an input parameter */ 
//     // FIXME: The alpha value and other parameters could be added to the Options structure and read from the config file for more flexibility.
//     // FIXME: the alpha can be set in data.h like max layer 
//     int NumTempLayers = (NumLayers != NULL) ? NumLayers : MAX_LAYERS - 2;  /* Default to MAX_LAYERS - 2 if not provided */
//     // if (SoilTempState.Initialized) {  // FIXME: The default is 0, and I don't see where it is set to 1, so this check may not work as intended. Need to set SoilTempState.Initialized = 1 at the end of successful initialization.
//     //     fprintf(stderr, "Warning: Soil temperature model already initialized\n");
//     //     return 0;
//     // }
    
//     printf("Initializing soil temperature model for DHSVM...\n");
//     //printf("  Grid size: %d x %d\n", Map->NX, Map->NY);
//     printf("  Number of temperature layers: %d\n", NumLayers);

    
//     /* start, from InitDhsvmData() */
//     /* Create Zb array if not provided (use default from InitDhsvmData) */
//     float ZbDefault[MAX_LAYERS + 1] = {  // FIXME: NEED to define MAX_LAYERS in data.h/constance.h or config file
//         0.000, 0.020, 0.060, 0.120, 0.200, 0.320, 0.480, 0.680, 0.920, 1.200, 
//         1.520, 1.880, 2.280, 2.720, 3.260, 3.900, 4.640, 5.480, 6.420, 7.460, 
//         8.600, 10.990, 15.666, 23.301, 34.441, 49.556, 70.00 }; //27 total
        
//     float *ZbToUse = (Zb != NULL) ? Zb : ZbDefault;

//     /* Initialize soil temperature structures via soil temp module functions */
//     printf("Building soil temperature column structure...\n");

//     /* Debug: dump first Zb values to verify contents */
//     printf("DEBUG: ZbToUse[0..9]:\n");
//     for (x = 0; x < 10; x++) {
//         if (isfinite((double)ZbToUse[x]))
//             printf(" Zb[%d]=%f\n", x, ZbToUse[x]);
//         else
//             printf(" Zb[%d]=%f (non-finite)\n", x, ZbToUse[x]);
//     }

// 	BuildCol(Stemp, ZbToUse);

//     printf("Initializing temperature map (rooting zone mapping)...\n");
	
//     InitTempMap(&TempMap, Map, TopoMap, NumTempLayers, SoilIce, &VegMap, &VType, &SoilMap, &SType, ZbToUse);

//     // if (status != 0) {
//     //     fprintf(stderr, "Error: TempMap initialization failed with status %d\n", status);
//     //     return status;
//     // }
    
//     printf("Soil temperature model initialized successfully\n");
    
//     return 0;
// }

/* ========================================================================== */
/*              PER-TIMESTEP CORE FUNCTIONS                                  */
/* ========================================================================== */

/*****************************************************************************
  UpdateSoilTempPerCell() - Calculate soil temperature for one cell per timestep
  
  This is the main per-timestep function called within DHSVM's grid loop.
  It orchestrates the soil temperature calculation workflow.
*****************************************************************************/
void UpdateSoilTempPerCell(int x, int y, PIXMET *LocalMet, int Dt,
                        VEGTABLE *VType, VEGPIX *LocalVeg, SOILTABLE *SType,
                        SOILPIX *LocalSoil, TEMPPIX *LocalSoilTemp, TEMPTABLE *Stemp,
                        SNOWPIX *LocalSnow, OPTIONSTRUCT *Options)
{
    float UpperCond, UpperTemp, Solar, Tsurf, T3;
    int status;
    float PackSwq, Ice, SurfaceSwq;
    /* Extract column data from grid cell (MapPixToCol) */
    MapPixToCol(LocalSoilTemp, VType, SType, LocalSoil->Depth, Stemp);

    /*Decide which temperature to use for the upper boundary condition based on snow conditions */
    if (LocalSnow->HasSnow) {
        /*If Pack is present, use pack temperature, if not, use the warmer temperature between air and snow pack*/

        /*Copied from SnowMelt.c so no need to change that function*/
        /* Initialize snowpack variables */
        Ice = LocalSnow->Swq - LocalSnow->PackWater - LocalSnow->SurfWater;

        /* Reconstruct snow pack */
        if (Ice > MAX_SURFACE_SWE)
            SurfaceSwq = MAX_SURFACE_SWE;
        else
            SurfaceSwq = Ice;
        PackSwq = Ice - SurfaceSwq;
        /*End of SnowMelt.c*/
    
        if (PackSwq > 0.0) {  /* If pack snow is present, use pack temperature for upper boundary condition */
            UpperTemp = LocalSnow->TPack;  /* Example snow temperature (°C), could be refined based on actual conditions or model parameters */
        } else {
            
            UpperTemp = (LocalMet->Tair > LocalSnow->TSurf) ? LocalMet->Tair : LocalSnow->TSurf;  /* Air temperature (°C) */
        }
        Solar = 0;            /* no incoming shortwave radiation when snow present */
    } else {
        /* Extract meteorological data for upper boundary condition */
        UpperTemp = LocalMet->Tair;      /* Air temperature (°C) */
        Solar = LocalMet->Sin;            /* Incoming shortwave radiation (W/m²) */
    }
    LocalSoilTemp->Ts[0] = UpperTemp;  /* Set surface layer temperature to air temp for boundary condition */

    
    // /* Compute surface conductance (radiative + convective heat transfer) */
    // Tsurf = LocalSoilTemp->Ts[0] + 273.15;  /* Convert to Kelvin for radiation calc */
    // T3 = Tsurf * Tsurf * Tsurf;
    
    // /* Radiative conductance (4th order term in Stefan-Boltzmann) */
    // float CRad = 4.0 * Stemp->Emissivity * STEFAN * T3;
    
    /* Simple convective conductance function of wind speed */
    //float CConv = 1.5 + 1.5 * LocalMet->Wind;  /* W/(m²·K), empirical relation */
    
    //UpperCond = CRad + CConv;
    UpperCond = Stemp->KhSol[1] / Stemp->Zn[1];
    
    /* Solve 1D heat equation using Crank-Nicholson scheme */
    CalcSoilTemp(Stemp, Dt, UpperCond, UpperTemp, Solar);
    
    CalcPhaseChange(Stemp, Dt);

    /* Map column results back to grid cell (MapColToPix) */
    MapColToPix(LocalSoilTemp, Stemp);
    
    // /* Update DHSVM soil temperature: Map 27 detailed layers to 3 DHSVM layers */
    // UpdateSoilMapTemp(SoilPixCell, &SoilTempState.TempMap[y][x]);
    
    return;
}
