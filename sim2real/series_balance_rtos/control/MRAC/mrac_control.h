/*
 * File: mrac_control.h
 *
 * Code generated for Simulink model 'mrac_control'.
 *
 * Model version                  : 1.1
 * Simulink Coder version         : 23.2 (R2023b) 01-Aug-2023
 * C/C++ source code generated on : Sat Mar 29 14:22:14 2025
 *
 * Target selection: ert.tlc
 * Embedded hardware selection: ARM Compatible->ARM Cortex-M
 * Code generation objectives:
 *    1. Execution efficiency
 *    2. RAM efficiency
 * Validation result: Not run
 */

#ifndef RTW_HEADER_mrac_control_h_
#define RTW_HEADER_mrac_control_h_
#ifndef mrac_control_COMMON_INCLUDES_
#define mrac_control_COMMON_INCLUDES_
#include <stdbool.h>
#include <stdint.h>
#include "rtw_continuous.h"
#include "rtw_solver.h"
#endif                                 /* mrac_control_COMMON_INCLUDES_ */

#include <string.h>

/* Macros for accessing real-time model data structure */
#ifndef rtmGetErrorStatus
#define rtmGetErrorStatus(rtm)         ((rtm)->errorStatus)
#endif

#ifndef rtmSetErrorStatus
#define rtmSetErrorStatus(rtm, val)    ((rtm)->errorStatus = (val))
#endif

#ifndef rtmGetStopRequested
#define rtmGetStopRequested(rtm)       ((rtm)->Timing.stopRequestedFlag)
#endif

#ifndef rtmSetStopRequested
#define rtmSetStopRequested(rtm, val)  ((rtm)->Timing.stopRequestedFlag = (val))
#endif

#ifndef rtmGetStopRequestedPtr
#define rtmGetStopRequestedPtr(rtm)    (&((rtm)->Timing.stopRequestedFlag))
#endif

#ifndef rtmGetT
#define rtmGetT(rtm)                   (rtmGetTPtr((rtm))[0])
#endif

#ifndef rtmGetTPtr
#define rtmGetTPtr(rtm)                ((rtm)->Timing.t)
#endif

#ifndef rtmGetTStart
#define rtmGetTStart(rtm)              ((rtm)->Timing.tStart)
#endif

/* Forward declaration for rtModel */
typedef struct tag_RTM RT_MODEL;

/* Block signals and states (default storage) for system '<Root>' */
typedef struct {
  double gamma_Kr[40];                 /* '<S18>/gamma_Kr' */
  double gamma_Kx[40];                 /* '<S15>/gamma_Kx' */
  double Gain5[40];                    /* '<S11>/Gain5' */
  int Integrator_IWORK;                /* '<S15>/Integrator' */
  int Integrator_IWORK_n;              /* '<S18>/Integrator' */
} DW;

/* Continuous states (default storage) */
typedef struct {
  double ReferenceSystem_CSTATE[10];   /* '<S4>/Reference System' */
  double Integrator_CSTATE[40];        /* '<S15>/Integrator' */
  double Integrator_CSTATE_a[40];      /* '<S18>/Integrator' */
  double Integrator2_CSTATE[40];       /* '<S11>/Integrator2' */
} X;

/* State derivatives (default storage) */
typedef struct {
  double ReferenceSystem_CSTATE[10];   /* '<S4>/Reference System' */
  double Integrator_CSTATE[40];        /* '<S15>/Integrator' */
  double Integrator_CSTATE_a[40];      /* '<S18>/Integrator' */
  double Integrator2_CSTATE[40];       /* '<S11>/Integrator2' */
} XDot;

/* State disabled  */
typedef struct {
  bool ReferenceSystem_CSTATE[10];     /* '<S4>/Reference System' */
  bool Integrator_CSTATE[40];          /* '<S15>/Integrator' */
  bool Integrator_CSTATE_a[40];        /* '<S18>/Integrator' */
  bool Integrator2_CSTATE[40];         /* '<S11>/Integrator2' */
} XDis;

/* Invariant block signals (default storage) */
typedef struct {
  const double Kx0[40];                /* '<S15>/Kx0' */
  const double Kr0[40];                /* '<S18>/Kr0' */
} ConstB;

#ifndef ODE3_INTG
#define ODE3_INTG

/* ODE3 Integration Data */
typedef struct {
  double *y;                           /* output */
  double *f[3];                        /* derivatives */
} ODE3_IntgData;

#endif

/* Constant parameters (default storage) */
typedef struct {
  /* Computed Parameter: ReferenceSystem_A_pr
   * Referenced by: '<S4>/Reference System'
   */
  double ReferenceSystem_A_pr[47];

  /* Computed Parameter: ReferenceSystem_B_pr
   * Referenced by: '<S4>/Reference System'
   */
  double ReferenceSystem_B_pr[42];

  /* Pooled Parameter (Expression: blkParam.P)
   * Referenced by:
   *   '<S13>/Gain'
   *   '<S16>/Gain2'
   *   '<S11>/Gain4'
   */
  double pooled2[100];

  /* Pooled Parameter (Expression: B)
   * Referenced by:
   *   '<S13>/Gain1'
   *   '<S16>/Gain3'
   *   '<S11>/Gain5'
   */
  double pooled3[40];

  /* Computed Parameter: ReferenceSystem_A_ir
   * Referenced by: '<S4>/Reference System'
   */
  uint32_t ReferenceSystem_A_ir[47];

  /* Computed Parameter: ReferenceSystem_A_jc
   * Referenced by: '<S4>/Reference System'
   */
  uint32_t ReferenceSystem_A_jc[11];

  /* Computed Parameter: ReferenceSystem_B_ir
   * Referenced by: '<S4>/Reference System'
   */
  uint32_t ReferenceSystem_B_ir[42];

  /* Computed Parameter: ReferenceSystem_B_jc
   * Referenced by: '<S4>/Reference System'
   */
  uint32_t ReferenceSystem_B_jc[11];
} ConstP;

/* Real-time Model Data Structure */
struct tag_RTM {
  const char *errorStatus;
  RTWSolverInfo solverInfo;
  X *contStates;
  int *periodicContStateIndices;
  double *periodicContStateRanges;
  double *derivs;
  XDis *contStateDisabled;
  bool zCCacheNeedsReset;
  bool derivCacheNeedsReset;
  bool CTOutputIncnstWithState;
  double odeY[130];
  double odeF[3][130];
  ODE3_IntgData intgData;
  DW *dwork;

  /*
   * Sizes:
   * The following substructure contains sizes information
   * for many of the model attributes such as inputs, outputs,
   * dwork, sample times, etc.
   */
  struct {
    int numContStates;
    int numPeriodicContStates;
    int numSampTimes;
  } Sizes;

  /*
   * Timing:
   * The following substructure contains information regarding
   * the timing information for the model.
   */
  struct {
    uint32_t clockTick0;
    double stepSize0;
    uint32_t clockTick1;
    bool firstInitCondFlag;
    double tStart;
    SimTimeStep simTimeStep;
    bool stopRequestedFlag;
    double *t;
    double tArray[2];
  } Timing;
};

extern const ConstB rtConstB;          /* constant block i/o */

/* Constant parameters (default storage) */
extern const ConstP rtConstP;

/* Model entry point functions */
extern void mrac_control_initialize(RT_MODEL *const rtM);
extern void mrac_control_step(RT_MODEL *const rtM, double rtU_r[10], double
  rtU_state[10], double rtY_u[4]);

/*-
 * These blocks were eliminated from the model due to optimizations:
 *
 * Block '<S1>/phi' : Unused code path elimination
 * Block '<S11>/Gamma_w' : Eliminated nontunable gain of 1
 */

/*-
 * The generated code includes comments that allow you to trace directly
 * back to the appropriate location in the model.  The basic format
 * is <system>/block_name, where system is the system number (uniquely
 * assigned by Simulink) and block_name is the name of the block.
 *
 * Use the MATLAB hilite_system command to trace the generated code back
 * to the model.  For example,
 *
 * hilite_system('<S3>')    - opens system 3
 * hilite_system('<S3>/Kp') - opens and selects block Kp which resides in S3
 *
 * Here is the system hierarchy for this model
 *
 * '<Root>' : 'mrac_control'
 * '<S1>'   : 'mrac_control/Model Reference Adaptive Control'
 * '<S2>'   : 'mrac_control/Model Reference Adaptive Control/MRAC Subsystem'
 * '<S3>'   : 'mrac_control/Model Reference Adaptive Control/MRAC Subsystem/MRAC Controller'
 * '<S4>'   : 'mrac_control/Model Reference Adaptive Control/MRAC Subsystem/MRAC Controller/Direct'
 * '<S5>'   : 'mrac_control/Model Reference Adaptive Control/MRAC Subsystem/MRAC Controller/Direct/Disturbance Model'
 * '<S6>'   : 'mrac_control/Model Reference Adaptive Control/MRAC Subsystem/MRAC Controller/Direct/Feedback Control'
 * '<S7>'   : 'mrac_control/Model Reference Adaptive Control/MRAC Subsystem/MRAC Controller/Direct/Feedforward Control'
 * '<S8>'   : 'mrac_control/Model Reference Adaptive Control/MRAC Subsystem/MRAC Controller/Direct/Disturbance Model/CONV_SHL_MRAC'
 * '<S9>'   : 'mrac_control/Model Reference Adaptive Control/MRAC Subsystem/MRAC Controller/Direct/Disturbance Model/Feature Vector'
 * '<S10>'  : 'mrac_control/Model Reference Adaptive Control/MRAC Subsystem/MRAC Controller/Direct/Disturbance Model/CONV_SHL_MRAC/Conventional_MRAC'
 * '<S11>'  : 'mrac_control/Model Reference Adaptive Control/MRAC Subsystem/MRAC Controller/Direct/Disturbance Model/CONV_SHL_MRAC/Conventional_MRAC/default'
 * '<S12>'  : 'mrac_control/Model Reference Adaptive Control/MRAC Subsystem/MRAC Controller/Direct/Disturbance Model/Feature Vector/State'
 * '<S13>'  : 'mrac_control/Model Reference Adaptive Control/MRAC Subsystem/MRAC Controller/Direct/Feedback Control/Adaptive'
 * '<S14>'  : 'mrac_control/Model Reference Adaptive Control/MRAC Subsystem/MRAC Controller/Direct/Feedback Control/Adaptive/Feedback Update'
 * '<S15>'  : 'mrac_control/Model Reference Adaptive Control/MRAC Subsystem/MRAC Controller/Direct/Feedback Control/Adaptive/Feedback Update/default'
 * '<S16>'  : 'mrac_control/Model Reference Adaptive Control/MRAC Subsystem/MRAC Controller/Direct/Feedforward Control/Adaptive'
 * '<S17>'  : 'mrac_control/Model Reference Adaptive Control/MRAC Subsystem/MRAC Controller/Direct/Feedforward Control/Adaptive/Feedforward Update'
 * '<S18>'  : 'mrac_control/Model Reference Adaptive Control/MRAC Subsystem/MRAC Controller/Direct/Feedforward Control/Adaptive/Feedforward Update/default'
 */
#endif                                 /* RTW_HEADER_mrac_control_h_ */

/*
 * File trailer for generated code.
 *
 * [EOF]
 */
