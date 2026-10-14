///*
// * File: mrac_control.c
// *
// * Code generated for Simulink model 'mrac_control'.
// *
// * Model version                  : 1.1
// * Simulink Coder version         : 23.2 (R2023b) 01-Aug-2023
// * C/C++ source code generated on : Sat Mar 29 14:22:14 2025
// *
// * Target selection: ert.tlc
// * Embedded hardware selection: ARM Compatible->ARM Cortex-M
// * Code generation objectives:
// *    1. Execution efficiency
// *    2. RAM efficiency
// * Validation result: Not run
// */

//#include "mrac_control.h"
//#include <string.h>
//#include <stdint.h>

///* Private macros used by the generated code to access rtModel */
//#ifndef rtmSetFirstInitCond
//#define rtmSetFirstInitCond(rtm, val)  ((rtm)->Timing.firstInitCondFlag = (val))
//#endif

//#ifndef rtmIsFirstInitCond
//#define rtmIsFirstInitCond(rtm)        ((rtm)->Timing.firstInitCondFlag)
//#endif

//#ifndef rtmIsMajorTimeStep
//#define rtmIsMajorTimeStep(rtm)        (((rtm)->Timing.simTimeStep) == MAJOR_TIME_STEP)
//#endif

//#ifndef rtmIsMinorTimeStep
//#define rtmIsMinorTimeStep(rtm)        (((rtm)->Timing.simTimeStep) == MINOR_TIME_STEP)
//#endif

//#ifndef rtmSetTPtr
//#define rtmSetTPtr(rtm, val)           ((rtm)->Timing.t = (val))
//#endif

///* private model entry point functions */
//extern void mrac_control_derivatives(RT_MODEL *const rtM, double rtU_r[10]);

///*
// * This function updates continuous states using the ODE3 fixed-step
// * solver algorithm
// */
//static void rt_ertODEUpdateContinuousStates(RTWSolverInfo *si , RT_MODEL *const
//  rtM, double rtU_r[10], double rtU_state[10], double rtY_u[4])
//{
//  /* Solver Matrices */
//  static const double rt_ODE3_A[3] = {
//    1.0/2.0, 3.0/4.0, 1.0
//  };

//  static const double rt_ODE3_B[3][3] = {
//    { 1.0/2.0, 0.0, 0.0 },

//    { 0.0, 3.0/4.0, 0.0 },

//    { 2.0/9.0, 1.0/3.0, 4.0/9.0 }
//  };

//  double t = rtsiGetT(si);
//  double tnew = rtsiGetSolverStopTime(si);
//  double h = rtsiGetStepSize(si);
//  double *x = rtsiGetContStates(si);
//  ODE3_IntgData *id = (ODE3_IntgData *)rtsiGetSolverData(si);
//  double *y = id->y;
//  double *f0 = id->f[0];
//  double *f1 = id->f[1];
//  double *f2 = id->f[2];
//  double hB[3];
//  int i;
//  int nXc = 130;
//  rtsiSetSimTimeStep(si,MINOR_TIME_STEP);

//  /* Save the state values at time t in y, we'll use x as ynew. */
//  (void) memcpy(y, x,
//                (unsigned int)nXc*sizeof(double));

//  /* Assumes that rtsiSetT and ModelOutputs are up-to-date */
//  /* f0 = f(t,y) */
//  rtsiSetdX(si, f0);
//  mrac_control_derivatives(rtM, rtU_r);

//  /* f(:,2) = feval(odefile, t + hA(1), y + f*hB(:,1), args(:)(*)); */
//  hB[0] = h * rt_ODE3_B[0][0];
//  for (i = 0; i < nXc; i++) {
//    x[i] = y[i] + (f0[i]*hB[0]);
//  }

//  rtsiSetT(si, t + h*rt_ODE3_A[0]);
//  rtsiSetdX(si, f1);
//  mrac_control_step(rtM, rtU_r, rtU_state, rtY_u);
//  mrac_control_derivatives(rtM, rtU_r);

//  /* f(:,3) = feval(odefile, t + hA(2), y + f*hB(:,2), args(:)(*)); */
//  for (i = 0; i <= 1; i++) {
//    hB[i] = h * rt_ODE3_B[1][i];
//  }

//  for (i = 0; i < nXc; i++) {
//    x[i] = y[i] + (f0[i]*hB[0] + f1[i]*hB[1]);
//  }

//  rtsiSetT(si, t + h*rt_ODE3_A[1]);
//  rtsiSetdX(si, f2);
//  mrac_control_step(rtM, rtU_r, rtU_state, rtY_u);
//  mrac_control_derivatives(rtM, rtU_r);

//  /* tnew = t + hA(3);
//     ynew = y + f*hB(:,3); */
//  for (i = 0; i <= 2; i++) {
//    hB[i] = h * rt_ODE3_B[2][i];
//  }

//  for (i = 0; i < nXc; i++) {
//    x[i] = y[i] + (f0[i]*hB[0] + f1[i]*hB[1] + f2[i]*hB[2]);
//  }

//  rtsiSetT(si, tnew);
//  rtsiSetSimTimeStep(si,MAJOR_TIME_STEP);
//}

///* Model step function */
//void mrac_control_step(RT_MODEL *const rtM, double rtU_r[10], double rtU_state
//  [10], double rtY_u[4])
//{
//  DW *rtDW = rtM->dwork;
//  X *rtX = rtM->contStates;
//  double rtb_Transpose[100];
//  double rtb_Transpose1[100];
//  double rtb_Transpose_f[100];
//  double tmp[100];
//  double rtb_Integrator2_0[40];
//  double rtb_Integrator2_1[40];
//  double rtb_Transpose_1[4];
//  double rtb_Transpose_p[4];
//  double rtb_Transpose1_0;
//  double rtb_Transpose_2;
//  int iy;
//  int32_t ri;
//  int32_t rtb_Transpose_p_tmp;
//  if (rtmIsMajorTimeStep(rtM)) {
//    /* set solver stop time */
//    rtsiSetSolverStopTime(&rtM->solverInfo,((rtM->Timing.clockTick0+1)*
//      rtM->Timing.stepSize0));
//  }                                    /* end MajorTimeStep */

//  /* Update absolute time of base rate at minor time step */
//  if (rtmIsMinorTimeStep(rtM)) {
//    rtM->Timing.t[0] = rtsiGetT(&rtM->solverInfo);
//  }

//  /* StateSpace: '<S4>/Reference System' */
//  memset(&rtb_Transpose1[0], 0, 10U * sizeof(double));
//  for (iy = 0; iy < 10; iy++) {
//    for (ri = iy; (uint32_t)ri < (uint32_t)iy + 1U; ri = (int32_t)((uint32_t)ri
//          + 1U)) {
//      rtb_Transpose1[ri] += rtX->ReferenceSystem_CSTATE[(uint32_t)iy];
//    }
//  }

//  /* End of StateSpace: '<S4>/Reference System' */
//  for (ri = 0; ri < 10; ri++) {
//    /* Sum: '<S4>/Sum1' incorporates:
//     *  Inport: '<Root>/state'
//     */
//    rtb_Transpose1_0 = rtU_state[ri] - rtb_Transpose1[ri];
//    rtb_Transpose1[ri] = rtb_Transpose1_0;

//    /* Math: '<S11>/Transpose' */
//    rtb_Transpose[ri] = rtb_Transpose1_0;

//    /* Math: '<S13>/Transpose1' */
//    rtb_Transpose_f[ri] = rtb_Transpose1_0;

//    /* Product: '<S16>/re(t)' incorporates:
//     *  Inport: '<Root>/r'
//     */
//    for (iy = 0; iy < 10; iy++) {
//      tmp[iy + 10 * ri] = rtU_r[iy] * rtb_Transpose1[ri];
//    }

//    /* End of Product: '<S16>/re(t)' */
//  }

//  for (iy = 0; iy < 10; iy++) {
//    /* Gain: '<S16>/Gain2' */
//    for (ri = 0; ri < 10; ri++) {
//      rtb_Transpose1_0 = 0.0;
//      for (rtb_Transpose_p_tmp = 0; rtb_Transpose_p_tmp < 10;
//           rtb_Transpose_p_tmp++) {
//        rtb_Transpose1_0 += tmp[10 * rtb_Transpose_p_tmp + iy] *
//          rtConstP.pooled2[10 * ri + rtb_Transpose_p_tmp];
//      }

//      rtb_Transpose1[iy + 10 * ri] = rtb_Transpose1_0;
//    }

//    /* End of Gain: '<S16>/Gain2' */

//    /* Gain: '<S16>/Gain3' */
//    for (ri = 0; ri < 4; ri++) {
//      rtb_Transpose1_0 = 0.0;
//      for (rtb_Transpose_p_tmp = 0; rtb_Transpose_p_tmp < 10;
//           rtb_Transpose_p_tmp++) {
//        rtb_Transpose1_0 += rtb_Transpose1[10 * rtb_Transpose_p_tmp + iy] *
//          rtConstP.pooled3[10 * ri + rtb_Transpose_p_tmp];
//      }

//      rtb_Integrator2_1[iy + 10 * ri] = rtb_Transpose1_0;
//    }
//  }

//  for (ri = 0; ri < 40; ri++) {
//    /* Gain: '<S18>/gamma_Kr' incorporates:
//     *  Gain: '<S16>/Gain3'
//     */
//    rtDW->gamma_Kr[ri] = -5.0 * rtb_Integrator2_1[ri];
//  }

//  /* Product: '<S13>/x(t)e(t)' incorporates:
//   *  Inport: '<Root>/state'
//   */
//  for (iy = 0; iy < 10; iy++) {
//    for (ri = 0; ri < 10; ri++) {
//      tmp[ri + 10 * iy] = rtU_state[ri] * rtb_Transpose_f[iy];
//    }
//  }

//  /* End of Product: '<S13>/x(t)e(t)' */
//  for (iy = 0; iy < 10; iy++) {
//    /* Gain: '<S13>/Gain' */
//    for (ri = 0; ri < 10; ri++) {
//      rtb_Transpose1_0 = 0.0;
//      for (rtb_Transpose_p_tmp = 0; rtb_Transpose_p_tmp < 10;
//           rtb_Transpose_p_tmp++) {
//        rtb_Transpose1_0 += tmp[10 * rtb_Transpose_p_tmp + iy] *
//          rtConstP.pooled2[10 * ri + rtb_Transpose_p_tmp];
//      }

//      rtb_Transpose1[iy + 10 * ri] = rtb_Transpose1_0;
//    }

//    /* End of Gain: '<S13>/Gain' */

//    /* Gain: '<S13>/Gain1' */
//    for (ri = 0; ri < 4; ri++) {
//      rtb_Transpose1_0 = 0.0;
//      for (rtb_Transpose_p_tmp = 0; rtb_Transpose_p_tmp < 10;
//           rtb_Transpose_p_tmp++) {
//        rtb_Transpose1_0 += rtb_Transpose1[10 * rtb_Transpose_p_tmp + iy] *
//          rtConstP.pooled3[10 * ri + rtb_Transpose_p_tmp];
//      }

//      rtb_Integrator2_1[iy + 10 * ri] = rtb_Transpose1_0;
//    }
//  }

//  for (ri = 0; ri < 40; ri++) {
//    /* Gain: '<S15>/gamma_Kx' incorporates:
//     *  Gain: '<S13>/Gain1'
//     */
//    rtDW->gamma_Kx[ri] = -10.0 * rtb_Integrator2_1[ri];
//  }

//  /* Product: '<S11>/Product' incorporates:
//   *  Inport: '<Root>/state'
//   */
//  for (iy = 0; iy < 10; iy++) {
//    for (ri = 0; ri < 10; ri++) {
//      tmp[ri + 10 * iy] = rtU_state[ri] * rtb_Transpose[iy];
//    }
//  }

//  /* End of Product: '<S11>/Product' */
//  for (iy = 0; iy < 10; iy++) {
//    /* Gain: '<S11>/Gain4' */
//    for (ri = 0; ri < 10; ri++) {
//      rtb_Transpose1_0 = 0.0;
//      for (rtb_Transpose_p_tmp = 0; rtb_Transpose_p_tmp < 10;
//           rtb_Transpose_p_tmp++) {
//        rtb_Transpose1_0 += tmp[10 * rtb_Transpose_p_tmp + iy] *
//          rtConstP.pooled2[10 * ri + rtb_Transpose_p_tmp];
//      }

//      rtb_Transpose1[iy + 10 * ri] = rtb_Transpose1_0;
//    }

//    /* End of Gain: '<S11>/Gain4' */
//    for (ri = 0; ri < 4; ri++) {
//      /* Gain: '<S11>/Gain5' */
//      rtb_Transpose1_0 = 0.0;
//      for (rtb_Transpose_p_tmp = 0; rtb_Transpose_p_tmp < 10;
//           rtb_Transpose_p_tmp++) {
//        /* Gain: '<S11>/Gain5' */
//        rtb_Transpose1_0 += rtb_Transpose1[10 * rtb_Transpose_p_tmp + iy] *
//          rtConstP.pooled3[10 * ri + rtb_Transpose_p_tmp];
//      }

//      /* Gain: '<S11>/Gain5' */
//      rtDW->Gain5[iy + 10 * ri] = rtb_Transpose1_0;
//    }
//  }

//  /* Integrator: '<S15>/Integrator' incorporates:
//   *  Constant: '<S15>/Kx0'
//   */
//  for (ri = 0; ri < 40; ri++) {
//    if (rtDW->Integrator_IWORK != 0) {
//      rtX->Integrator_CSTATE[ri] = rtConstB.Kx0[ri];
//    }

//    rtb_Integrator2_0[ri] = rtX->Integrator_CSTATE[ri];
//  }

//  /* End of Integrator: '<S15>/Integrator' */

//  /* Math: '<S13>/Transpose' incorporates:
//   *  Integrator: '<S18>/Integrator'
//   */
//  for (iy = 0; iy < 10; iy++) {
//    ri = iy << 2;
//    rtb_Integrator2_1[ri] = rtb_Integrator2_0[iy];
//    rtb_Integrator2_1[ri + 1] = rtb_Integrator2_0[iy + 10];
//    rtb_Integrator2_1[ri + 2] = rtb_Integrator2_0[iy + 20];
//    rtb_Integrator2_1[ri + 3] = rtb_Integrator2_0[iy + 30];
//  }

//  for (ri = 0; ri < 40; ri++) {
//    rtb_Transpose_f[ri] = rtb_Integrator2_1[ri];

//    /* Integrator: '<S18>/Integrator' incorporates:
//     *  Constant: '<S18>/Kr0'
//     */
//    if (rtDW->Integrator_IWORK_n != 0) {
//      rtX->Integrator_CSTATE_a[ri] = rtConstB.Kr0[ri];
//    }

//    rtb_Integrator2_0[ri] = rtX->Integrator_CSTATE_a[ri];
//  }

//  /* End of Math: '<S13>/Transpose' */

//  /* Math: '<S16>/Transpose' */
//  for (iy = 0; iy < 10; iy++) {
//    ri = iy << 2;
//    rtb_Integrator2_1[ri] = rtb_Integrator2_0[iy];
//    rtb_Integrator2_1[ri + 1] = rtb_Integrator2_0[iy + 10];
//    rtb_Integrator2_1[ri + 2] = rtb_Integrator2_0[iy + 20];
//    rtb_Integrator2_1[ri + 3] = rtb_Integrator2_0[iy + 30];
//  }

//  memcpy(&rtb_Transpose[0], &rtb_Integrator2_1[0], 40U * sizeof(double));

//  /* End of Math: '<S16>/Transpose' */

//  /* Integrator: '<S11>/Integrator2' */
//  memcpy(&rtb_Integrator2_0[0], &rtX->Integrator2_CSTATE[0], 40U * sizeof(double));

//  /* Math: '<S11>/Transpose1' */
//  for (iy = 0; iy < 10; iy++) {
//    ri = iy << 2;
//    rtb_Integrator2_1[ri] = rtb_Integrator2_0[iy];
//    rtb_Integrator2_1[ri + 1] = rtb_Integrator2_0[iy + 10];
//    rtb_Integrator2_1[ri + 2] = rtb_Integrator2_0[iy + 20];
//    rtb_Integrator2_1[ri + 3] = rtb_Integrator2_0[iy + 30];
//  }

//  memcpy(&rtb_Transpose1[0], &rtb_Integrator2_1[0], 40U * sizeof(double));

//  /* End of Math: '<S11>/Transpose1' */

//  /* Product: '<S13>/Product' */
//  memcpy(&rtb_Integrator2_0[0], &rtb_Transpose_f[0], 40U * sizeof(double));

//  /* Product: '<S16>/Matrix Multiply' */
//  memcpy(&rtb_Integrator2_1[0], &rtb_Transpose[0], 40U * sizeof(double));
//  for (iy = 0; iy < 4; iy++) {
//    /* Product: '<S13>/Product' */
//    rtb_Transpose1_0 = 0.0;

//    /* Product: '<S16>/Matrix Multiply' */
//    rtb_Transpose_2 = 0.0;
//    for (ri = 0; ri < 10; ri++) {
//      /* Product: '<S13>/Product' incorporates:
//       *  Inport: '<Root>/state'
//       *  Product: '<S16>/Matrix Multiply'
//       */
//      rtb_Transpose_p_tmp = (ri << 2) + iy;
//      rtb_Transpose1_0 += rtb_Integrator2_0[rtb_Transpose_p_tmp] * rtU_state[ri];

//      /* Product: '<S16>/Matrix Multiply' incorporates:
//       *  Inport: '<Root>/r'
//       */
//      rtb_Transpose_2 += rtb_Integrator2_1[rtb_Transpose_p_tmp] * rtU_r[ri];
//    }

//    /* Product: '<S16>/Matrix Multiply' */
//    rtb_Transpose_1[iy] = rtb_Transpose_2;

//    /* Product: '<S13>/Product' */
//    rtb_Transpose_p[iy] = rtb_Transpose1_0;
//  }

//  /* Product: '<S11>/Matrix Multiply' */
//  memcpy(&rtb_Integrator2_1[0], &rtb_Transpose1[0], 40U * sizeof(double));

//  /* Outport: '<Root>/u' incorporates:
//   *  Inport: '<Root>/state'
//   *  Product: '<S11>/Matrix Multiply'
//   *  Sum: '<S4>/Sum'
//   */
//  for (iy = 0; iy < 4; iy++) {
//    rtb_Transpose1_0 = 0.0;
//    for (ri = 0; ri < 10; ri++) {
//      rtb_Transpose1_0 += rtb_Integrator2_1[(ri << 2) + iy] * rtU_state[ri];
//    }

//    rtY_u[iy] = (rtb_Transpose_p[iy] + rtb_Transpose_1[iy]) - rtb_Transpose1_0;
//  }

//  /* End of Outport: '<Root>/u' */
//  if (rtmIsMajorTimeStep(rtM)) {
//    /* Update for Integrator: '<S15>/Integrator' */
//    rtDW->Integrator_IWORK = 0;

//    /* Update for Integrator: '<S18>/Integrator' */
//    rtDW->Integrator_IWORK_n = 0;
//  }                                    /* end MajorTimeStep */

//  if (rtmIsMajorTimeStep(rtM)) {
//    rt_ertODEUpdateContinuousStates(&rtM->solverInfo, rtM, rtU_r, rtU_state,
//      rtY_u);

//    /* Update absolute time for base rate */
//    /* The "clockTick0" counts the number of times the code of this task has
//     * been executed. The absolute time is the multiplication of "clockTick0"
//     * and "Timing.stepSize0". Size of "clockTick0" ensures timer will not
//     * overflow during the application lifespan selected.
//     */
//    ++rtM->Timing.clockTick0;
//    rtM->Timing.t[0] = rtsiGetSolverStopTime(&rtM->solverInfo);

//    {
//      /* Update absolute timer for sample time: [0.002s, 0.0s] */
//      /* The "clockTick1" counts the number of times the code of this task has
//       * been executed. The resolution of this integer timer is 0.002, which is the step size
//       * of the task. Size of "clockTick1" ensures timer will not overflow during the
//       * application lifespan selected.
//       */
//      rtM->Timing.clockTick1++;
//    }
//  }                                    /* end MajorTimeStep */
//}

///* Derivatives for root system: '<Root>' */
//void mrac_control_derivatives(RT_MODEL *const rtM, double rtU_r[10])
//{
//  DW *rtDW = rtM->dwork;
//  X *rtX = rtM->contStates;
//  XDot *_rtXdot;
//  int32_t i;
//  uint32_t ri;
//  _rtXdot = ((XDot *) rtM->derivs);

//  /* Derivatives for StateSpace: '<S4>/Reference System' incorporates:
//   *  Inport: '<Root>/r'
//   */
//  memset(&_rtXdot->ReferenceSystem_CSTATE[0], 0, 10U * sizeof(double));
//  for (i = 0; i < 10; i++) {
//    for (ri = rtConstP.ReferenceSystem_A_jc[(uint32_t)i]; ri <
//         rtConstP.ReferenceSystem_A_jc[(uint32_t)i + 1U]; ri++) {
//      _rtXdot->ReferenceSystem_CSTATE[rtConstP.ReferenceSystem_A_ir[ri]] +=
//        rtConstP.ReferenceSystem_A_pr[ri] * rtX->ReferenceSystem_CSTATE
//        [(uint32_t)i];
//    }
//  }

//  for (i = 0; i < 10; i++) {
//    for (ri = rtConstP.ReferenceSystem_B_jc[(uint32_t)i]; ri <
//         rtConstP.ReferenceSystem_B_jc[(uint32_t)i + 1U]; ri++) {
//      _rtXdot->ReferenceSystem_CSTATE[rtConstP.ReferenceSystem_B_ir[ri]] +=
//        rtConstP.ReferenceSystem_B_pr[ri] * rtU_r[(uint32_t)i];
//    }
//  }

//  /* End of Derivatives for StateSpace: '<S4>/Reference System' */

//  /* Derivatives for Integrator: '<S15>/Integrator' incorporates:
//   *  Gain: '<S15>/gamma_Kx'
//   */
//  memcpy(&_rtXdot->Integrator_CSTATE[0], &rtDW->gamma_Kx[0], 40U * sizeof(double));

//  /* Derivatives for Integrator: '<S18>/Integrator' incorporates:
//   *  Gain: '<S18>/gamma_Kr'
//   */
//  memcpy(&_rtXdot->Integrator_CSTATE_a[0], &rtDW->gamma_Kr[0], 40U * sizeof
//         (double));

//  /* Derivatives for Integrator: '<S11>/Integrator2' incorporates:
//   *  Gain: '<S11>/Gain5'
//   */
//  memcpy(&_rtXdot->Integrator2_CSTATE[0], &rtDW->Gain5[0], 40U * sizeof(double));
//}

///* Model initialize function */
//void mrac_control_initialize(RT_MODEL *const rtM)
//{
//  DW *rtDW = rtM->dwork;
//  X *rtX = rtM->contStates;
//  XDis *rtXDis = ((XDis *) rtM->contStateDisabled);

//  /* Registration code */
//  {
//    /* Setup solver object */
//    rtsiSetSimTimeStepPtr(&rtM->solverInfo, &rtM->Timing.simTimeStep);
//    rtsiSetTPtr(&rtM->solverInfo, &rtmGetTPtr(rtM));
//    rtsiSetStepSizePtr(&rtM->solverInfo, &rtM->Timing.stepSize0);
//    rtsiSetdXPtr(&rtM->solverInfo, &rtM->derivs);
//    rtsiSetContStatesPtr(&rtM->solverInfo, (double **) &rtM->contStates);
//    rtsiSetNumContStatesPtr(&rtM->solverInfo, &rtM->Sizes.numContStates);
//    rtsiSetNumPeriodicContStatesPtr(&rtM->solverInfo,
//      &rtM->Sizes.numPeriodicContStates);
//    rtsiSetPeriodicContStateIndicesPtr(&rtM->solverInfo,
//      &rtM->periodicContStateIndices);
//    rtsiSetPeriodicContStateRangesPtr(&rtM->solverInfo,
//      &rtM->periodicContStateRanges);
//    rtsiSetContStateDisabledPtr(&rtM->solverInfo, (bool**)
//      &rtM->contStateDisabled);
//    rtsiSetErrorStatusPtr(&rtM->solverInfo, (&rtmGetErrorStatus(rtM)));
//    rtsiSetRTModelPtr(&rtM->solverInfo, rtM);
//  }

//  rtsiSetSimTimeStep(&rtM->solverInfo, MAJOR_TIME_STEP);
//  rtM->intgData.y = rtM->odeY;
//  rtM->intgData.f[0] = rtM->odeF[0];
//  rtM->intgData.f[1] = rtM->odeF[1];
//  rtM->intgData.f[2] = rtM->odeF[2];
//  rtM->contStates = ((X *) rtX);
//  rtM->contStateDisabled = ((XDis *) rtXDis);
//  rtM->Timing.tStart = (0.0);
//  rtsiSetSolverData(&rtM->solverInfo, (void *)&rtM->intgData);
//  rtsiSetIsMinorTimeStepWithModeChange(&rtM->solverInfo, false);
//  rtsiSetSolverName(&rtM->solverInfo,"ode3");
//  rtmSetTPtr(rtM, &rtM->Timing.tArray[0]);
//  rtM->Timing.stepSize0 = 0.002;
//  rtmSetFirstInitCond(rtM, 1);

//  /* InitializeConditions for StateSpace: '<S4>/Reference System' */
//  memset(&rtX->ReferenceSystem_CSTATE[0], 0, 10U * sizeof(double));

//  /* InitializeConditions for Integrator: '<S15>/Integrator' incorporates:
//   *  Integrator: '<S18>/Integrator'
//   */
//  if (rtmIsFirstInitCond(rtM)) {
//    rtX->Integrator_CSTATE[0] = 10.448;
//    rtX->Integrator_CSTATE[1] = 11.6317;
//    rtX->Integrator_CSTATE[2] = 9.8523;
//    rtX->Integrator_CSTATE[3] = 2.5558;
//    rtX->Integrator_CSTATE[4] = 23.2728;
//    rtX->Integrator_CSTATE[5] = 3.8747;
//    rtX->Integrator_CSTATE[6] = 12.585;
//    rtX->Integrator_CSTATE[7] = 1.9949;
//    rtX->Integrator_CSTATE[8] = 13.5652;
//    rtX->Integrator_CSTATE[9] = 1.2265;
//    rtX->Integrator_CSTATE[10] = 10.448;
//    rtX->Integrator_CSTATE[11] = 11.6317;
//    rtX->Integrator_CSTATE[12] = -9.8523;
//    rtX->Integrator_CSTATE[13] = -2.5558;
//    rtX->Integrator_CSTATE[14] = 12.585;
//    rtX->Integrator_CSTATE[15] = 1.9949;
//    rtX->Integrator_CSTATE[16] = 23.2728;
//    rtX->Integrator_CSTATE[17] = 3.8747;
//    rtX->Integrator_CSTATE[18] = 13.5652;
//    rtX->Integrator_CSTATE[19] = 1.2265;
//    rtX->Integrator_CSTATE[20] = -4.7175;
//    rtX->Integrator_CSTATE[21] = -5.5938;
//    rtX->Integrator_CSTATE[22] = 13.6462;
//    rtX->Integrator_CSTATE[23] = 3.5503;
//    rtX->Integrator_CSTATE[24] = -42.4542;
//    rtX->Integrator_CSTATE[25] = -6.6003;
//    rtX->Integrator_CSTATE[26] = 17.6211;
//    rtX->Integrator_CSTATE[27] = 2.4865;
//    rtX->Integrator_CSTATE[28] = 91.7806;
//    rtX->Integrator_CSTATE[29] = 6.9419;
//    rtX->Integrator_CSTATE[30] = -4.7175;
//    rtX->Integrator_CSTATE[31] = -5.5938;
//    rtX->Integrator_CSTATE[32] = -13.6462;
//    rtX->Integrator_CSTATE[33] = -3.5503;
//    rtX->Integrator_CSTATE[34] = 17.6211;
//    rtX->Integrator_CSTATE[35] = 2.4865;
//    rtX->Integrator_CSTATE[36] = -42.4542;
//    rtX->Integrator_CSTATE[37] = -6.6003;
//    rtX->Integrator_CSTATE[38] = 91.7806;
//    rtX->Integrator_CSTATE[39] = 6.9419;
//    rtX->Integrator_CSTATE_a[0] = -10.448;
//    rtX->Integrator_CSTATE_a[1] = -11.6317;
//    rtX->Integrator_CSTATE_a[2] = -9.8523;
//    rtX->Integrator_CSTATE_a[3] = -2.5558;
//    rtX->Integrator_CSTATE_a[4] = -23.2728;
//    rtX->Integrator_CSTATE_a[5] = -3.8747;
//    rtX->Integrator_CSTATE_a[6] = -12.585;
//    rtX->Integrator_CSTATE_a[7] = -1.9949;
//    rtX->Integrator_CSTATE_a[8] = -13.5652;
//    rtX->Integrator_CSTATE_a[9] = -1.2265;
//    rtX->Integrator_CSTATE_a[10] = -10.448;
//    rtX->Integrator_CSTATE_a[11] = -11.6317;
//    rtX->Integrator_CSTATE_a[12] = 9.8523;
//    rtX->Integrator_CSTATE_a[13] = 2.5558;
//    rtX->Integrator_CSTATE_a[14] = -12.585;
//    rtX->Integrator_CSTATE_a[15] = -1.9949;
//    rtX->Integrator_CSTATE_a[16] = -23.2728;
//    rtX->Integrator_CSTATE_a[17] = -3.8747;
//    rtX->Integrator_CSTATE_a[18] = -13.5652;
//    rtX->Integrator_CSTATE_a[19] = -1.2265;
//    rtX->Integrator_CSTATE_a[20] = 4.7175;
//    rtX->Integrator_CSTATE_a[21] = 5.5938;
//    rtX->Integrator_CSTATE_a[22] = -13.6462;
//    rtX->Integrator_CSTATE_a[23] = -3.5503;
//    rtX->Integrator_CSTATE_a[24] = 42.4542;
//    rtX->Integrator_CSTATE_a[25] = 6.6003;
//    rtX->Integrator_CSTATE_a[26] = -17.6211;
//    rtX->Integrator_CSTATE_a[27] = -2.4865;
//    rtX->Integrator_CSTATE_a[28] = -91.7806;
//    rtX->Integrator_CSTATE_a[29] = -6.9419;
//    rtX->Integrator_CSTATE_a[30] = 4.7175;
//    rtX->Integrator_CSTATE_a[31] = 5.5938;
//    rtX->Integrator_CSTATE_a[32] = 13.6462;
//    rtX->Integrator_CSTATE_a[33] = 3.5503;
//    rtX->Integrator_CSTATE_a[34] = -17.6211;
//    rtX->Integrator_CSTATE_a[35] = -2.4865;
//    rtX->Integrator_CSTATE_a[36] = 42.4542;
//    rtX->Integrator_CSTATE_a[37] = 6.6003;
//    rtX->Integrator_CSTATE_a[38] = -91.7806;
//    rtX->Integrator_CSTATE_a[39] = -6.9419;
//  }

//  rtDW->Integrator_IWORK = 1;

//  /* End of InitializeConditions for Integrator: '<S15>/Integrator' */

//  /* InitializeConditions for Integrator: '<S18>/Integrator' */
//  rtDW->Integrator_IWORK_n = 1;

//  /* InitializeConditions for Integrator: '<S11>/Integrator2' */
//  memset(&rtX->Integrator2_CSTATE[0], 0, 40U * sizeof(double));

//  /* set "at time zero" to false */
//  if (rtmIsFirstInitCond(rtM)) {
//    rtmSetFirstInitCond(rtM, 0);
//  }
//}

///*
// * File trailer for generated code.
// *
// * [EOF]
// */
