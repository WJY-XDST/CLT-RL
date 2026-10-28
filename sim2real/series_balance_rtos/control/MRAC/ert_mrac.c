///*
// * File: ert_main.c
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

//#include <stddef.h>
//#include <stdio.h>            /* This example main program uses printf/fflush */
//#include "mrac_control.h"              /* Model header file */

//#include "WBR.h" 
//#include "freertos.h"
//#include "FreeRTOS.h"
//#include "task.h"             /*user*/

//static RT_MODEL rtM_;
//static RT_MODEL *const rtMPtr = &rtM_; /* Real-time model */
//static DW rtDW;                        /* Observable states */
//static X rtX;                          /* Observable continuous states */
//static XDis rtXDis;                    /* Continuous states Disabled */

///* '<Root>/r' */
// double rtU_r[10];

///* '<Root>/state' */
// double rtU_state[10];

///* '<Root>/u' */
// double rtY_u[4];
//float T_out_test[4];
///*
// * Associating rt_OneStep with a real-time clock or interrupt service routine
// * is what makes the generated code "real-time".  The function rt_OneStep is
// * always associated with the base rate of the model.  Subrates are managed
// * by the base rate from inside the generated code.  Enabling/disabling
// * interrupts and floating point context switches are target specific.  This
// * example code indicates where these should take place relative to executing
// * the generated code step function.  Overrun behavior should be tailored to
// * your application needs.  This example simply sets an error status in the
// * real-time model and returns from rt_OneStep.
// */
//void rt_OneStep(RT_MODEL *const rtM);
//void rt_get(float* state,struct Model *model)
//{
//	for(int i=0;i<10;i++)
//	rtU_state[i]=state[i];
//	for(int i=0;i<10;i++)
//	rtU_r[0]=model->s;
//	rtU_r[1]=model->s_dot;
//	rtU_r[2]=model->yaw;
//	rtU_r[3]=model->yaw_dot;
//	rtU_r[4]=0;
//	rtU_r[5]=0;
//	rtU_r[6]=0;
//	rtU_r[7]=0;
//	rtU_r[8]=0;
//	rtU_r[9]=0;
//	
//}
//void rt_OneStep(RT_MODEL *const rtM)
//{
//  static bool OverrunFlag = false;

//  /* Disable interrupts here */

//  /* Check for overrun */
//  if (OverrunFlag) {
//    rtmSetErrorStatus(rtM, "Overrun");
//    return;
//  }

//  OverrunFlag = true;

//  /* Save FPU context here (if necessary) */
//  /* Re-enable timer or interrupt here */
//  /* Set model inputs here */

//  /* Step the model */
//  mrac_control_step(rtM, rtU_r, rtU_state, rtY_u);

//  /* Get model outputs here */
//  for(int i=0;i<4;i++)
//  T_out_test[i]=rtY_u[i];
//  /* Indicate task complete */
//  OverrunFlag = false;

//  /* Disable interrupts here */
//  /* Restore FPU context here (if necessary) */
//  /* Enable interrupts here */
//}

///*
// * The example main function illustrates what is required by your
// * application code to initialize, execute, and terminate the generated code.
// * Attaching rt_OneStep to a real-time clock is target specific. This example
// * illustrates how you do this relative to initializing the model.
// */
//int mrac_count;
//UBaseType_t uxHighWaterMark_mrac;
//int mrac_task(int argc, const char *argv[])
//{
//  RT_MODEL *const rtM = rtMPtr;

//  /* Unused arguments */
//  (void)(argc);
//  (void)(argv);

//  /* Pack model data into RTM */
//  rtM->dwork = &rtDW;
//  rtM->contStates = &rtX;
//  rtM->contStateDisabled = &rtXDis;

//  /* Initialize model */
//  mrac_control_initialize(rtM);
////	TickType_t xLastWakeTime;
////	xLastWakeTime = xTaskGetTickCount();
//	mrac_count=0;
//	
//  /* Simulating the model step behavior (in non real-time) to
//   *  simulate model behavior at stop time.
//   */
//  while ((rtmGetErrorStatus(rtM) == (NULL)) && !rtmGetStopRequested(rtM)) {
//	  mrac_count++;
//	  rt_get(state,&model_target);
//	  rt_OneStep(rtM);
//	  uxHighWaterMark_mrac = uxTaskGetStackHighWaterMark(NULL );
//	  vTaskDelay(2);
//  }
//  while (1) {
//	  mrac_count++;

//	  vTaskDelay(2);
//  }
//  vTaskDelete(NULL);

//  return 0;
//}

///*
// * File trailer for generated code.
// *
// * [EOF]
// */
