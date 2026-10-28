#ifndef __OBSERVE_TASK_H
#define __OBSERVE_TASK_H


#include "stdint.h"
#include "kalman_filter.h"

extern void Observe_task(void *param);
extern void xvEstimateKF_Init(KalmanFilter_t *EstimateKF);
extern void xvEstimateKF_Update(KalmanFilter_t *EstimateKF ,float acc,float vel);
extern float Real_real_x ;
extern float Real_real_xdot ;
#endif




