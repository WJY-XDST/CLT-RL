#include "wheel.h" 
#ifndef MOTOR_TEST_H
#define MOTOR_TEST_H

#ifdef __cplusplus
extern "C" {
#endif
 
	
void power_control(wheel *Wheel);
extern float sum_power;
extern float test_out[2];
	
#ifdef __cplusplus
}
#endif
 
#endif
