#ifndef __ALL_INIT_H
#define __ALL_INIT_H

#include "motor.h"
#include "pid.h"

void All_init(void);

void start_task_freertos(void *params);

extern Motor motor3508;
extern BasePID_Object pid_roll_compen;
extern BasePID_Object pid_chassis_follow;//底盘跟随pid
extern BasePID_Object pid_chassis_move;
extern BasePID_Object pid_legleft,pid_legright;//腿长pid
#define AtR 0.0174532f
#define RtD 57.2957795f
#define gyro_get 1
#define control_s 0//1里程控制，0速度控制
#endif
