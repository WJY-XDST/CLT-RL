#ifndef __ONLINE_CHECK_TASK_H
#define __ONLINE_CHECK_TASK_H
#include "all_init.h"

void online_check_task(void *param);
extern volatile uint8_t leg_reset_flag;//电机上电后需要复位标志位
extern uint8_t leg_elevtri_flag;
extern int8_t wheel_left_online ;
extern int8_t wheel_right_online ;
extern int wheel_left;
extern int wheel_right;
extern int left_leg_cnt;
extern int right_leg_cnt;
extern int leg_online;
#endif
