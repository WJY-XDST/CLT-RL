#ifndef __TORQUE__H
#define __TORQUE__H

#include "wheel.h"
#include "vmc.h"

#define REDUCTION_RATE 0.052631f                         //<1：19 3508减速比
//#define REDUCTION_RATE 0.07479f                         //<1：19 3508减速比
#define CHASSIS_MOTOR_CURRENT_TO_TORQUE_SEN  0.00030059//<(N.m/A）输出力矩系数3508  0.0003662f
#define CHASSIS_MOTOR_CURRENT_TO_TORQUE_DM 1428.57f //<(N.m/A）输出力矩系数达妙
#define reduction_torque_ratio 0.110f //<(N.m/A）输出力矩系数
#define acc_g  9.81


void torque_motor3508(wheel* Wheel,float torquel,float torquer);
void torque_motorDM(VMC_joint* vmcl,VMC_joint* vmcr);
void Length_control(VMC_joint* vmc);
void F0_control(VMC_joint* vmc);
void turn_control();
void torque_MIT_DM(VMC_joint* vmcl,VMC_joint* vmcr);
void torque_LK(VMC_joint* vmcl,VMC_joint* vmcr);

extern int leg_flag;
extern float length_target;
extern float roll_compen;//roll补偿


extern int8_t dance_flag;
extern float spring_feedforwardl;//气弹簧前馈
extern float spring_feedforwardr;

#endif