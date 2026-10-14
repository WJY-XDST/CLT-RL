#ifndef __VMC_H_
#define __VMC_H_


#include "arm_math.h"
#include "pid.h"
typedef struct
{
  float u0;
  float u2;
  float u3;
  float theta0;
  float u0_last;
  float u0_dot;
  float theta0_speed;
  float L;
  float L_last;
  float L_dot;

  float theta1;//前
  float theta2;//后
  float link_length[5];
  /*VMC计算变量*/
  float torque_f;//front
  float torque_b;//back
  float testf;
  float testb;
  /*逆解电机力矩*/
  BasePID_Object length_PID;
  float F0;//前馈力矩
  float TP;//关节转向力矩
  float TW;//驱动轮力矩
  float T_balance;//驱动轮力矩保持姿态分量
  float T_s;//驱动轮力矩速度里程分量
  float T_yaw;//驱动轮力矩yaw分量
  float T_turn;//转向力矩
  float Fn;//地面支持力
//  float Fy;
  float feed_torque;//前馈力矩
  float length_pid_out;//PID控制力
  float F_intertial;//侧向惯性控制力
  float roll_flag;
  float leng_target;
  float reset_speed;//复位速度
  int   slow_flag;//缓启动标志位
  float reset_speed_kd;//复位速度的kd

	float v1;
	float v2;
}VMC_joint;

extern VMC_joint VMC_joint_left;//左五连杆
extern VMC_joint VMC_joint_right;//右五连杆
	

void link_length_init(VMC_joint *vmc_joint_left,VMC_joint *vmc_joint_right);//腿部连杆长度初始化
void get_l_theta(VMC_joint* vmc);//计算五连杆参数
void VMC_Count(VMC_joint* vmc1,VMC_joint* vmc2);//VMC
void F_Calc(VMC_joint* vmc);
void F_Calc_xy(VMC_joint* vmc);//以xy方向解算
void mySaturate(float *in,float min,float max);
extern int slow_flag;





#endif
