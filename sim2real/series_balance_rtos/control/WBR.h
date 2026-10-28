#ifndef __WBR_H_
#define __WBR_H_

#include "arm_math.h"


struct  Model
{
    float s;    //里程
    float s_dot;//速度
	float yaw;//yaw角度
	float yaw_dot;//yaw角速度
	float thetal;//左腿角度
	float thetal_dot;//左腿角速度
	float thetar;//右腿角度
	float thetar_dot;//右腿角速度
	float phi;//机体pitch
	float phi_dot;//机体pitch角速度
};


extern struct Model model;

float LQR_K_calc(float *coe,float len);

void model_state();
void model_matrix_init();

void state_control();

extern arm_matrix_instance_f32 T_out;  //输出矩阵
extern int8_t length_flag;
extern int8_t follow_perior;
extern float distance_target;
extern float follow_target;//底盘跟随目标角度
extern float roll_com;
extern float pitch_com;
extern float pit_com;
extern float follow_target_com;
extern float speed_target;//存储速度
extern float follow_add;//底盘跟随pid做累加
extern float32_t state[10]__attribute__((at(0x24004000)));
extern float s_set;

extern int8_t length_flag;
extern int8_t spin_flag;
extern int8_t spin_enable;
extern int8_t lqr_flag;
extern int lqr_ready_flag;
extern int lqr_reset_flag;
extern int leg_left_ready;
extern int leg_right_ready;
extern int8_t fire_flag;
extern struct Model model_target;
extern int pitch_ready;
extern int reset_to_lqr_flag;
extern int all_ready_flag;
#endif