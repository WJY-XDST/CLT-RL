#ifndef __ADDITIONAL_CON_H_
#define __ADDITIONAL_CON_H_

#include "vmc.h"
#include "wheel.h"
#include "WBR.h"
#include "arm_math.h"
#include "matrix.h"


typedef struct
{
  float X_P1;
  float X_P2;
  float X_P3;
  float phi_P1;
  float phi_P2;
  float phi_P3;

}A_k;
//typedef struct
//{
//  float X_P1;
//  float X_P2;
//  float X_P3;
//  float X_P4;
//  float thetal_P1;
//  float thetal_P2;
//  float thetal_P3;
//  float thetal_P4;
//  float thetar_P1;
//  float thetar_P2;
//  float thetar_P3;
//  float thetar_P4;
//}B_k;
typedef struct
{
  float Sdot_perior;
  float phidot_perior;
  float xldot_perior;
  float xrdot_perior;
}perior;


int8_t ground_dection(VMC_joint * vmcl,VMC_joint * vmcr);
void w_perior(perior* Perior,float *A_K,float *B_K);
void jump_control(VMC_joint* vmcl,VMC_joint* vmcr);
void gleiten_dection(VMC_joint * vmcl,VMC_joint * vmcr);
void Wheel_predictor_com(wheel* Wheel,perior* Perior);
void auto_jump();
void auto_jump_double();

void collide_check();

//void sideway_control();



extern int8_t gleiten_flag;
extern int8_t groundoff_flag;
extern int8_t groundoff_flag_R;
extern int8_t groundoff_flag_L;

extern int8_t length_change_flag;
extern perior PErior;
extern uint8_t jump_flag;//跳跃标志位
extern uint8_t jump_flag_com;
extern int16_t jump_count;
extern int16_t gleiten_count2;
extern int8_t landing_flag;
extern float Fnl;
extern float Fnr;
extern uint8_t  nextjump_flag;
extern uint8_t  autojump_flag;
extern uint8_t  autojump_double_flag;//可以自动跳标志
extern uint8_t collide_flag;
extern int8_t spin_move_flag;
extern float spin_add;
extern int8_t left_front ;
extern int8_t left_back ;
extern int8_t right_front ;
extern int8_t right_back ;
extern uint8_t  nextjump_double_flag;//下一次跳标志位，目的是为了跳完一次后防止继续跳，需手动重启
extern int16_t  auto_jump_double_count;
extern int8_t tick_flag;
extern int16_t jump_landing_flag;
extern int gostep_flag;
extern float PL,PR;

void A1_motor_check();//检查A1是否进保护
void change_leg_gostep();

#endif


