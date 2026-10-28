#include "vmc.h"
#include "wheel.h"
#include "WBR.h"
#include "arm_math.h"
#include "matrix.h"
#include "wheel.h"
#include "et08.h"
#include "HI04.h"
#include "all_init.h"
#include "ins.h"
#include "additional_con.h"
#include "filter.h"
#include "INS_task.h"
#include "power_con.h"
#include "online_check_task.h"
#include "observe_task.h"

extern float Real_real_xdot;
extern float Real_real_x;
extern float Pitch_Angle_raw;
/*Yaw灵敏度*/
float YawSentivity=0.00001;
/*里程控制*/
float distance_target=0;//存储距离
float target_add=0.00;//距离累加步长
/*速度控制*/
float speed_target=0;//存储速度
int sdot_count = 0;
float pitch_com=2.5;//pitch角补偿
int8_t length_flag=0;//腿长标志
/*两个遥感回拨标志*/
int8_t call_flag=0;
int8_t call_flag1=0;
int8_t call_flag_keyboard=0;//键盘回0标志
/*各种控制标志位*/
int8_t lqr_flag=0;//lqr计算标志（由云台传过来）
int8_t turn_flag=0;//转向标志
int8_t spin_enable_count=0;
int8_t spin_enable=0;//自旋使能位，停下后才能自旋
int8_t spin_flag=0;//自旋标志位 
int8_t fire_flag=0;//遥控开火标志位
int16_t leg_reset_B_count=0;//遥控开火标志位
/*运动中间变量*/
float keyspeed_sense=5;//键盘控制运动速度
float follow_add=0;//底盘跟随pid做累加
float follow_target=90;//底盘跟随目标角度，侧身没用了
float follow_target_com = 90;
int8_t follow_perior;//转向优先标志，太快就让底盘停下
//int16_t follow_perior_count;//加入计数判断
float TP_sensel=0;//削减TP输出用
float TP_senser=0;
float TW_sensel=0;//削减TW输出用
float TW_senser=0;
/*矩阵运算使用*/
arm_status status;//这是什么，DSP库的东西，好像没用
struct Model model_target;//目标状态
float32_t state[10]__attribute__((at(0x24004000)));
arm_matrix_instance_f32 State;//输入状态矩阵
arm_matrix_instance_f32 K0,K1,K2,K3,K4,KL,KR;  //K矩阵
arm_matrix_instance_f32 T_out,T_outL,T_outR;  //输出矩阵
float32_t t_out[4];
float test_check[4];

float s_set=0.7;//重心偏置补偿
int lqr_ready_flag=0;
int lqr_cnt=0;
int lqr_reset_flag=0;
int leg_left_ready=0;
int leg_right_ready=0;
int pitch_ready=0;
int reset_to_lqr_cnt=0;
int reset_to_lqr_flag=0;
int pitch_ready_cnt=0;
int all_ready_flag = 0;

/*0档，一阶倒立摆*/
const float32_t lqr_k0[40]={ 
-5.4023, -10.2302, -5.5227, -1.1688, -38.6180, -5.3582, -12.8482, -1.7352, -2.4945, -1.9508,
-5.4023, -10.2302, 5.5227, 1.1688, -12.8482, -1.7352, -38.6180, -5.3582, -2.4945, -1.9508,
2.8804, 5.2287, -6.2449, -1.3591, 38.9838, 5.0594, -15.1011, -1.6218, -9.3571, -6.5845,
2.8804, 5.2287, 6.2449, 1.3591, -15.1011, -1.6218, 38.9838, 5.0594, -9.3571, -6.5845

//-9.961, -13.61, -15.41, -2.385, -28.94, -3.375, -11.85, -1.625,  -15.5, -1.309,
//-9.961, -13.61,  15.41,  2.385, -11.85, -1.625, -28.94, -3.375,  -15.5, -1.309,
// 1.242,  1.792, -13.86, -2.179,   28.7,  2.716, -14.49, -1.268, -66.68, -3.894,
// 1.242,  1.792,  13.86,  2.179, -14.49, -1.268,   28.7,  2.716, -66.68, -3.894


};
//Hinf K
const float32_t lqr_k1[40]={
-5.5496, -10.2526, -5.5755, -1.1756, -39.7395, -5.6783, -14.1785, -2.0921, -1.9502, -1.5332,
-5.5496, -10.2526, 5.5755, 1.1756, -14.1785, -2.0921, -39.7395, -5.6783, -1.9502, -1.5332,
2.2519, 3.9822, -6.1504, -1.3568, 38.1268, 4.8979, -19.2240, -2.0568, -9.6121, -6.7962,
2.2519, 3.9822, 6.1504, 1.3568, -19.2240, -2.0568, 38.1268, 4.8979, -9.6121, -6.7962


};


const float32_t lqr_k2[40]={
-5.6307, -10.2523, -5.6626, -1.1887, -40.4063, -5.9591, -15.2950, -2.4406, -1.5632, -1.2366,
-5.6307, -10.2523, 5.6626, 1.1887, -15.2950, -2.4406, -40.4063, -5.9591, -1.5632, -1.2366,
1.8050, 3.1381, -5.9892, -1.3425, 38.1485, 4.8660, -22.8431, -2.4630, -9.7526, -6.9128,
1.8050, 3.1381, 5.9892, 1.3425, -22.8431, -2.4630, 38.1485, 4.8660, -9.7526, -6.9128


};
const float32_t lqr_k3[40]={
	
//-10.7895 , -10.8754  , -9.7198  , -1.9555  ,-23.9071  , -3.6763 , -11.2706 ,  -1.9634 , -12.6263  , -1.0105,
//  -10.7895  ,-10.8754 ,   9.7198 ,   1.9555 , -11.2706  , -1.9634 , -23.9071 ,  -3.6763 , -12.6263 ,  -1.0105,
//    8.2877  ,  8.3426 , -29.0552  , -6.3397 ,  40.1887  ,  4.7657 , -14.1504 ,  -0.3933 ,-100.8357,   -5.1857,
//    8.2877  ,  8.3426 ,  29.0552  ,  6.3397 , -14.1504  , -0.3933 ,  40.1887 ,   4.7657, -100.8357 , -5.1857
-9.992, -13.21, -5.035, -2.318, -29.49, -3.678, -13.83, -2.058, -10.87, -0.9833,
-9.992, -13.21,  5.035,  2.318, -13.83, -2.058, -29.49, -3.678, -10.87, -0.9833,
 0.566, 0.8402, -3.995, -1.861,  28.11,   1.63,  -18.3, -1.673,  -38.3,  -4.016,
 0.566, 0.8402,  3.995,  1.861,  -18.3, -1.673,  28.11,   1.63,  -38.3,  -4.016
};

const float32_t lqr_k4[40]={
	
-9.992, -13.21, -5.035, -2.318, -29.49, -3.678, -13.83, -2.058, -10.87, -0.9833,
-9.992, -13.21,  5.035,  2.318, -13.83, -2.058, -29.49, -3.678, -10.87, -0.9833,
 0.566, 0.8402, -3.995, -1.861,  28.11,   1.63,  -18.3, -1.673,  -38.3,  -4.016,
 0.566, 0.8402,  3.995,  1.861,  -18.3, -1.673,  28.11,   1.63,  -38.3,  -4.016
};

float32_t LQR_KL[40];
float32_t LQR_KR[40];

//三次多项式拟合系数
float32_t Poly_Coefficient[40][4]=
{
{-12.59,  11.99, -3.671,  -9.636},
{93.06, -95.06,  33.22,  -16.79},
{ 14.64, -6.595, -2.438,  -4.399},
{ 6.604, -3.023, -1.061,  -2.037},
{-259.7,  224.0, -65.63,  -23.26},
{-20.82,  14.66,  -9.32,  -2.235},
{-67.08,  89.01, -66.36,  -3.592},
{-2.475,  2.303, -9.359, -0.2591},
{ 629.1, -690.5,  278.1,  -43.89},
{ 44.66, -48.55,  19.53,  -3.304},
{-12.59,  11.99, -3.671,  -9.636},
{ 93.06, -95.06,  33.22,  -16.79},
{-14.64,  6.595,  2.438,   4.399},
{-6.604,  3.023,  1.061,   2.037},
{-67.08,  89.01, -66.36,  -3.592},
{-2.475,  2.303, -9.359, -0.2591},
{-259.7,  224.0, -65.63,  -23.26},
{-20.82,  14.66,  -9.32,  -2.235},
{ 629.1, -690.5,  278.1,  -43.89},
{ 44.66, -48.55,  19.53,  -3.304},
{-39.25,  57.44,  -29.54,   4.498},
{-100.6,  119.3,  -51.25,   7.133},
{ -50.8,  35.62,  0.3767,   -5.09},
{-22.63,  15.89,  0.1466,  -2.345},
{-303.7,  314.0,  -93.69,   36.71},
{-25.45,  31.79,   -10.5,   3.661},
{-188.0,  247.6,  -146.3,   2.581},
{-11.99,  13.13,   -11.6,  0.2187},
{-567.3,  540.2,  -172.7,  -50.86},
{-37.37,  36.12,  -11.85,  -2.793},
{-39.25,  57.44,  -29.54,   4.498},
{-100.6,  119.3,  -51.25,   7.133},
{  50.8, -35.62, -0.3767,    5.09},
{ 22.63, -15.89, -0.1466,   2.345},
{-188.0,  247.6,  -146.3,   2.581},
{-11.99,  13.13,   -11.6,  0.2187},
{-303.7,  314.0,  -93.69,   36.71},
{-25.45,  31.79,   -10.5,   3.661},
{-567.3,  540.2,  -172.7,  -50.86},
{-37.37,  36.12,  -11.85,  -2.793}
};

float32_t LQR_K_calc(float *coe,float len)
{
   
  return coe[0]*len*len*len+coe[1]*len*len+coe[2]*len+coe[3];
}

/*变量矩阵及目标初始化*/
void model_matrix_init()
{
    model_target.s=s_set;
    model_target.s_dot=0;
    model_target.yaw=0;
	model_target.yaw_dot=0;
	model_target.thetal=0;
	model_target.thetal_dot=0;
	model_target.thetar=0;
	model_target.thetar_dot=0;
	model_target.phi=0;
    model_target.phi_dot=0;
	
  matrix_init(&K0,4,10,(float32_t *)lqr_k0);//K矩阵   4行10列
  matrix_init(&K1,4,10,(float32_t *)lqr_k1);//K矩阵   4行10列
  matrix_init(&K2,4,10,(float32_t *)lqr_k2);//K矩阵   4行10列
  matrix_init(&K3,4,10,(float32_t *)lqr_k3);//K矩阵   4行10列
  matrix_init(&K4,4,10,(float32_t *)lqr_k4);//K矩阵   4行10列
  matrix_init(&KL,4,10,(float32_t *)LQR_KL);//K矩阵   4行10列
  matrix_init(&KR,4,10,(float32_t *)LQR_KR);//K矩阵   4行10列

//  matrix_init(&K4,4,10,(float32_t *)lqr_k4);//K矩阵   4行10列
  matrix_init(&State,10,1,(float32_t *)state);//状态变量
  matrix_init(&T_out,4,1,t_out);//输出变量

}


/*状态矩阵求解*/
void model_state()
{
	//状态矩阵更新
	//matrix_init(&State,10,1,state);//状态变量
	if((groundoff_flag==1)||jump_flag >= 1||pitch_ready==0||lqr_reset_flag==1)
	{
		state[0]=0;
		state[1]=0;
	}
	else{
		
	state[0]=-(model_target.s - Real_real_x);
  state[1]=-(model_target.s_dot - Real_real_xdot);
//	  state[0]=-(model_target.s-Wheel.S);
//    state[1]=-(model_target.s_dot-Wheel.S_dot);
//		state[0]=0;
//		state[1]=0;
	}
	/*使用bmi088，EKF*/

    if(pitch_ready==0||groundoff_flag==1||lqr_reset_flag==1)
    {
		state[2]=0;
		state[3]=0;
    }
    else
    {
		state[2]=-(model_target.yaw-INS_attitude->yaw_total_angle*AtR);
		state[3]=-(model_target.yaw_dot-INS_attitude->yaw_speed);
    }


		
    if(lqr_reset_flag == 1 || groundoff_flag == 1)
{
		state[8]=0;
		state[9]=0;
}
    else
{
		state[8]= -(INS_attitude->pitch+pitch_com)*AtR;
  	state[9]= -INS_attitude->pitch_speed;
}

//		state[8]= 0;
//		state[9]= 0;
	 
 

	 if(pitch_ready==0)
   {
	 state[4]=0;
	 state[5]=0;
	 state[6]=0;
	 state[7]=0;
   }
   else if(lqr_reset_flag==1)
   {
	 state[4]=-VMC_joint_left.theta0;
	 state[5]=0;
	 state[6]=-VMC_joint_right.theta0;
	 state[7]=0;
   }
   else
   {
	 state[4]=-VMC_joint_left.theta0;
	 state[5]=-VMC_joint_left.theta0_speed;
	 state[6]=-VMC_joint_right.theta0;
	 state[7]=-VMC_joint_right.theta0_speed;
   }
//	state[4]=0;
//	state[5]=0;
//	state[6]=0;
//	state[7]=0;
		for(int i=0;i<40;i++)
	{
		LQR_KL[i]=LQR_K_calc(&Poly_Coefficient[i][0],VMC_joint_left.L );	
	}
		for(int i=0;i<40;i++)
	{
		LQR_KR[i]=LQR_K_calc(&Poly_Coefficient[i][0],VMC_joint_right.L );	
	}
//  matrix_multy(&KL,&State,&T_outL);
//  matrix_multy(&KR,&State,&T_outR);


	//求解输出矩阵
	if(length_flag == 1)
	{
	  matrix_multy(&K1,&State,&T_out);
	  //get_TW_component(&K1);
	}
	else if(length_flag ==2)
	{
	  matrix_multy(&K2,&State,&T_out);
	  //get_TW_component(&K2);
	}
	else if(length_flag ==3)
	{
	  matrix_multy(&K3,&State,&T_out);
	  //get_TW_component(&K3);
	}
	else if(length_flag ==4)
	{
	  matrix_multy(&K4,&State,&T_out);
	  //get_TW_component(&K3);
	}
	else if(length_flag == 0)
	{
	  matrix_multy(&K0,&State,&T_out);
	  // get_TW_component(&K0);
	}
	/*轮毂输出*/
	VMC_joint_left.TW=T_out.pData[0]*1;
	VMC_joint_right.TW=T_out.pData[1]*1;

	/*关节输出*/
  VMC_joint_left.TP=T_out.pData[2]*1;
	VMC_joint_right.TP=T_out.pData[3]*1;


}



//标志位判断
void flag_judge()
{

	/*起立标志位*/
  if(rc_Ctrl.isOnline == 0)
    rc_Ctrl.rc.s1_last = 3;
	if(rc_Ctrl.isOnline == 1 && rc_Ctrl.rc.s1 == 2)
   {  
	  lqr_flag = 0;
    lqr_reset_flag = 0;
		rc_Ctrl.rc.s1_last = rc_Ctrl.rc.s1;
   }
   if(rc_Ctrl.isOnline == 1 && rc_Ctrl.rc.s1 == 3 && rc_Ctrl.rc.s1_last == 2)
   {
		lqr_reset_flag = 1;
		rc_Ctrl.rc.s1_last = rc_Ctrl.rc.s1;
   }
   
   if(VMC_joint_left.theta0 < 1.32 && VMC_joint_left.theta0 > -1.3)
		leg_left_ready=1;
   else
    leg_left_ready=0;

   if(VMC_joint_right.theta0<1.32 && VMC_joint_right.theta0>-1.3)
		leg_right_ready=1;
   else
    leg_right_ready=0;


    if(fabs(INS_attitude->pitch)<20){
     pitch_ready_cnt++;
     if(pitch_ready_cnt>50){
     pitch_ready_cnt=50;
     pitch_ready=1;}}
    else{
     pitch_ready_cnt--;
     if(pitch_ready_cnt<-50){
     pitch_ready_cnt=-50;
     pitch_ready=0;}}

   if(lqr_reset_flag==1 && leg_left_ready==1 && leg_right_ready==1 && pitch_ready==1)
   {
     lqr_reset_flag=0;
     lqr_flag=1;
   }
 
   if(lqr_flag==1&&gostep_flag==0)
   lqr_cnt++;
   else if(gostep_flag!=0)
   {
   lqr_cnt=0;
   all_ready_flag=0;
   }
   else
   {
   lqr_cnt=0;
   all_ready_flag=0;
   }
   if(lqr_cnt>3000)
   {
   all_ready_flag=1;
   lqr_cnt=3000;
   }

    if(lqr_flag==1){
    if(VMC_joint_left.theta0<1.05&&VMC_joint_right.theta0<1.05)
    lqr_ready_flag=1;}



	
   /*控制腿长标志*/
	if((rc_Ctrl.rc.ch0 <= 600||rc_Ctrl.rc.ch0 >= 1500)&&lqr_flag == 1)
	 call_flag=1;

	if(call_flag==1)
	{
		if(rc_Ctrl.rc.ch0<=1400 && rc_Ctrl.rc.ch0>=1024)
			{length_flag++;
			call_flag=0;}
		if(rc_Ctrl.rc.ch0<=1024 && rc_Ctrl.rc.ch0>=800)
		{	length_flag--;
			call_flag=0;}
	}
	
		if(length_flag<0)
			length_flag=0;
		if(length_flag>4)
			length_flag=4;
	
		
	   
   /*跳跃标志判断*/
   if(rc_Ctrl.rc.s2 == 2&&lqr_flag==1&&call_flag1==0)
	   call_flag1=1;
   if(call_flag1==1)
   {
	  if(rc_Ctrl.rc.s2== 3)
	  {
		if(length_flag==0)
		{jump_flag=2;jump_count=70;}
		if(length_flag==1)	
		 jump_flag=3;
		call_flag1=0;
	  }
	}
   
}
//状态清零
void target_state_clean()
{     
	lqr_flag=0;
  lqr_reset_flag=0;
  lqr_ready_flag=0;
	distance_target = s_set;
	LPF_XS.last_output = s_set;
	LPF_XS.output = 0;
	model_target.yaw = INS_attitude->yaw_total_angle*AtR;
	model_target.s=s_set;
	Real_real_x = 0;
	model_target.s_dot = 0;
	speed_target = 0;
	LPF_XSdot.last_output = 0;
//	model_target.yaw = HI04_data_t.gyro_data.yaw_total*AtR;
	length_flag = 0;
	sdot_count=0;
	spin_flag = 0;
	groundoff_flag = 0;
	groundoff_flag_L=0;
	groundoff_flag_R=0;
	gleiten_flag = 0;
  jump_flag=0;
}
float limit(float energy)
{
    if(energy >60)
		energy = 60;
	return energy;
		
}
//里程累加
void s_control()
{
//	if(length_flag==0)
//		target_add=0.0002;
//	else if(length_flag<4)
//		target_add=0.0001;

//		LPF_XSdot.filter_coefficient=0.002;
//	
//	
//		LPF_XS.filter_coefficient=0.015f-fabs(Wheel.S_dot)*0.005;//将低通(急停)系数和速度相关，防止速度太快急停太大滑铲
//	if(LPF_XS.filter_coefficient<0.004)
//		 LPF_XS.filter_coefficient=0.004;
//	if(gleiten_flag == 0 && jump_flag < 1 && groundoff_flag == 0)
//	{
//		if(rc_Ctrl.rc.ch3>=1324)
//	  {
//		  distance_target += target_add*(rc_Ctrl.rc.ch3-1024)/100;
//		  if(length_flag>0)
//			speed_target = 0.2*(rc_Ctrl.rc.ch3-1024)/100;
//		  else 
//			speed_target = 0;
//		  sdot_count =0;
//		  spin_enable =0;
//	  }
//	  else if(rc_Ctrl.rc.ch3 <= 724)
//	  {
//		  distance_target+=target_add*(rc_Ctrl.rc.ch3-1024)/100;
//		  if(length_flag>0)
//			speed_target = 0.2*(rc_Ctrl.rc.ch3-1024)/100;
//		  else 
//			speed_target = 0;
//		  sdot_count =0;
//		  spin_enable =0;
//	  }
//	  else //急停
//	  {
//		  sdot_count=1;
//		  speed_target=0;
////		  if(distance_target>Wheel.S+0.4+0.06*pitch_com)
////			distance_target=Wheel.S+0.2;//急停
////		  else if(distance_target<Wheel.S-0.2)
////			distance_target=Wheel.S;//急停
////		  else
////			 distance_target=model_target.s;
//		   
//	  }
//  }
//	  if(sdot_count == 1)//急停
//	  {
//			LPF_XSdot.filter_coefficient=0.08;	 
//	  }

//		if(distance_target>Wheel.S+4)
//			distance_target=Wheel.S+4.5;
//		if(distance_target<Wheel.S-4)
//			distance_target=Wheel.S-4.5;//限输出

//	  model_target.s_dot = LPFilter(speed_target,&LPF_XSdot);
//	  model_target.s=LPFilter(distance_target,&LPF_XS);  
//	 
}
//速度控制
void sdot_control()
{
		static float flag = 1;
		LPF_XSdot.filter_coefficient=0.01-fabs(Real_real_xdot)*0.0008;//将低通(急停)系数和速度相关，防止速度太快急停太大滑铲
		if(LPF_XSdot.filter_coefficient<0.003)
		LPF_XSdot.filter_coefficient=0.003;
		if(gleiten_flag==0&&jump_flag<1&&groundoff_flag==0)
	  {   
			//速度控制
			if(rc_Ctrl.rc.ch3 > 1174 || rc_Ctrl.rc.ch3 < 974)
			{
					speed_target = 0.2*(rc_Ctrl.rc.ch3-1024)/100;
					sdot_count =0;

					if(rc_Ctrl.rc.ch3 > 1174)
						flag = 0;
					 if(rc_Ctrl.rc.ch3 < 974)
						flag = 0.5; 
			}

			else
			{
				speed_target = 0;
				sdot_count++;
			}
			if(sdot_count >1)
				 LPF_XSdot.filter_coefficient=0.08;
		
			if(sdot_count < 500)
			{
//				model_target.s = 0.1 + 0.07*(2-length_flag);
       model_target.s = s_set * flag;
				Real_real_x = 0;
			}

			model_target.s_dot = LPFilter(speed_target,&LPF_XSdot);
	}  

	
}
//里程误差控制
float x_error;
void s_error_control()
{
//	/*里程状态控制*/
//    x_error = 5;

//	/*添加功率控制 误差调整*/
//	if(length_flag == 0)
//		x_error = 2;
//		LPF_XS.filter_coefficient=0.001f-fabs(Real_real_xdot)*0.0002;//将低通(急停)系数和速度相关，防止速度太快急停太大滑铲
//	if(LPF_XS.filter_coefficient<0.0002)
//		 LPF_XS.filter_coefficient=0.0002;
//	if(spin_flag==0&&gleiten_flag==0&&jump_flag<1&&groundoff_flag==0)
//	{
//		if((rc_Ctrl.rc.ch3>=1124)&&fire_flag==0)
//	  {
//		  follow_target=0;
//		  distance_target=Real_real_x + 0.2 + x_error;
//		  spin_enable =0;
//		  
//	  }
//	  else if(rc_Ctrl.rc.ch3<=924)
//	  {
//		  follow_target=90;
//		  distance_target = Real_real_x + 0.2 - x_error ;
//		  spin_enable =0;
//		  
//	  }
//	  else 
//	  {
//		  LPF_XS.filter_coefficient = 0.1;
//		  follow_target=0;
//		  distance_target=model_target.s; 
//	  }
//  }
//		if(Real_real_xdot > -0.1 && Real_real_xdot <0.1)
//		{
//				spin_enable = 1;
//		
//		}
//		if(spin_flag == 1)
//		{
//			//speed_target = Real_real_xdot;
//			//if(spin_move_flag==1)
//				//speed_target = 0;
//		}
//	  model_target.s=LPFilter(distance_target,&LPF_XS);  
//	 
}

//目标状态控制
void state_control()
{   
   flag_judge();
	
	if(lqr_flag == 1 && tick_flag == 0){
   /*基本前后运动控制，二选一*/
	// s_error_control();
	if(control_s)
		s_control();//里程累加控制
	else
		sdot_control();//速度控制

    if((rc_Ctrl.rc.ch2-1124)> 0 || (rc_Ctrl.rc.ch2-924)< 0)
{
   model_target.yaw +=  -(rc_Ctrl.rc.ch2-1024)*YawSentivity;
   turn_flag=1;
}
   else
   turn_flag=0;
  
}
	if(rc_Ctrl.isOnline == 0 || (lqr_reset_flag == 0 && lqr_flag == 0) || leg_reset_flag == 1)
		target_state_clean();
}
