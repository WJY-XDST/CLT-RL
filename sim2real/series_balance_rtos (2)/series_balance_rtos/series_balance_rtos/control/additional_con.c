#include "vmc.h"
#include "wheel.h"
#include "WBR.h"
#include "arm_math.h"
#include "matrix.h"
#include "additional_con.h"
#include "WBR.h"
#include "all_init.h"
#include "torque.h"
#include "filter.h"
#include "ins.h"
#include "stp_task.h"
#include "power_con.h"
#include "A1_stm32.h"
#include "pid.h"

const float Rlb = 0.45f;//驱动轮间距

float t= 0.002;
perior PErior;
extern float Real_real_xdot;
extern float Real_real_x;
uint8_t jump_flag=0;//跳跃标志位
int16_t jump_count=0;//跳跃时间
uint8_t collide_flag=0;
int8_t tick_flag;
/*轮子自适应控制*/
//选了矩阵不为0的元素
const A_k A_K=
{
   .X_P1=1,
   .X_P2=-0.0446,
   .X_P3=-0.0446,
   .phi_P1=1,
   .phi_P2=-0.0498,
   .phi_P3=0.0498,
};

float32_t u[4];
const float A_k1[6]={1,-0.0446,0.0446,1,-0.0498,0.0498};
const float A_k2[6]={1,-0.0408,0.0408,1,-0.0452,0.0452};
const float A_k3[6]={1,-0.0390,0.0390,1,-0.0430,0.0430};
const float B_k1[8]={0.0143,0.0143,-0.0050,-0.0050,-0.0081,0.0081 ,-0.0056,0.0056};
const float B_k2[8]={0.0119,0.0119,-0.0033,-0.0033,-0.0109,0.0109,-0.0037,0.0037};
const float B_k3[8]={0.0110,0.0110,-0.0027,-0.0027,-0.0120,0.0120,-0.0029,0.0029};
//先验估计里程速度及摆角速度

void w_perior(perior* Perior,float *A_K, float *B_K)
{
	if(rc_Ctrl.isOnline ==1&&lqr_flag ==1)
	{
		u[0]=Wheel.Motors3508.motor[0].Data.Output*CHASSIS_MOTOR_CURRENT_TO_TORQUE_SEN;
		u[1]=-Wheel.Motors3508.motor[1].Data.Output*CHASSIS_MOTOR_CURRENT_TO_TORQUE_SEN;
		u[2]=VMC_joint_left.TP*9.11f;
		u[3]=VMC_joint_right.TP*9.11f;
	}	
	else
	{
		u[0]=0;
		u[1]=0;
		u[2]=0;
		u[3]=0;
	}
	 Perior->Sdot_perior=A_K[0]*Real_real_xdot+A_K[1]*VMC_joint_left.theta0+A_K[2]*VMC_joint_right.theta0
		+B_K[0]*u[0]+B_K[1]*u[1]+B_K[2]*u[2]+B_K[3]*u[3];
	 Perior->phidot_perior=A_K[3]*INS_attitude->yaw_speed+A_K[4]*VMC_joint_left.theta0+A_K[5]*VMC_joint_right.theta0
		+B_K[4]*u[0]+B_K[5]*u[1]+B_K[6]*u[2]+B_K[7]*u[3];
	 Perior->xldot_perior=(Perior->Sdot_perior)-Perior->phidot_perior*Rlb*0.5;
	 Perior->xrdot_perior=(Perior->Sdot_perior)+Perior->phidot_perior*Rlb*0.5;
}


/*轮角速预测补偿力矩*/
void Wheel_predictor_com(wheel* Wheel,perior* Perior)
{
	float K_adapt=1;
	if(length_flag>=2)
		w_perior(Perior,(float *)A_k3,(float *)B_k3);
	else if(length_flag==1)
		w_perior(Perior,(float *)A_k2,(float *)B_k2);
	else 
		w_perior(Perior,(float *)A_k1,(float *)B_k1);
	//Wheel->TW_adapt_L=K_adapt*(Perior->xldot_perior-Wheel->S_L_dot);
	//Wheel->TW_adapt_R=K_adapt*(Perior->xrdot_perior-Wheel->S_R_dot);
}


/*离地检测*/
float GroundOffFnl = 60;//离地判断阈值
float GroundOnFnl =  20;//落地判断阈值
float GroundOffFnr = 60;//离地判断阈值
float GroundOnFnr =  20;//落地判断阈值
int16_t GroundOffcount;//离地延时计数
int16_t GroundOncount;//落地延时计数
int16_t GroundOffcount_L = 0;//左腿离地延时计数
int16_t GroundOffcount_R = 0;//右腿离地延时计数
int16_t GroundOncount_L = 0;//左腿落地延时计数
int16_t GroundOncount_R = 0;//右腿落地延时计数
int16_t landingcount = 0;//飞坡落地计数00
int16_t body_error_count = 0;//机体腿姿态错误判断计数（防止被撞或急旋导致单腿芭蕾舞）


int8_t groundoff_flag=0;//机体离地标志位
int8_t groundoff_flag_R=0;//左腿离地标志位
int8_t groundoff_flag_L=0;//右腿离地标志位
int8_t landing_flag=1;//飞坡落地标志位
/*支持力解算判断离地*/
float L0llast , L0ldotlast, thetallast, thetaldotlast;
float L0rlast, L0rdotlast, thetarlast, thetardotlast;
float PL,PR,al,ar,Mw = 0.635;
int8_t ground_dection(VMC_joint * vmcl,VMC_joint * vmcr)//参考王工离地判断
{
	float az = 0;//机体z方向加速度
	//腿部各运动变量，腿长，速度，加速度
    float L0l ,L0ldot, L0ldotdot, thetal, thetaldot, thetaldotdot,L0r, L0rdot, L0rdotdot, thetar, thetardot, thetardotdot;
    az = ((INS_attitude->accel[1]) * arm_sin_f32((INS_attitude->pitch) * AtR) + (INS_attitude->accel[2]) * arm_cos_f32(INS_attitude->pitch*AtR) ) - 9.9;
    if ((az < 0.3) && (az > -0.3))
		az = 0;
	if(INS_attitude->pitch>-20&&INS_attitude->pitch<20)//机体正常角度才判断
	{
	  PL=(vmcl->feed_torque+vmcl->length_pid_out)*arm_cos_f32(vmcl->theta0)+vmcl->TP*arm_sin_f32(vmcl->theta0)/vmcl->L;
	  PR=(vmcr->feed_torque+vmcr->length_pid_out)*arm_cos_f32(vmcr->theta0)+vmcr->TP*arm_sin_f32(vmcr->theta0)/vmcr->L;
	}
	else
	{
	  PL=0;
	  PR=0;
	}
	//左腿
	L0l=vmcl->L;
	L0ldot=(L0l-L0llast)*500;
	L0ldotdot=(L0ldot-L0ldotlast)*500;
	L0llast=L0l;
	L0ldotlast=L0ldot;
	
	thetal=vmcl->theta0;
	thetaldot=vmcl->theta0_speed;
	thetaldotdot=(thetaldot-thetaldotlast)*500;
	thetaldotlast=thetaldot;
	
	al=az-L0ldotdot*arm_cos_f32(thetal)+2*L0ldot*thetaldot*arm_sin_f32(thetal)+L0l*arm_cos_f32(thetal)*thetaldot*thetaldot
	   +L0l*arm_sin_f32(thetal)*thetaldotdot;
	
	//右腿
	L0r=vmcr->L;
	L0rdot=(L0r-L0rlast)*500;
	L0rdotdot=(L0rdot-L0rdotlast)*500;
	L0rlast=L0r;
	L0rdotlast=L0rdot;
	
	thetar=vmcr->theta0;
	thetardot=vmcr->theta0_speed;
	thetardotdot=(thetardot-thetardotlast)*500;
	thetardotlast=thetardot;
	
	ar=az-L0rdotdot*arm_cos_f32(thetar)+2*L0rdot*thetardot*arm_sin_f32(thetar)+L0r*arm_cos_f32(thetar)*thetardot*thetardot
	+L0r*arm_sin_f32(thetar)*thetardotdot;
	
	
    vmcl->Fn=Mw*al + PL + Mw * 9.8;
	vmcr->Fn=Mw*ar + PR + Mw * 9.8;
	
	if((vmcl->Fn<GroundOffFnl)&&(vmcr->Fn<GroundOffFnr)&&all_ready_flag==1&&gleiten_flag==0&&jump_flag<=1)
	{
	     GroundOffcount++;
		if(GroundOffcount>=10)
		{
			groundoff_flag=1;
			GroundOffcount=0;
			speed_target = 0;

				model_target.s =s_set;
				Real_real_x = 0;
//				model_target.s_dot = 0;
//				speed_target = 0;
				LPF_XSdot.last_output = 0;
			
		}
		
	}
	else if(groundoff_flag==1&&(vmcl->Fn>GroundOnFnl)&&(vmcr->Fn>GroundOnFnr))//落地
	{
	    GroundOncount++;
		if(GroundOncount>=4)
		{
		    groundoff_flag=0;
//        if(land_leg==1){
//        length_flag=0;
//        land_leg=0;}
		    GroundOncount=0;
			groundoff_flag_L=0;
		    GroundOncount_L=0;
			groundoff_flag_R=0;
		    GroundOncount_R=0;

		}
	}
	else 
	{
		GroundOffcount=0;
	    GroundOncount=0;
	}
	
	 if(landing_flag==0&&groundoff_flag==0)
	 {
			landingcount++;
//			distance_target=Real_real_x+(follow_target_com/90)*1;
//			model_target.s = distance_target;
//			//speed_target=-(follow_target_com/90)*0.3;
//			LPF_XS.last_output = distance_target;
		  // if(landingcount>200)
		//{
			   landingcount=0;
			   landing_flag=1;
		//}
	 }
	   
//	if((vmcl->Fn<GroundOffFnl)&&all_ready_flag==1&&gleiten_flag==0&&jump_flag<=1&&fabs(INS_attitude->pitch)<20)//左腿的离地
   if(vmcl->Fn<GroundOffFnl&&all_ready_flag==1)
	{
	     GroundOffcount_L++;
		if(GroundOffcount_L>=10)
		{
			groundoff_flag_L=1;
			GroundOffcount_L=0;
		}
	}
	else if(((vmcl->Fn>GroundOnFnl)&&(length_flag>=0)))
	{
	    GroundOncount_L++;
		if(GroundOncount_L>=4)
		{
			
		   groundoff_flag_L=0;
		   GroundOncount_L=0;
		}
	}
	else 
	{
		GroundOffcount_L=0;
	    GroundOncount_L=0;
	}
	
 // 	if((vmcr->Fn<GroundOffFnr)&&all_ready_flag==1&&gleiten_flag==0&&jump_flag<=1&&fabs(INS_attitude->pitch)<20)//右腿的离地
   if(vmcr->Fn<GroundOffFnr&&all_ready_flag==1)
	{
	     GroundOffcount_R++;
		if(GroundOffcount_R>=10)
		{
			
			groundoff_flag_R=1;
			GroundOffcount_R=0;
		}
	}
	else if(((vmcr->Fn>GroundOnFnr)&&(length_flag>=0)))
	{
	    GroundOncount_R++;
		if(GroundOncount_R>=4)
		{
		   groundoff_flag_R=0;
		   GroundOncount_R=0;
		}
	}
	else 
	{
		GroundOffcount_R=0;
	    GroundOncount_R=0;
	}

}


/*滑铲检测*/
int8_t gleiten_flag=0;//滑铲标志
int16_t gleiten_count1=0;//滑铲延时计数
int16_t gleiten_count2=0;//滑铲起立延时计数
void gleiten_dection(VMC_joint * vmcl,VMC_joint * vmcr)
{
   if((INS_attitude->pitch>12||INS_attitude->pitch<-12)&&(Real_real_xdot>1.7||Real_real_xdot<-1.7))
   {   
	   gleiten_count1++;
	   if(gleiten_count1>=70)
	   { gleiten_flag=1;gleiten_count1=0;}
   }
   else
   {
     gleiten_count1=0;
   }
   if(gleiten_flag==1)
   {
	   length_flag=0;
	   spin_flag = 0;
	   model_target.yaw = INS_attitude->yaw_total_angle*AtR;
	   if(Real_real_xdot<0.7&&Real_real_xdot>-0.7&&vmcl->L<0.2&&vmcr->L<0.2)
	   {
			gleiten_count2++;
			if(!control_s)
			{
				model_target.s=0;
				Real_real_x = 0;
				model_target.s_dot = 0;
				speed_target = 0;
				LPF_XSdot.last_output = 0;
			}
			else
			{
				distance_target=Real_real_x;
				model_target.s=Real_real_x;
				LPF_XS.last_output=model_target.s;
				LPF_XS.output=model_target.s;//清里程误差
			}
			collide_flag = 0;
		 
		   if(gleiten_count2>=100)
		   {
			   length_flag=0;//先0档起立（防打滑）
			   gleiten_flag=0;
		   }
	   } 
   }
   if(gleiten_flag==0)
   {
	   if(gleiten_count2 > 95&&gleiten_count2<300)
	   {
		   if(!control_s)
			{
				model_target.s=0;
				Real_real_x = 0;
				model_target.s_dot = 0;
				speed_target = 0;
				LPF_XSdot.last_output = 0;
			}
		   gleiten_count2++;
	   }
		else if(gleiten_count2 >=300)
		{
			if(!control_s)
			{
				model_target.s=0;
				Real_real_x = 0;
				model_target.s_dot = 0;
				speed_target = 0;
				LPF_XSdot.last_output = 0;
			}
			length_flag =1;//再一档起立
			gleiten_count2=0;
		}
		else
			gleiten_count2=0;
		
	   
   }
   }
float spin_add=0;//用于自旋缓启动
float move_rad;//自旋平移方向角
int8_t spin_move_flag=0;//自旋平移标志位
float spin_com = 0;//自旋平移方向补偿
int16_t spin_error_count=0; //自旋出错计数，用于自旋被撞
int16_t spin_error_count1=0;//自旋出错计数，用于自旋被撞
int16_t spin_error_count2=0;//自旋出错计数，用于自旋被撞
float spin_speed=0;

int16_t jump_landing_count= 0;
int16_t jump_landing_flag= 0;
/*跳跃控制函数*/
void jump_control(VMC_joint* vmcl,VMC_joint* vmcr)
{
   if(jump_flag==3)//起跳收腿，二挡降一档再起跳，现在限制起跳一档，故没用了
   {
      length_flag=0;
      jump_count++;
	    if(jump_count>70)
		   jump_flag=2;
   }
   if(jump_flag==2&&jump_count<160)//一段跳起跳输出
   {
		 pitch_com=10;//pitch补偿值

     vmcl->F0=-80;
	   vmcr->F0=-80;
		 jump_count++;
   }
	
   if(jump_flag==4&&jump_count<150)//二段跳第一跳起跳输出
   {
		 pitch_com=4;
     vmcl->F0=-200;
		 vmcr->F0=-200;
	   jump_count++;
   }
	 
   if(jump_flag==5&&jump_count<150)//二段跳第二跳起跳输出
   {
		 pitch_com=4;
     vmcl->F0=-200;
	   vmcr->F0=-200;
	   jump_count++;
   }
	 
	 if(jump_count>=160&&jump_count<320)//起跳后收腿
	{

		 jump_count++;
	   jump_flag=1;
		 length_flag=0;
	}
//	 if(jump_count>=250&&jump_count<400)//收腿后伸腿缓冲
//	{
//		
//		 jump_count++;
//	   jump_flag=1;
//		 length_flag=2;
//	}
	
	if(jump_count>=320)//完毕
	{ 	
//			speed_target = 0;

//				model_target.s =s_set;
//				Real_real_x = 0;
			pitch_com=2;
			jump_flag=0;
			jump_count=0;
			length_flag=1;
	}
		
	if(rc_Ctrl.isOnline==0)
	{ 
			jump_flag=0;
	    jump_count=0;
			jump_landing_count = 0;
			jump_landing_flag = 0;
	}
}


int16_t  jump_ditance=900;//起跳距离
int16_t  jump_double_ditance=900;//起跳距离
uint8_t  autojump_flag=0;//可以自动跳标志
int16_t  auto_jump_count=0;//延时计数，消除距离测量出现一帧错误值

uint8_t  autojump_double_flag=0;//可以自动跳标志
int16_t  auto_jump_double_count=0;//二连跳自动跳延时计数，消除距离测量出现一帧错误值

uint8_t  nextjump_flag=1;//下一次跳标志位，目的是为了跳完一次后防止继续跳，需手动重启
uint8_t  nextjump_double_flag=1;//下一次跳标志位，目的是为了跳完一次后防止继续跳，需手动重启
/*自动跳跃*/
void auto_jump()
{
    if(rc_Ctrl.rc.s2==1)
	{
		autojump_flag=1;
	}
	else
	{ 
		autojump_flag=0;
		nextjump_flag=1;
	}
//	if(autojump_flag==1&&nextjump_flag==1)
//		length_flag = 0;
	jump_ditance=(int)(Real_real_xdot*50);//将起跳距离与速度相关
	if(Pack_Data.distance<jump_ditance&&Pack_Data.distance>40&&Real_real_xdot>0.6&&INS_attitude->pitch>-9&&INS_attitude->pitch<9)
	{
		auto_jump_count++;//检测时间阈值，防止干扰距离
		if(auto_jump_count>=20&&autojump_flag==1&&nextjump_flag==1)
		{
		   if(length_flag==1)	
		    jump_flag=3;
		   auto_jump_count=0;
		   nextjump_flag=0;
	   }
	}
	else 
		auto_jump_count=0;
}
/*自动跳跃（二连）*/
//void auto_jump_double()
//{
//	if(rc_Ctrl.key_E ==1||(vT13.rc.mode_sw* vT13.key_E)==1)
//	{
//		//autojump_double_flag=1;
//		follow_target_com = 90;
//		follow_target = 90;
//	}
//	else
//	{ 
//		autojump_double_flag=0;
//		nextjump_double_flag=1;
//		jump_double_ditance  = 0;
//	}
//	if(autojump_double_flag==1&&nextjump_double_flag==1)
//		length_flag = 1;
//	jump_double_ditance=(int)(Real_real_xdot*5);//将起跳距离与速度相关
//	if(stp_distance<jump_double_ditance&&stp_distance>200&&Real_real_xdot>0.4&&INS_attitude->pitch>-9&&INS_attitude->pitch<9)
//	{
//		auto_jump_double_count++;//检测时间阈值，防止干扰距离
//		if(auto_jump_double_count>=20&&autojump_double_flag==1&&nextjump_double_flag==1)
//		{
//			jump_flag=4;jump_count=70;
//			nextjump_double_flag=2;
//	   }
//	}
//	if(auto_jump_double_count >= 20 && nextjump_double_flag ==2)
//	{
//		auto_jump_double_count ++;
//		if(auto_jump_double_count >235)
//		{
//			jump_flag=5;
//			jump_count=100;
//			auto_jump_double_count=0;
//			nextjump_double_flag = 0;
//		}
//	}
//	if(gleiten_flag ==1||(nextjump_double_flag==1&&auto_jump_double_count<20))
//		auto_jump_double_count = 0;


//}
int16_t collide_count1=0;//被撞后计数
int16_t collide_count2=0;//被撞后计数
void collide_check()
{
	/*被撞后单腿独立（一长一短直接滑铲）*/
	if((groundoff_flag_L == 1||groundoff_flag_R ==1 )&& fabs(INS_attitude->roll)>15)
	{
		collide_count1 ++;
		if(collide_count1>20)
		{
			collide_flag = 1;
			gleiten_flag = 1;
			collide_count1 = 0;
		}
	}
	else
		collide_count1 = 0;
	if((groundoff_flag_L == 1||groundoff_flag_R ==1)&&VMC_joint_right.L>0.21&&VMC_joint_right.L>0.21)//还有一个可能
	{
	
	}

}


int8_t left_front = 1;//左前关节电机
int8_t left_back = 1;//左后关节电机
int8_t right_front = 1;//右前关节电机
int8_t right_back = 1;//右后关节电机

void A1_motor_check()//检查A1是否进保护
{
	if(Motor_zero_receive_left.mode == 10)
		left_back = 1;
	else
		left_back = 0;
	if(Motor_one_receive_left.mode == 10)
		left_front = 1;
	else
		left_front = 0;
	if(Motor_zero_receive_right.mode == 10)
		right_front = 1;
	else
		right_front = 0;
	if(Motor_one_receive_right.mode == 10)
		right_back = 1;
	else
		right_back = 0;
	
}


int gostep_flag=0;
//抬腿上台阶
void change_leg_gostep()
{
  if(length_flag==4&&VMC_joint_left.theta0<-0.5&&VMC_joint_right.theta0<-0.5){
    length_flag=0;
//			speed_target = 0;
//				model_target.s =s_set;

TARget_length_l = 0.25;
TARget_length_r = 0.25;
    gostep_flag=1;}
}


   






