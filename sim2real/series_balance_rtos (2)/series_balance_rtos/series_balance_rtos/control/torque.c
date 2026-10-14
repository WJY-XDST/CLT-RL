#include "motor.h"
#include "WBR.h"
#include "wheel.h"
#include "torque.h"
#include "vmc.h"
#include "A1_stm32.h"
#include "pid.h"
#include "additional_con.h"
#include "mpu6050.h"
#include "all_init.h"
#include "power_con.h"
#include "online_check_task.h"
#include "dmmotor.h"
#include "math.h"
#include "stdlib.h"
#include "all_init.h"
#include "LK_motor.h"


//819.2为1A
//总质量20.18kg（不带弹）
//腿部 1.175kg
//轮子00.625kg
float M1=6.17;//算机体质量一半5.97
extern float Real_real_xdot;
float torque_sen=0.3;//电机输出放缩系数，0档小一点
float torque_sen_dm=1.0;//达妙电机输出放缩系数
float torque_leg_sen=0.1;//腿部复位放缩系数

int leg_flag=0; //检测机体是否已到可使用lqr的标志位

int enable_flag=0;
int disable_flag=0;
int able_time;

int torque_cnt=0;
/*3508电机力矩转换*/
void torque_motor3508(wheel* Wheel,float torquel,float torquer)
{
	if(length_flag != 0)
	{
			torque_sen=0.3;
	}
	else 
		torque_sen=0.3;
	
	if(lqr_flag == 1)//起立正常输出
	{
		Wheel->Motors3508.motor[0].Data.Output=(int)((torquel - Wheel->TW_adapt_L)/CHASSIS_MOTOR_CURRENT_TO_TORQUE_SEN*torque_sen);
		Wheel->Motors3508.motor[1].Data.Output=(int)((-torquer+ Wheel->TW_adapt_R )/CHASSIS_MOTOR_CURRENT_TO_TORQUE_SEN*torque_sen);
	}
	if(gleiten_flag==1)//滑铲不输出
	{
		Wheel->Motors3508.motor[0].Data.Output = BasePID_SpeedControl(&pid_chassis_move,0,Wheel->Motors3508.motor[0].Data.SpeedRPM);
		Wheel->Motors3508.motor[1].Data.Output = BasePID_SpeedControl(&pid_chassis_move,0,Wheel->Motors3508.motor[1].Data.SpeedRPM);
	}
	if(groundoff_flag_L==1)//单腿离地单腿不输出
		Wheel->Motors3508.motor[0].Data.Output = 0;

	if(groundoff_flag_R==1)//单腿离地单腿不输出
		Wheel->Motors3508.motor[1].Data.Output = 0;
	if(groundoff_flag == 1 || (jump_count>180&&jump_count < 280))//机体离地都不输出
	{
		Wheel->Motors3508.motor[0].Data.Output = 0;
		Wheel->Motors3508.motor[1].Data.Output = 0;
	}
//	if(rc_Ctrl.isOnline == 1&&lqr_flag == 1)
//		power_control(Wheel);//港科方案功率控制，代码有误用不了
	/*限你幅*/
	if( Wheel->Motors3508.motor[0].Data.Output>16000)
		  Wheel->Motors3508.motor[0].Data.Output=16000;
	if( Wheel->Motors3508.motor[0].Data.Output<-16000)
		 Wheel->Motors3508.motor[0].Data.Output=-16000;
	if( Wheel->Motors3508.motor[1].Data.Output>16000)
		 Wheel->Motors3508.motor[1].Data.Output=16000;
	if( Wheel->Motors3508.motor[1].Data.Output<-16000)
		 Wheel->Motors3508.motor[1].Data.Output=-16000;
	/*最终输出*/
	//遥控在线，起立
    if(rc_Ctrl.isOnline == 1 && lqr_ready_flag==1)
	{
		MotorFillData(&Wheel->Motors3508.motor[1],Wheel->Motors3508.motor[1].Data.Output);
		MotorFillData(&Wheel->Motors3508.motor[0],Wheel->Motors3508.motor[0].Data.Output);
	}
	//遥控离线，关节复位，轮毂电机离线
	if(rc_Ctrl.isOnline==0 || leg_reset_flag ==1||wheel_left_online==0||wheel_right_online==0|| lqr_flag==0|| leg_flag==0)
	{
		MotorFillData(&Wheel->Motors3508.motor[0],0);
		MotorFillData(&Wheel->Motors3508.motor[1],0);
	}
}



void torque_LK(VMC_joint* vmcl,VMC_joint* vmcr)
{
 torque_cnt++;

 if(rc_Ctrl.isOnline == 1)
{
	  if(lqr_flag == 1 || lqr_reset_flag==1)
	 {
     leg_flag=1;
     if(lqr_ready_flag == 1)
		{
			 if(fabs(INS_attitude->pitch)<30 && vmcr->theta0<pi/2 && vmcl->theta0<pi/2)
			 leg_flag=1;
     else
     {
			 leg_flag=0;
			 lqr_flag=0;
			 lqr_reset_flag=0;                                                                                                                            
			 lqr_ready_flag=0;
			 LK_MotorFillData(&lkmotor[0],0);
			 LK_MotorFillData(&lkmotor[1],0);
			 LK_MotorFillData(&lkmotor[2],0);
			 LK_MotorFillData(&lkmotor[3],0);
	   }
		}
	   if(leg_flag==1)
			{
			 LK_MotorFillData(&lkmotor[0],vmcl->torque_f);//最长大腿2.6，小腿11.7，最短大腿6，小腿15
			 LK_MotorFillData(&lkmotor[1],vmcl->torque_b);
			 LK_MotorFillData(&lkmotor[3],-vmcr->torque_f);//-vmcr->torque_f+11.7
			 LK_MotorFillData(&lkmotor[2],-vmcr->torque_b);//-vmcr->torque_b-2.6
		   LK_MotorFillDataInSpeed(&lkmotor[0],vmcl->reset_speed);
		   LK_MotorFillDataInSpeed(&lkmotor[1],vmcl->reset_speed);
		   LK_MotorFillDataInSpeed(&lkmotor[2],-vmcr->reset_speed);
       LK_MotorFillDataInSpeed(&lkmotor[3],-vmcr->reset_speed);
			}
   }
	  else
	  {
			float torr;
		if(vmcl->L > 0.35 && vmcr->L > 0.35) torr=0;
else torr = 9;
		 LK_MotorFillData(&lkmotor[0],-torr);
		 LK_MotorFillData(&lkmotor[1],torr);
		 LK_MotorFillData(&lkmotor[2],-torr);
		 LK_MotorFillData(&lkmotor[3],torr);
	  }
}
  else
{
		 LK_MotorFillData(&lkmotor[0],0);
		 LK_MotorFillData(&lkmotor[1],0);
		 LK_MotorFillData(&lkmotor[2],0);
		 LK_MotorFillData(&lkmotor[3],0);
		 LK_MotorFillDataInSpeed(&lkmotor[0],0);
		 LK_MotorFillDataInSpeed(&lkmotor[1],0);
		 LK_MotorFillDataInSpeed(&lkmotor[2],0);
     LK_MotorFillDataInSpeed(&lkmotor[3],0);
}

   if(torque_cnt%2==0)
{
		if(vmcl->reset_speed != 0 && rc_Ctrl.isOnline == 1)
	{
		LK_Motor_CanOutputInSpeed(&lkmotor[0]);
		LK_Motor_CanOutputInSpeed(&lkmotor[1]);
	}
	else
	{
		LK_Motor_CanOutput(&lkmotor[0]);
		LK_Motor_CanOutput(&lkmotor[1]);
	}
}
   if(torque_cnt%2==1)
{
		if(vmcr->reset_speed != 0 && rc_Ctrl.isOnline == 1)
	{
		LK_Motor_CanOutputInSpeed(&lkmotor[2]);
		LK_Motor_CanOutputInSpeed(&lkmotor[3]);
	}
	else
	{
		LK_Motor_CanOutput(&lkmotor[2]);
		LK_Motor_CanOutput(&lkmotor[3]);
	}
}


}
void torque_MIT_DM(VMC_joint* vmcl,VMC_joint* vmcr)
{
 torque_cnt++;
 if(rc_Ctrl.isOnline == 1&& disable_flag == 1)
	{
		enable_flag = 0;
		disable_flag = 0;
}
 if(enable_flag == 0 && disable_flag== 0){
    able_time ++;
	  if(able_time >= 100)
	  {
		able_time  = 0;
		enable_flag  = 1;
		disable_flag = 0;	
	  }}
 if(rc_Ctrl.isOnline == 1){
	  if(lqr_flag == 1 || lqr_reset_flag==1)
	 {
     leg_flag=1;
     if(lqr_ready_flag == 1){
     if(fabs(INS_attitude->pitch)<70&&vmcr->theta0<pi/2&&vmcl->theta0<pi/2)
     leg_flag=1;
     else
     {
     leg_flag=0;
     lqr_flag=0;
     lqr_reset_flag=0;                                                                                                                            
     lqr_ready_flag=0;
     DMiaoMitControl(&dmmotor[1],0,0,0,0,0);
     DMiaoMitControl(&dmmotor[2],0,0,0,0,0);
     DMiaoMitControl(&dmmotor[0],0,0,0,0,0);
     DMiaoMitControl(&dmmotor[3],0,0,0,0,0);
	   }}
	   if(leg_flag==1){
     DMiaoMitControl(&dmmotor[1],0,vmcl->reset_speed,0,vmcl->reset_speed_kd,vmcl->torque_f);//最长大腿2.6，小腿11.7，最短大腿6，小腿15
     DMiaoMitControl(&dmmotor[3],0,vmcl->reset_speed,0,vmcl->reset_speed_kd,vmcl->torque_b);
     DMiaoMitControl(&dmmotor[0],0,-vmcr->reset_speed,0,vmcr->reset_speed_kd,-vmcr->torque_f);//-vmcr->torque_f+11.7
     DMiaoMitControl(&dmmotor[2],0,-vmcr->reset_speed,0,vmcr->reset_speed_kd,-vmcr->torque_b);//-vmcr->torque_b-2.6
     }
   }
	  else
	  {
    DMiaoMitControl(&dmmotor[1],0,0,0,0,0);
    DMiaoMitControl(&dmmotor[2],0,0,0,0,0);
    DMiaoMitControl(&dmmotor[0],0,0,0,0,0);
    DMiaoMitControl(&dmmotor[3],0,0,0,0,0);
	  }}
  else
	 {
		enable_flag  = 0;
		disable_flag = 1;
		able_time = 0;
	 }
 if(enable_flag == 0 && disable_flag== 0){
  if(able_time>0&&able_time<=10)
	DMiao_Enable(can1,&dmmotor[0]);
  if(able_time<=20&&able_time>10)
	DMiao_Enable(can1,&dmmotor[1]);	 
  if(able_time<=30&&able_time>20)
	DMiao_Enable(can1,&dmmotor[2]);	
  if(able_time<=40&&able_time>30)
	DMiao_Enable(can1,&dmmotor[3]);
}	
	
 if(enable_flag == 1 && disable_flag== 0){
//	DMiao_Enable(can1,&dmmotor[2]);	

   if(torque_cnt%2==0){
	DMiao_CanOutput(can1, &dmmotor[0]);
	DMiao_CanOutput(can1, &dmmotor[1]);}
   if(torque_cnt%2==1){
	DMiao_CanOutput(can1, &dmmotor[2]);
	DMiao_CanOutput(can1, &dmmotor[3]);}

}

 if(enable_flag == 0 && disable_flag== 1){
	DMiao_Disable(can1,&dmmotor[0]);
	DMiao_Disable(can1,&dmmotor[1]);
	DMiao_Disable(can1,&dmmotor[2]);
	DMiao_Disable(can1,&dmmotor[3]);}

}
float length_target=0;//目标腿长0.28-0.16
float roll_compen;//roll补偿

/*目标腿长赋值*/
void length_control()
{
		if(lqr_flag==1)
	{
		 if(length_flag==0)
		 {
			 length_target=0.17; 	
s_set = 0.7;
		 } 
		else if(length_flag==1)
		{
			length_target=0.22;  
s_set = 0;

		 
		}
		 else if(length_flag==2)
		{
			length_target=0.27;  
s_set = -0.3;

		}
		else if(length_flag==3)
		{
			length_target=0.32;  
s_set = -0.6;

		}
			else if(length_flag==4)
		{
			length_target=0.37;  
s_set = 0;

		}
	 }
		else
			length_target=0.2; 
}
float32_t Coefficientl[2][4]=
{{-40156.0149971786,	32293.5788141137,	-8082.00724933927,	730.864371815381},
{-38259.1908194822,	31059.4559906550,	-7743.55823979902	,693.056173348898}};

float spring_feedforwardl=0;//气弹簧前馈
float spring_feedforwardr=0;

static float F0_TD(VMC_joint* vmc,float Expect,float r,float h)
{
    double fh= -r * r * (vmc->v1 - Expect) - 2 * r * vmc->v2;
    vmc->v1 += vmc->v2 * h;
    vmc->v2 += fh * h;
		return vmc->v1;
}
float r = 160;
/*腿部支持力控制器，用于腿长控制，roll补偿*/
void F0_control(VMC_joint* vmc)
{  
	
	length_control();
//	if(spin_flag==0)
//        vmc->F_intertial=M1*vmc->L/(0.4)*INS_attitude->yaw_speed*Real_real_xdot*0.11f;
//	else
//		vmc->F_intertial= 0;
	    //vmc->F_intertial=-M1*vmc->L/(0.4)*INS_attitude->yaw_speed*Real_real_xdot*0.11f;//和正常极性反的自旋能不飘一点，为什么呢？
	if(jump_flag<2||jump_flag==3)
	{
		if(gleiten_flag == 0 && collide_flag == 0 && nextjump_flag == 1 && lqr_flag == 1 && jump_flag == 0)
			roll_compen=BasePID_rollControl(&pid_roll_compen,2.5,INS_attitude->roll);
		else 
			roll_compen = 0;
    if(lqr_ready_flag == 1){
		vmc->feed_torque=acc_g*M1/arm_cos_f32(vmc->theta0);
    }
    else{
    vmc->feed_torque=0;
		}
    if(vmc->feed_torque>150) vmc->feed_torque=150;
    if(vmc->feed_torque<-150) vmc->feed_torque=-150;
    if(vmc->theta0>pi/2) vmc->feed_torque=-vmc->feed_torque;

		if(vmc==&VMC_joint_left)
			vmc->length_pid_out=BasePID_lengthControl_l(&pid_legleft,length_target,roll_compen,vmc->L);
		if(vmc==&VMC_joint_right)
			vmc->length_pid_out=BasePID_lengthControl_r(&pid_legright,length_target,roll_compen,vmc->L);
	
	} 
	else vmc->length_pid_out=0;

	spring_feedforwardl=LQR_K_calc(&Coefficientl[0][0],VMC_joint_left.L);	
	spring_feedforwardr=LQR_K_calc(&Coefficientl[1][0],VMC_joint_right.L);	

//	spring_feedforwardl=LQR_K_calc(&Coefficientl[0][0],VMC_joint_left.L );	
//	spring_feedforwardr=LQR_K_calc(&Coefficientl[1][0],VMC_joint_right.L );	
//if(vmc==&VMC_joint_right)
//	vmc->F0=spring_feedforwardl-vmc->feed_torque-vmc->length_pid_out -vmc->roll_flag*vmc->F_intertial*0;
//if(vmc==&VMC_joint_left)
//	vmc->F0=spring_feedforwardl-vmc->feed_torque-vmc->length_pid_out -vmc->roll_flag*vmc->F_intertial*0;

//	vmc->F0=-vmc->length_pid_out;
	if(vmc==&VMC_joint_right)
		vmc->F0=spring_feedforwardr-vmc->feed_torque-vmc->length_pid_out;
	if(vmc==&VMC_joint_left)
		vmc->F0=spring_feedforwardl-vmc->feed_torque-vmc->length_pid_out;

	if(groundoff_flag==1&&jump_flag==0)
  mySaturate(&vmc->F0,-120.0f,120.0f);//限幅
}


/*腿部转向控制器*/
void turn_control(VMC_joint* vmc)
{ 
	//float turn_t = -BasePID_Turncontrol(&Turn_PID,model_target.yaw,INS_attitude->yaw_total_angle*AtR,INS_attitude->yaw_speed);
	//VMC_joint_left.T_turn = turn_t;
	//VMC_joint_right.T_turn = turn_t;	
}
