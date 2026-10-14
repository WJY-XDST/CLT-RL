#include "A1_stm32.h"
#include "vmc.h"
#include "mpu6050.h"
#include "arm_math.h"
#include "WBR.h"
#include "HI04.h"
#include "additional_con.h"
#include "INS_task.h"
#include "all_init.h"
#include "LK_Motor.h"
#include "motor.h"


VMC_joint VMC_joint_left;//左五连杆
VMC_joint VMC_joint_right;//右五连杆

/*腿部连杆长度初始化*/
void link_length_init(VMC_joint *vmc_joint_left,VMC_joint *vmc_joint_right)
{
	vmc_joint_left->link_length[0]=0.21;
	vmc_joint_left->link_length[1]=0.25;
	vmc_joint_left->link_length[2]=0.25;
	vmc_joint_left->link_length[3]=0.21;
	vmc_joint_left->link_length[4]=0;//以L5中点为原点
	
	vmc_joint_right->link_length[0]=0.21;
	vmc_joint_right->link_length[1]=0.25;
	vmc_joint_right->link_length[2]=0.25;
	vmc_joint_right->link_length[3]=0.21;
	vmc_joint_right->link_length[4]=0;
	
  vmc_joint_left->roll_flag=-1;
	vmc_joint_right->roll_flag=1;

	// VMC_joint_left.theta1_init=DMotor[1].Param.EcdOffset;
	// VMC_joint_left.theta2_init=DMotor[3].Param.EcdOffset;
	// VMC_joint_left.theta2_init=DMotor[0].Param.EcdOffset;
	// VMC_joint_left.theta2_init=DMotor[2].Param.EcdOffset;

}

//0 3
//1 2
/*计算四个关节角度*/
void get_joint_angle_left()
{  
     VMC_joint_left.theta2=-AtR*lkmotor[1].angle;
     VMC_joint_left.theta1=AtR*(180-lkmotor[0].angle);
}
void get_joint_angle_right()
{  
     VMC_joint_right.theta1=AtR*(180+lkmotor[3].angle);
     VMC_joint_right.theta2=AtR*(lkmotor[2].angle);
}
/*计算五连杆参数(DSP) */	
void get_l_theta(VMC_joint* vmc)
{
    float xb,yb,xd,yd,lbd,A0,B0,C0,xc,yc,Sin,Cos,u_sqrt;
	Cos=arm_cos_f32(vmc->theta1);
	Sin=arm_sin_f32(vmc->theta1);
	xb=Cos*vmc->link_length[0];
	yb=Sin*vmc->link_length[0];
  xd=vmc->link_length[4]+vmc->link_length[3]* arm_cos_f32(vmc->theta2);
	yd = vmc->link_length[3] * arm_sin_f32(vmc->theta2);
	arm_sqrt_f32(((xd - xb)*(xd - xb)+(yd - yb)*(yd - yb)),&lbd);
	A0 = 2 * vmc->link_length[1] * (xd - xb);
	B0 = 2 * vmc->link_length[1]  * (yd - yb);
	C0 = vmc->link_length[1]*vmc->link_length[1] + lbd*lbd - vmc->link_length[2]*vmc->link_length[2];
	arm_sqrt_f32(A0*A0 + B0*B0 - C0*C0,&u_sqrt);
	vmc->u2 = 2 * atan2((B0 + u_sqrt),(A0 + C0));
	vmc->u3=atan2((yb-yd+vmc->link_length[1]*arm_sin_f32(vmc->u2)),(xb-xd+vmc->link_length[1]*arm_cos_f32(vmc->u2)));
	xc = xb + vmc->link_length[1] * arm_cos_f32(vmc->u2);
	yc = yb + vmc->link_length[1] * arm_sin_f32(vmc->u2);
	vmc->u0=atan2(yc,xc-vmc->link_length[4]/2);

	vmc->theta0=-(vmc->u0-1.57)+(INS_attitude->pitch+pitch_com)*AtR;

    
	vmc->u0_dot = (vmc->u0-vmc->u0_last)*500;

	vmc->theta0_speed= -vmc->u0_dot + INS_attitude->pitch_speed;

	vmc->u0_last=vmc->u0;
	arm_sqrt_f32(((xc-vmc->link_length[4]/2)*(xc-vmc->link_length[4]/2)+yc*yc),&vmc->L);
	vmc->L_dot = (vmc->L - vmc->L_last)*500;
	vmc->L_last = vmc->L;
  if(vmc->theta0>1.1) vmc->theta0-=2*pi;
//  if(vmc->theta0>pi/2) vmc->theta0=pi/2;

}
//
/*计算虚拟连杆模型参数*/
void VMC_Count(VMC_joint* vmc1,VMC_joint* vmc2)//VMC
{
  get_joint_angle_left();
	get_joint_angle_right();
	get_l_theta(vmc1);
	get_l_theta(vmc2);
}
/*matlab
j11 = (l1*sin(phi0-phi3)*sin(phi1-phi2))/sin(phi3-phi2);
j12 = (l1*cos(phi0-phi3)*sin(phi1-phi2))/L0*sin(phi3-phi2);
j21 = (l4*sin(phi0-phi2)*sin(phi3-phi4))/sin(phi3-phi2);
j22 = (l4*cos(phi0-phi2)*sin(phi3-phi4))/L0*sin(phi3-phi2);
J = [j11 j12; j21 j22];
F = [F0;Tp];
T = J * F;
T1 = T(1,1);
T2 = T(2,1);
*/
float J11,J12,J21,J22;
/*雅可比矩阵求解各关节输出*/
void F_Calc(VMC_joint* vmc)
{

	J11 = ((vmc->link_length[0]*arm_sin_f32(vmc->u0-vmc->u3)*arm_sin_f32(vmc->theta1-vmc->u2))/arm_sin_f32(vmc->u3-vmc->u2));
	J12 = ((vmc->link_length[0]*arm_cos_f32(vmc->u0-vmc->u3)*arm_sin_f32(vmc->theta1-vmc->u2))/(vmc->L*arm_sin_f32(vmc->u3-vmc->u2)));
	J21 = ((vmc->link_length[3]*arm_sin_f32(vmc->u0-vmc->u2)*arm_sin_f32(vmc->u3-vmc->theta2))/arm_sin_f32(vmc->u3-vmc->u2));
	J22 = ((vmc->link_length[3]*arm_cos_f32(vmc->u0-vmc->u2)*arm_sin_f32(vmc->u3-vmc->theta2))/(vmc->L*arm_sin_f32(vmc->u3-vmc->u2)));
		vmc->testf = J11 * vmc->F0+ J12 * vmc->TP;
		vmc->testb = J21 * vmc->F0+ J22 * vmc->TP;
  if(lqr_reset_flag==1)
	{
		vmc->torque_f = 0;
		vmc->torque_b = 0;
    if(vmc->TP>0 || pitch_ready == 0)
    vmc->reset_speed=350;
    if(vmc->TP<0)
    vmc->reset_speed=-350;

  if(vmc==&VMC_joint_right && leg_right_ready==1 && leg_left_ready==0 && pitch_ready == 1)
    vmc->reset_speed=0;
  if(vmc==&VMC_joint_left && leg_left_ready==1 && leg_right_ready==0 && pitch_ready == 1)
    vmc->reset_speed=0;
  }

  if(lqr_flag==1)
	{ 
    vmc->reset_speed=0;
    vmc->reset_speed_kd=0;

		vmc->torque_f = J11 * vmc->F0+ J12 * vmc->TP;
		vmc->torque_b = J21 * vmc->F0+ J22 * vmc->TP;

	}
  if(lqr_flag==0&&lqr_reset_flag==0)
	{ 
    vmc->slow_flag=0;
		vmc->torque_f = 0;
		vmc->torque_b = 0;
	}
}




void mySaturate(float *in,float min,float max)
{
  if(*in < min)
  {
    *in = min;
  }
  else if(*in > max)
  {
    *in = max;
  }
}