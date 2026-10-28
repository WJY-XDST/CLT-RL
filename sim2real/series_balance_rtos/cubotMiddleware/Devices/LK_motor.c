#include "LK_motor.h"
#include "online_check_task.h"

/*和freertos相关*/
#include "cmsis_os.h"
#include "freertos.h"
#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"


#define EcdFullRange_LK 65535

CAN_TxBuffer txBufferInLKGB={
	.Identifier = 0x280
};

LK_Motor lkmotor[4];

static int torque_to_out(float torque)
{
	int16_t output;
	output = (torque / 1.078) * 2048 / 33;
	if(output > 2048)
		output = 2048;
	if(output < -2048)
		output = -2048;
	return output;
}

void LK_Motor_TorqueControl(LK_Motor* motor)//单电机
{
		motor->txBufferInTortque.Data[4] = *(uint8_t*)(&motor->output);
		motor->txBufferInTortque.Data[5] = *((uint8_t*)(&motor->output) + 1);
}

void LK_Motor_TorqueGBControl(void)//多电机
{
	txBufferInLKGB.Data[0] = *(uint8_t*)(&lkmotor[0].output);
	txBufferInLKGB.Data[1] = *(uint8_t*)((&lkmotor[0].output) + 1);
	txBufferInLKGB.Data[2] = 0;
	txBufferInLKGB.Data[3] = 0;
	txBufferInLKGB.Data[4] = 0;
	txBufferInLKGB.Data[5] = 0;
	txBufferInLKGB.Data[6] = 0;
	txBufferInLKGB.Data[7] = 0;

}
void LK_MotorInit(LK_Motor* motor, uint16_t ecd_Offset,CanNumber canx, uint16_t id)
{

	motor->id         = id;
	motor->ecd_offset = ecd_Offset;

	motor->txBufferforData.Identifier = 0x140 + id;
	motor->txBufferforData.txHeader.DataLength  = FDCAN_DLC_BYTES_8;
	motor->txBufferforData.Data[0] = 0x9C;
	motor->txBufferforData.Data[1] = 0x00;
	motor->txBufferforData.Data[2] = 0x00;
	motor->txBufferforData.Data[3] = 0x00;
	motor->txBufferforData.Data[4] = 0x00;
	motor->txBufferforData.Data[5] = 0x00;
	motor->txBufferforData.Data[6] = 0x00;
	motor->txBufferforData.Data[7] = 0x00;

  motor->txBufferInTortque.Identifier = 0x140 + id;
	motor->txBufferInTortque.txHeader.DataLength  = FDCAN_DLC_BYTES_8;
	motor->txBufferInTortque.Data[0] = 0xA1;
	motor->txBufferInTortque.Data[1] = 0x00;
	motor->txBufferInTortque.Data[2] = 0x00;
	motor->txBufferInTortque.Data[3] = 0x00;
	motor->txBufferInTortque.Data[4] = 0x00;
	motor->txBufferInTortque.Data[5] = 0x00;
	motor->txBufferInTortque.Data[6] = 0x00;
	motor->txBufferInTortque.Data[7] = 0x00;

  motor->txBufferInSpeed.Identifier = 0x140 + id;
	motor->txBufferInSpeed.txHeader.DataLength  = FDCAN_DLC_BYTES_8;
	motor->txBufferInSpeed.Data[0] = 0xA2;
	motor->txBufferInSpeed.Data[1] = 0x00;
	motor->txBufferInSpeed.Data[2] = 0x00;
	motor->txBufferInSpeed.Data[3] = 0x00;
	motor->txBufferInSpeed.Data[4] = 0x00;
	motor->txBufferInSpeed.Data[5] = 0x00;
	motor->txBufferInSpeed.Data[6] = 0x00;
	motor->txBufferInSpeed.Data[7] = 0x00;

}

static void MotorEcdtoAngle(LK_Motor* motor)
{
	if(motor->ecd_offset < (EcdFullRange_LK/2))
	{
		if(motor->ecd > (motor->ecd_offset + EcdFullRange_LK/2))
			motor->ecd = motor->ecd - EcdFullRange_LK;
	}
	else
	{
		if(motor->ecd < (motor->ecd_offset - EcdFullRange_LK/2))
			motor->ecd = motor->ecd + EcdFullRange_LK;
	}
		motor->angle = (360.0 / 65536) * (motor->ecd - motor->ecd_offset);
}

void LK_CanUpdata(CAN_Object canx,CAN_RxBuffer can)
{
    static uint8_t i = 0; 
 //get motor id 
	if(can.Data[0] == 0xA1 || can.Data[0] == 0xA2 || can.Data[0] == 0x9C)//0x9C
	{
							i                 = can.Header.Identifier - 0x141; 
			lkmotor[i].temperature    = can.Data[1];
			lkmotor[i].torque_current = ((int16_t)((can.Data[2])|(can.Data[3]<<8))) * 0.016113f;
			lkmotor[i].speed          = (int16_t)((can.Data[4]|(can.Data[5]<<8)))/6;//1dps
			lkmotor[i].Rawecd         = (can.Data[6])|((can.Data[7])<<8);

			lkmotor[i].ecd            = lkmotor[i].Rawecd;

						MotorEcdtoAngle(&lkmotor[i]);
//					lkmotor[i].angle       = (lkmotor[i].ecd - lkmotor[i].ecd_offset) * 360.0f / EcdFullRange_LK; 
					lkmotor[i].speed_rpm   = (lkmotor[i].speed / 6.0f);
					lkmotor[i].torque      = lkmotor[i].torque_current * 1.078f;
		lkmotor[i].online_cnt = 0;
	}
}

void LK_MotorData_Task(void)
{
	while(1)
	{
LK_MotorFillData(&lkmotor[0],0);
LK_MotorFillData(&lkmotor[1],0);
LK_MotorFillData(&lkmotor[2],0);
LK_MotorFillData(&lkmotor[3],0);

//		LK_Motor_TorqueGBControl();

		LK_Motor_CanOutput(&lkmotor[0]);
//		LK_Motor_GBCanOutput();
		LK_Motor_CanOutput(&lkmotor[1]);
		LK_Motor_CanOutput(&lkmotor[2]);
		LK_Motor_CanOutput(&lkmotor[3]);
	}
}

void LK_MotorFillData(LK_Motor* motor,float torque)
{
	motor->output = torque_to_out(torque);
	LK_Motor_TorqueControl(motor);
}

void LK_Motor_CanOutput(LK_Motor* motor)//单电机
{
			CAN_Send(&can1, &motor->txBufferInTortque);
}	

void LK_MotorFillDataInSpeed(LK_Motor* motor,float speed)//单电机
{
				int32_t speedControl = speed/0.01;
			motor->txBufferInSpeed.Data[2] = (uint8_t)(1800 & 0xFF);
			motor->txBufferInSpeed.Data[3] = (uint8_t)(1800 >> 8);
			motor->txBufferInSpeed.Data[4] = (uint8_t)(speedControl & 0xFF);
			motor->txBufferInSpeed.Data[5] = (uint8_t)((speedControl >> 8) & 0xFF);
			motor->txBufferInSpeed.Data[6] = (uint8_t)((speedControl >> 16) & 0xFF);
			motor->txBufferInSpeed.Data[7] = (uint8_t)((speedControl >> 24) & 0xFF);
}
void LK_Motor_CanOutputInSpeed(LK_Motor* motor)//单电机
{
			CAN_Send(&can1, &motor->txBufferInSpeed);
}
void LK_Motor_GBCanOutput(void)//多电机
{
			CAN_Send(&can1, &txBufferInLKGB);
}

