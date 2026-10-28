#include "DMmotor.h"
#include "user_lib.h"
#include "all_init.h"

/**
 * @brief  達妙電機初始化
 */
DMiao_t dmmotor[4];
Motor DMotor[4];
void DMiaoInit(DMiao_t *damiao,  uint16_t canid, uint16_t masterid, DMode_t mode, float ecd_offset)
{
	damiao->id   = masterid;
	damiao->mode = mode;
	damiao->ecd_offset=ecd_offset;
	damiao->txBufferforEnable.Identifier        = canid;
	damiao->txBufferforDisable.Identifier       = canid;
	damiao->txBufferforMitMode.Identifier       = canid;
	damiao->txBufferforPositionSpeed.Identifier = 0x100+canid;
	damiao->txBufferforSpeed.Identifier         = 0x200+canid;
	damiao->txBufferforCurrent.Identifier       = 0x3FE;
	
	damiao->txBufferforEnable.txHeader.DataLength  = FDCAN_DLC_BYTES_8;
	damiao->txBufferforDisable.txHeader.DataLength = FDCAN_DLC_BYTES_8;
	switch(damiao->mode)
	{
		case MIT:
		case POSITIONSPEED:
		{
			damiao->txBufferforMitMode.txHeader.DataLength = FDCAN_DLC_BYTES_8;
			break;
		};
		case SPEED:
		{
			damiao->txBufferforSpeed.txHeader.DataLength = FDCAN_DLC_BYTES_4;
			break;
		};
		case CURRENT:
		{
			damiao->txBufferforCurrent.txHeader.DataLength = FDCAN_DLC_BYTES_2;
			break;
		};
		default:;	
	}
		damiao->txBufferforEnable.Data[0] = 0xFF;
		damiao->txBufferforEnable.Data[1] = 0xFF;
		damiao->txBufferforEnable.Data[2] = 0xFF;
		damiao->txBufferforEnable.Data[3] = 0xFF;
		damiao->txBufferforEnable.Data[4] = 0xFF;
		damiao->txBufferforEnable.Data[5] = 0xFF;
		damiao->txBufferforEnable.Data[6] = 0xFF;
		damiao->txBufferforEnable.Data[7] = 0xFC;
	
		damiao->txBufferforDisable.Data[0] = 0xFF;
		damiao->txBufferforDisable.Data[1] = 0xFF;
		damiao->txBufferforDisable.Data[2] = 0xFF;
		damiao->txBufferforDisable.Data[3] = 0xFF;
		damiao->txBufferforDisable.Data[4] = 0xFF;
		damiao->txBufferforDisable.Data[5] = 0xFF;
		damiao->txBufferforDisable.Data[6] = 0xFF;
		damiao->txBufferforDisable.Data[7] = 0xFD;
}

/**
 * @brief  達妙電機MIT模式
 */
void DMiaoMitControl(DMiao_t *damiao, float position, float speed, float kp, float kd, float torque)
{
	static uint16_t i_position, i_speed, i_kp, i_kd, i_torque;
	
	i_position = float_to_uint(position, -3.14159f, 3.14159f, 16);
	i_speed    = float_to_uint(speed, -45.0f, 45.0f, 12);
	i_kp       = float_to_uint(kp, 0, 500.0f, 12);
	i_kd       = float_to_uint(kd, 0, 5.0f, 12);
	i_torque   = float_to_uint(torque, -54.0f, 54.0f, 12);
	
	damiao->txBufferforMitMode.Data[0] = (i_position >> 8);
	damiao->txBufferforMitMode.Data[1] = i_position;
	damiao->txBufferforMitMode.Data[2] = (i_speed >> 4);
	damiao->txBufferforMitMode.Data[3] = ((i_speed&0xF)<<4)|(i_kp>>8);
	damiao->txBufferforMitMode.Data[4] = i_kp;
	damiao->txBufferforMitMode.Data[5] = (i_kd >> 4);
	damiao->txBufferforMitMode.Data[6] = ((i_kd&0xF)<<4)|(i_torque>>8);
	damiao->txBufferforMitMode.Data[7] = i_torque;
}

/**
 * @brief  達妙電機位置速度模式
 */
void DMiaoPositionSpeedControl(DMiao_t *damiao, float position, float speed)
{
	static uint8_t *i_position, *i_speed;
	
	i_position=(uint8_t*)&position;
	i_speed=(uint8_t*)&speed;
	
	damiao->txBufferforPositionSpeed.Data[0] = *i_position;
	damiao->txBufferforPositionSpeed.Data[1] = *(i_position+1);
	damiao->txBufferforPositionSpeed.Data[2] = *(i_position+2);
	damiao->txBufferforPositionSpeed.Data[3] = *(i_position+3);
	damiao->txBufferforPositionSpeed.Data[4] = *i_speed;
	damiao->txBufferforPositionSpeed.Data[5] = *(i_speed+1);
	damiao->txBufferforPositionSpeed.Data[6] = *(i_speed+2);
	damiao->txBufferforPositionSpeed.Data[7] = *(i_speed+3);
}

/**
 * @brief  達妙電機速度模式
 */
void DMiaoSpeedControl(DMiao_t *damiao, float speed)
{
	static uint8_t *i_speed;
	
	i_speed=(uint8_t*)&speed;
	
	damiao->txBufferforSpeed.Data[0] = *i_speed;
	damiao->txBufferforSpeed.Data[1] = *(i_speed+1);
	damiao->txBufferforSpeed.Data[2] = *(i_speed+2);
	damiao->txBufferforSpeed.Data[3] = *(i_speed+3);
}

/**
 * @brief  達妙電機電流模式
 */
void DMiaoCurrentControl(DMiao_t *damiao,int16_t current0,int16_t current1,int16_t current2,int16_t current3)
{
	damiao->txBufferforCurrent.Data[0] =  current0;
	damiao->txBufferforCurrent.Data[1] = (current0 >> 8);
	damiao->txBufferforCurrent.Data[2] =  current1;
	damiao->txBufferforCurrent.Data[3] = (current1 >> 8);	
  damiao->txBufferforCurrent.Data[4] =  current2;
	damiao->txBufferforCurrent.Data[5] = (current2 >> 8);	
  damiao->txBufferforCurrent.Data[6] =  current3;
	damiao->txBufferforCurrent.Data[7] = (current3 >> 8);	

}
/**
 * @brief  達妙電機輸出函數
 */
void DMiao_CanOutput(CAN_Object can, DMiao_t *damiao)
{
	switch(damiao->mode)
	{
		case MIT: 
		{
			CAN_Send(&can, &damiao->txBufferforMitMode);
			break;
		}
		case POSITIONSPEED: 
		{
			CAN_Send(&can, &damiao->txBufferforPositionSpeed);
			break;
		}
		case SPEED: 
		{
			CAN_Send(&can, &damiao->txBufferforSpeed);
			break;
		}
		case CURRENT:
		{
			CAN_Send(&can, &damiao->txBufferforCurrent);
			break;
		}
		default:;
	}
}	

/**
 * @brief  達妙電機使能函數
 */
void DMiao_Enable(CAN_Object can, DMiao_t *damiao)
{
	CAN_Send(&can, &damiao->txBufferforEnable);
}
	
/**
 * @brief  達妙電機失能函數
 */
void DMiao_Disable(CAN_Object can, DMiao_t *damiao)
{
	CAN_Send(&can, &damiao->txBufferforDisable);
}

/**
 * @brief  達妙電機反饋處理函數
 */
void DMiao_CanUpdata(CAN_Object canx,CAN_RxBuffer can, DMiao_t *damiao)
{
	switch(damiao->mode)
	{
		static uint16_t i_position[4], i_speed[4], i_torque[4];
    static uint8_t i = 0; 
		case MIT:
		case POSITIONSPEED: 
		case SPEED:
		{	
 //get motor id 
    i = can.Header.Identifier; 
    dmmotor[i].error   	= (can.Data[0]>>4);
		i_position[i] 			= ((can.Data[2])|(can.Data[1]<<8));
		i_speed[i]   			  = ((can.Data[4]|(can.Data[3]<<8))>>4);
	  i_torque[i]   			= (can.Data[5])|((can.Data[4]&0x0f)<<8);
					
				dmmotor[i].angle       = (uint_to_float(i_position[i], -3.1415926f, 3.1415926f, 16))/AtR; // (-12.5,12.5)
        angletoangle(&dmmotor[i].angle,dmmotor[i].ecd_offset);
				dmmotor[i].speed_rpm   = uint_to_float(i_speed[i], -45.0f, 45.0f, 12);// (-45.0,45.0)
				dmmotor[i].torque      = uint_to_float(i_torque[i], -18.0f, 18.0f, 12); // (-18.0,18.0)
			break;
		}
		case CURRENT:	
		{	
			if(can.Header.Identifier == damiao->id)
			{
				damiao->ecd            = ((can.Data[0] << 8)|(can.Data[1]));
				damiao->angle          = (damiao->ecd / 8192.0f * 360.0f)-180.0f;
				damiao->omega          = ((can.Data[2] << 8)|(can.Data[3]));
				damiao->speed_rpm      = damiao->omega / 100.0f;
				damiao->torque_current = (can.Data[4] << 8)|(can.Data[5]);
				damiao->torque_current = (damiao->torque_current / 16384.0f) * 20.0f;
				damiao->temperature    = (float)(can.Data[6]);
				damiao->error 		   = (can.Data[7]);
			}
			break;
		}
		default:;
	}
	damiao->online_cnt = 0;
//	if((damiao->angle < -100) && (damiao->last_angle > 100))
//		damiao->total_angle += 360  + damiao->angle - damiao->last_angle;
//	else if(( damiao->angle > 100) && (damiao->last_angle < -100))
//		damiao->total_angle += -360 + damiao->angle - damiao->last_angle;
//	else 
//		damiao->total_angle +=damiao->angle-damiao->last_angle;
//	damiao->last_angle = damiao->angle;
//	damiao->axis_angle = damiao->total_angle/2.0f;
}
void angletoangle(float *angle, float ecd_offset)
{
  float angle1;
   angle1=*angle;
  if(angle1>180&&angle1<=540)
     angle1-=360;
  else if(angle1>540)
     angle1-=720;
  else if(angle1<-180&&angle1>=-540)
     angle1+=360;
  else if(angle1<-540)
     angle1+=720;

if(angle1-ecd_offset>180)
   angle1-=ecd_offset+360;
else if(angle1-ecd_offset<-180)
   angle1-=ecd_offset-360;
else
   angle1-=ecd_offset;
   *angle=angle1;
}