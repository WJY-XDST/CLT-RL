
#include "motor.h"
#include "fdcan.h"
#include "can_base.h"
/**
 * @brief  ß_ÃîëŠ™CÄ£Ê½
 */
typedef enum 
{
	MIT,
	POSITIONSPEED,
	SPEED,
	CURRENT
}DMode_t;
	
/**
 * @brief  ß_ÃîëŠ™C½Y˜‹ów
 */
typedef struct
{
	uint16_t id;
	uint16_t online_cnt;
	int16_t torque_current;
    uint8_t temperature;
	uint8_t error;
	float angle;
	float torque;
	float speed_rpm;
  float ecd;
  float omega;
  float ecd_offset;
	int32_t motor_output;
	CAN_TxBuffer txBufferforEnable;
	CAN_TxBuffer txBufferforDisable;
	CAN_TxBuffer txBufferforMitMode;
	CAN_TxBuffer txBufferforPositionSpeed;
	CAN_TxBuffer txBufferforSpeed;
	CAN_TxBuffer txBufferforCurrent;
	CAN_TxBuffer rxBufferforCurrent;
	DMode_t mode;
	struct
	{
		float kp;
		float kd;
	}MitMode;
}DMiao_t;

extern DMiao_t dmmotor[4];
extern Motor DMotor[4];

void DMiaoInit(DMiao_t *damiao,  uint16_t canid, uint16_t masterid, DMode_t mode,float ecd_offset);
void DMiaoMitControl(DMiao_t *damiao, float position, float speed, float kp, float kd, float torque);
void DMiaoPositionSpeedControl(DMiao_t *damiao, float position, float speed);
void DMiaoSpeedControl(DMiao_t *damiao, float speed);
void DMiaoCurrentControl(DMiao_t *damiao, int16_t current0,int16_t current1,int16_t current2,int16_t current3);
void DMiao_Enable(CAN_Object can, DMiao_t *damiao);
void DMiao_Disable(CAN_Object can, DMiao_t *damiao);
void DMiao_CanOutput(CAN_Object can, DMiao_t *damiao);
void DMiao_CanUpdata(CAN_Object canx,CAN_RxBuffer can, DMiao_t *damiao);
void angletoangle(float *angle, float ecd_offset);

