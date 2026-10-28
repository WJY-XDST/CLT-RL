#include "HI04.h"
#include "all_init.h"

 /*和freertos相关*/
#include "freertos.h"
#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"
HI04_data HI04_data_t;


uint8_t HI04_recData[HI04_rxBufferLengh]__attribute__((at(0x24006800)));
static void crc16_update(uint16_t *currectCrc, const uint8_t *src, uint32_t );
void HI04_modify(int8_t init_check);


UART_RxBuffer uart8_buffer={
	.Data = HI04_recData,
	.Size = HI04_rxBufferLengh
};
uint8_t HI04_callback(uint8_t *recBuffer, uint16_t len)
{
	xQueueSendFromISR(uart8.g_xQueueUart, &(recBuffer), 0);
	return 0;

}
uint8_t HI04_data_receive(uint8_t * recBuffer)
{
	uint16_t payload_len;
	uint16_t crc;
	crc = 0;
	payload_len = recBuffer[2] + (recBuffer[3] << 8);
	/* calulate 5A A5 and LEN filed crc */
	crc16_update(&crc, recBuffer, 4);
	/* calulate payload crc */
	crc16_update(&crc, recBuffer + 6, payload_len);
	HI04_data_t.tag = recBuffer[6];
	if((crc ==recBuffer[4] + (recBuffer[5] << 8)) &&HI04_data_t.tag == 0x92)
	{
		HI04_data_t.status    = recBuffer[7]| (recBuffer[8] << 8);
		HI04_data_t.temperate = recBuffer[9];
		HI04_data_t.pps       = recBuffer[10]| (recBuffer[11] << 8);
		HI04_data_t.air_press = recBuffer[12]| (recBuffer[13] << 8);
		HI04_data_t.heave     = recBuffer[14]| (recBuffer[15] << 8);
		HI04_data_t.gyr_b[0]  = recBuffer[16]| (recBuffer[17] << 8);
		HI04_data_t.gyr_b[1]  = recBuffer[18]| (recBuffer[19] << 8);
		HI04_data_t.gyr_b[2]  = recBuffer[20]| (recBuffer[21] << 8);
		HI04_data_t.acc_b[0]  = recBuffer[22]| (recBuffer[23] << 8);
		HI04_data_t.acc_b[1]  = recBuffer[24]| (recBuffer[25] << 8);
		HI04_data_t.acc_b[2]  = recBuffer[26]| (recBuffer[27] << 8);
		HI04_data_t.mag_b[0]  = recBuffer[28]| (recBuffer[29] << 8);
		HI04_data_t.mag_b[1]  = recBuffer[30]| (recBuffer[31] << 8);
		HI04_data_t.mag_b[2]  = recBuffer[32]| (recBuffer[33] << 8);
		HI04_data_t.roll      = recBuffer[34]| (recBuffer[35] << 8)| (recBuffer[36] << 16)| (recBuffer[37] << 24);
		HI04_data_t.pitch     = recBuffer[38]| (recBuffer[39] << 8)| (recBuffer[40] << 16)| (recBuffer[41] << 24);
		HI04_data_t.yaw       = recBuffer[42]| (recBuffer[43] << 8)| (recBuffer[44] << 16)| (recBuffer[45] << 24);
		HI04_data_t.quat[0]   = recBuffer[46]| (recBuffer[47] << 8);
		HI04_data_t.quat[1]   = recBuffer[48]| (recBuffer[49] << 8);
		HI04_data_t.quat[2]   = recBuffer[50]| (recBuffer[51] << 8);
		HI04_data_t.quat[3]   = recBuffer[52]| (recBuffer[53] << 8);
	}
	
}


float int16_float(int16_t *input ,float c)
{
	float output;
	output =(float)(*input) * c;
	return output;
}

float  int32_float(int32_t *input ,float c)
{
	float output;
	output =(float)(*input) * c;
	return output;
}

void HI04_modify(int8_t init_check)
{
	
	HI04_data_t.gyro_data.roll_speed = int16_float(&HI04_data_t.gyr_b[1],0.001);
	//HI04_data_t.gyro_data.pitch_speed = int16_float(&HI04_data_t.gyr_b[0],0.001);/*很抖*/
	HI04_data_t.gyro_data.yaw_speed = int16_float(&HI04_data_t.gyr_b[2],0.001);
	HI04_data_t.gyro_data.roll  = int32_float(&HI04_data_t.roll,0.001);
	HI04_data_t.gyro_data.pitch = int32_float(&HI04_data_t.pitch,0.001);
	HI04_data_t.gyro_data.pitch_speed = (HI04_data_t.gyro_data.pitch - HI04_data_t.gyro_data.pitch_last)*500*AtR;
	HI04_data_t.gyro_data.pitch_last = HI04_data_t.gyro_data.pitch;
	HI04_data_t.gyro_data.yaw   =  int32_float(&HI04_data_t.yaw,0.001);
	HI04_data_t.accel[0] = int16_float(&HI04_data_t.acc_b[0],0.0048828);
	HI04_data_t.accel[1] = int16_float(&HI04_data_t.acc_b[1],0.0048828);
	HI04_data_t.accel[2] = int16_float(&HI04_data_t.acc_b[2],0.0048828);
	if(init_check == 0)
    HI04_data_t.gyro_data.yaw_last = HI04_data_t.gyro_data.yaw; 
	else
	{
		if(HI04_data_t.gyro_data.yaw_last - HI04_data_t.gyro_data.yaw > 200)
		HI04_data_t.gyro_data.yaw_total += HI04_data_t.gyro_data.yaw +180;
	else if(HI04_data_t.gyro_data.yaw_last - HI04_data_t.gyro_data.yaw <-200)
		HI04_data_t.gyro_data.yaw_total += HI04_data_t.gyro_data.yaw -180;
	else
      HI04_data_t.gyro_data.yaw_total +=HI04_data_t.gyro_data.yaw - HI04_data_t.gyro_data.yaw_last;
}
	HI04_data_t.gyro_data.yaw_last = HI04_data_t.gyro_data.yaw;
	
}



/*
 currectCrc: previous crc value, set 0 if it's first section
 src: source stream data
 lengthInBytes: length
*/
static void crc16_update(uint16_t *currectCrc, const uint8_t *src, uint32_t lengthInBytes)
{
  uint32_t crc = *currectCrc;
  uint32_t j;
  for (j=0; j < lengthInBytes; ++j)
  {
  uint32_t i;
  uint32_t byte = src[j];
 crc ^= byte << 8;
  for (i = 0; i < 8; ++i)
  {
  uint32_t temp = crc << 1;
  if (crc & 0x8000)
  {
 temp ^= 0x1021;
  }
  crc = temp;
  }
  } 
  *currectCrc = crc;
}

