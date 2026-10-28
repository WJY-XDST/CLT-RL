#include "A1.h"
#include <string.h>

//A1 通信协议文件

COMData32 crc32_core(COMData32* ptr, COMData32 len); 


/*发送数据填充*/
void modify_data(MOTOR_send* pMotor_s,unsigned char* pBuffer)
{
	pMotor_s->hex_len = 34;
	pMotor_s->motor_send_data.head.start[0] = 0xFE;
	pMotor_s->motor_send_data.head.start[1] = 0xEE;
	pMotor_s->motor_send_data.head.motorID = pMotor_s->id;
	pMotor_s->motor_send_data.head.reserved = 0x0;

	pMotor_s->motor_send_data.Mdata.mode = pMotor_s->mode;
	pMotor_s->motor_send_data.Mdata.ModifyBit = 0xFF;
	pMotor_s->motor_send_data.Mdata.ReadBit = 0x0;
	pMotor_s->motor_send_data.Mdata.reserved = 0x0;
	pMotor_s->motor_send_data.Mdata.Modify = 0;
	pMotor_s->motor_send_data.Mdata.T = pMotor_s->T*256;
	pMotor_s->motor_send_data.Mdata.W = pMotor_s->W*128;
	pMotor_s->motor_send_data.Mdata.Pos = (int)((pMotor_s->Pos/6.2832)*16384.0);
	pMotor_s->motor_send_data.Mdata.K_P = pMotor_s->K_P*2048;
	pMotor_s->motor_send_data.Mdata.K_W = pMotor_s->K_W*1024;      

	pMotor_s->motor_send_data.Mdata.LowHzMotorCmdIndex = 0;
	pMotor_s->motor_send_data.Mdata.LowHzMotorCmdByte = 0;
//	pMotor_s->motor_send_data.Mdata.Res[0] = pMotor_s->Res;
	pMotor_s->motor_send_data.Mdata.Res[0] = 0;    //这个数据位没有用到，可以忽略，所以强制为0
	pMotor_s->motor_send_data.CRCdata = crc32_core((COMData32*)(&(pMotor_s->motor_send_data)), 7);
	
	memcpy(pBuffer,&(pMotor_s->motor_send_data),34);
	
}

unsigned char check_data_forID(unsigned char* pBuffer)
{
	COMData32 temp_32;
	
	//CRC
    temp_32	= pBuffer[74]|(pBuffer[75]<<8)|(pBuffer[76]<<16)|(pBuffer[77]<<24);
	
	if(temp_32 !=crc32_core((uint32_t*)pBuffer,18))
		return 10;//不可能出现的ID 即 数据包校验不成功
	else
		return pBuffer[2];//返回电机ID
}


/*解读数据*/
//数据包校验正确后方可调用
void extract_data(MOTOR_recv* pMotor_r,unsigned char* pBuffer)
{	
	COMData32 temp_32;

	pMotor_r->correct = 1;
		
	//>heat
	pMotor_r->motor_recv_data.head.start[0] = pBuffer[0];
	pMotor_r->motor_recv_data.head.start[1] = pBuffer[1];
    pMotor_r->motor_recv_data.head.motorID  = pBuffer[2];
	pMotor_r->motor_recv_data.head.reserved  = pBuffer[3];
	
	//Data
	pMotor_r->motor_recv_data.Mdata.mode = pBuffer[4];
	pMotor_r->motor_recv_data.Mdata.ReadBit = pBuffer[5];
	
	pMotor_r->motor_recv_data.Mdata.Temp = pBuffer[6];
	pMotor_r->motor_recv_data.Mdata.ReadBit = pBuffer[7];
	
  temp_32	= pBuffer[8]|(pBuffer[9]<<8)|(pBuffer[10]<<16)|(pBuffer[11]<<24);
	pMotor_r->motor_recv_data.Mdata.Read = temp_32;
	
	
	pMotor_r->motor_recv_data.Mdata.T = pBuffer[12]|(pBuffer[13]<<8);
	pMotor_r->motor_recv_data.Mdata.W = pBuffer[14]|(pBuffer[15]<<8);
	
  temp_32	= pBuffer[16]|(pBuffer[17]<<8)|(pBuffer[18]<<16)|(pBuffer[19]<<24);
	pMotor_r->motor_recv_data.Mdata.LW = temp_32;
	
	pMotor_r->motor_recv_data.Mdata.W2 = pBuffer[20]|(pBuffer[21]<<8);
	
  temp_32	= pBuffer[22]|(pBuffer[23]<<8)|(pBuffer[24]<<16)|(pBuffer[25]<<24);
	pMotor_r->motor_recv_data.Mdata.LW2 = temp_32;

	pMotor_r->motor_recv_data.Mdata.Acc = pBuffer[26]|(pBuffer[27]<<8);
	pMotor_r->motor_recv_data.Mdata.OutAcc = pBuffer[28]|(pBuffer[29]<<8);
	
	pMotor_r->motor_recv_data.Mdata.Pos 	= pBuffer[30]|(pBuffer[31]<<8)|(pBuffer[32]<<16)|(pBuffer[33]<<24);
  pMotor_r->motor_recv_data.Mdata.Pos2	= pBuffer[34]|(pBuffer[35]<<8)|(pBuffer[36]<<16)|(pBuffer[37]<<24);
	
	pMotor_r->motor_recv_data.Mdata.gyro[0] = pBuffer[38]|(pBuffer[39]<<8);
	pMotor_r->motor_recv_data.Mdata.gyro[1] = pBuffer[40]|(pBuffer[41]<<8);	
	pMotor_r->motor_recv_data.Mdata.gyro[2] = pBuffer[42]|(pBuffer[43]<<8);

	pMotor_r->motor_recv_data.Mdata.acc[0] = pBuffer[44]|(pBuffer[45]<<8);
	pMotor_r->motor_recv_data.Mdata.acc[1] = pBuffer[46]|(pBuffer[47]<<8);
	pMotor_r->motor_recv_data.Mdata.acc[2] = pBuffer[48]|(pBuffer[49]<<8);
	
	pMotor_r->motor_recv_data.Mdata.Fgyro[0] = pBuffer[50]|(pBuffer[51]<<8);
	pMotor_r->motor_recv_data.Mdata.Fgyro[1] = pBuffer[52]|(pBuffer[53]<<8);
	pMotor_r->motor_recv_data.Mdata.Fgyro[2] = pBuffer[54]|(pBuffer[55]<<8);
	
  pMotor_r->motor_recv_data.Mdata.Facc[0] = pBuffer[56]|(pBuffer[57]<<8);
	pMotor_r->motor_recv_data.Mdata.Facc[1] = pBuffer[58]|(pBuffer[59]<<8);
	pMotor_r->motor_recv_data.Mdata.Facc[2] = pBuffer[60]|(pBuffer[61]<<8);
	
	
	pMotor_r->motor_recv_data.Mdata.Fmag[0] = pBuffer[62]|(pBuffer[63]<<8);
	pMotor_r->motor_recv_data.Mdata.Fmag[1] = pBuffer[64]|(pBuffer[65]<<8);
	pMotor_r->motor_recv_data.Mdata.Fmag[2] = pBuffer[66]|(pBuffer[67]<<8);
	
	pMotor_r->motor_recv_data.Mdata.Ftemp = pBuffer[68];
	
	pMotor_r->motor_recv_data.Mdata.Force16 = pBuffer[69]|(pBuffer[70]<<8);
	
	pMotor_r->motor_recv_data.Mdata.Force8 = pBuffer[71];
	
	pMotor_r->motor_recv_data.Mdata.FError = pBuffer[72];
	
	pMotor_r->motor_recv_data.Mdata.Res[0] = pBuffer[73];
	
	pMotor_r->motor_id = pMotor_r->motor_recv_data.head.motorID;
	pMotor_r->mode = pMotor_r->motor_recv_data.Mdata.mode;
	pMotor_r->Temp = pMotor_r->motor_recv_data.Mdata.Temp;
	pMotor_r->MError = pMotor_r->motor_recv_data.Mdata.MError;
	pMotor_r->T = ((float)pMotor_r->motor_recv_data.Mdata.T) / 256;
	pMotor_r->W = ((float)pMotor_r->motor_recv_data.Mdata.W) / 128;
	pMotor_r->LW = pMotor_r->motor_recv_data.Mdata.LW;

	pMotor_r->Acc = (int)pMotor_r->motor_recv_data.Mdata.Acc;
	pMotor_r->Pos = 6.2832*((float)pMotor_r->motor_recv_data.Mdata.Pos) / 16384;
	
	pMotor_r->gyro[0] = ((float)pMotor_r->motor_recv_data.Mdata.gyro[0]) * 0.00107993176;
	pMotor_r->gyro[1] = ((float)pMotor_r->motor_recv_data.Mdata.gyro[1]) * 0.00107993176;
	pMotor_r->gyro[2] = ((float)pMotor_r->motor_recv_data.Mdata.gyro[2]) * 0.00107993176;

	pMotor_r->acc[0] = ((float)pMotor_r->motor_recv_data.Mdata.acc[0]) * 0.0023911132;
	pMotor_r->acc[1] = ((float)pMotor_r->motor_recv_data.Mdata.acc[1]) * 0.0023911132;
	pMotor_r->acc[2] = ((float)pMotor_r->motor_recv_data.Mdata.acc[2]) * 0.0023911132;
	
//	pMotor_r->angle = pMotor_r->Pos * 57.3f / 9.1f;  //< 转子角度才需要除以9.1，实际使用出轴角度
	pMotor_r->angle = pMotor_r->Pos * 57.3f;
}




//CRC校验函数
COMData32 crc32_core(COMData32* ptr, COMData32 len)
{
	COMData32 xbit = 0;
	COMData32 data = 0;
	COMData32 CRC32 = 0xFFFFFFFF;
	const COMData32 dwPolynomial = 0x04c11db7;
	for (COMData32 i = 0; i < len; i++)
	{
		xbit = 1 << 31;
		data = ptr[i];
		for (COMData32 bits = 0; bits < 32; bits++)
		{
			if (CRC32 & 0x80000000)
			{
				CRC32 <<= 1;
				CRC32 ^= dwPolynomial;
			}
			else
				CRC32 <<= 1;
			if (data & xbit)
				CRC32 ^= dwPolynomial;

			xbit >>= 1;
		}
	}
	return CRC32;
}