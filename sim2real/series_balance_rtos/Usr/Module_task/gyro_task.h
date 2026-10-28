#ifndef __GYRO_TASK_H_
#define __GYRO_TASK_H_


void Gyrodata_task(void *param);
void HI04data_task(void *param);

extern float bmi088_ay;//向心力分量补偿
extern float bmi088_ax;//向心力分量补偿
#endif