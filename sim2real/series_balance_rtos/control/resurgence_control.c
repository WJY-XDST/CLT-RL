/**
  *********************************************************************
  * @file      resurgence_control.c/h
  * @brief     此文件存放关于复活可能执行操作的函数
  * @note       
  * @history
  *
  @verbatim
  ==============================================================================

  ==============================================================================
  @endverbatim
  *********************************************************************
  */

#include "DMmotor.h"
#include "vmc.h"
#include "online_check_task.h"
#include "ins.h"

/*电机离线判断*/
int16_t leg_reset_count = 0;//复位执行计数
extern uint16_t leg_reset_judge_count;
/*腿部复位执行函数*/
void leg_reset()
{

}
void series_leg_reset()
{


}