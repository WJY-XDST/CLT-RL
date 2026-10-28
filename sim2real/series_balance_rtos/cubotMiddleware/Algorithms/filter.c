#include "filter.h"

struct LowPassFilter_Info LPF_pitch_speed={
  .filter_coefficient=1.0f,
	.last_output=0,
};

struct LowPassFilter_Info LPF_yaw_speed={
  .filter_coefficient=0.1f,
	.last_output=0,
};

struct LowPassFilter_Info LPF_pitch_vision={
  .filter_coefficient=1.5f,
	.last_output=0,
};

struct LowPassFilter_Info LPF_yaw_vision={
  .filter_coefficient=1.0f,
	.last_output=0,
};

struct LowPassFilter_Info LPF_none={
  .filter_coefficient=1,
	.last_output=0,
};

struct LowPassFilter_Info LPF_pitch_mpu={
  .filter_coefficient=0.2f,
	.last_output=0,
};

struct LowPassFilter_Info LPF_yaw_mpu={
  .filter_coefficient=0.2f,
	.last_output=0,
};

struct LowPassFilter_Info LPF_XS={
  .filter_coefficient=0.05f,
	.last_output=0,
};

struct LowPassFilter_Info LPF_XSdot={
  .filter_coefficient=0.01f,
	.last_output=0,
};
float LPFilter(float sampling ,struct LowPassFilter_Info *LPF){
	//Ò»½×µÍÍ¨ÂË²¨Æ÷£ºp(n) = c¡¤q(n) + (1 - c)¡¤p(n - 1) 
	(*LPF).sampling =sampling;
	
	(*LPF).output=(*LPF).filter_coefficient *(*LPF).sampling +(1-(*LPF).filter_coefficient)*(*LPF).last_output;
	
	(*LPF).last_output =(*LPF).output ;
	
	return (*LPF).output ;
};
