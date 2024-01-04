#include "display.h"

bool DynaFlashProjector::init()
{
	// apparently this is needed or else the projector will not work. perhaps dynaflash can't tolerate page faults?
	if (!SetProcessWorkingSetSizeEx(::GetCurrentProcess(), (2000UL * 1024 * 1024), (3000UL * 1024 * 1024), QUOTA_LIMITS_HARDWS_MIN_ENABLE))
	{
		std::cout << "SetProcessWorkingSetSize Failed!\n";
		return false;
	}
	/* create a DynaFlash instance */
	pDynaFlash = CreateDynaFlash();
	if (pDynaFlash == NULL)
	{
		return false;
	}

	/* connect to DynaFlash */
	if (pDynaFlash->Connect(board_index) != STATUS_SUCCESSFUL)
	{
		return false;
	}

	/* reset DynaFlash */
	if (pDynaFlash->Reset() != STATUS_SUCCESSFUL)
	{
		gracefully_close();
		return false;
	}

	DYNAFLASH_PARAM tDynaFlash_Param = {0};
	/* projection parameter setting */
	tDynaFlash_Param.dFrameRate = frame_rate;
	// tDynaFlash_Param.dRProportion = (double)2 * 100 / (double)5;
	// tDynaFlash_Param.dGProportion = (double)100 / (double)5;
	// tDynaFlash_Param.dBProportion = (double)2 * 100 / (double)5;
	tDynaFlash_Param.dRProportion = (double)100 / (double)3;
	tDynaFlash_Param.dGProportion = (double)100 / (double)3;
	tDynaFlash_Param.dBProportion = (double)100 / (double)3;
	tDynaFlash_Param.nBinaryMode = frame_mode;
	tDynaFlash_Param.nBitDepth = bit_depth;
	tDynaFlash_Param.nMirrorMode = (m_flip_hor ? 0 : 1);
	tDynaFlash_Param.nFlipMode = (m_flip_ver ? 1 : 0);
	if (pDynaFlash->SetParam(&tDynaFlash_Param) != STATUS_SUCCESSFUL)
	{
		gracefully_close();
		return false;
	}
	// ILLUMINANCE_MODE cur_ilum_mode;
	// if (pDynaFlash->GetIlluminance(&cur_ilum_mode) != STATUS_SUCCESSFUL)
	// {
	// 	gracefully_close();
	// 	return false;
	// }
	// std::cout << "DynaFlash current illuminance mode: " << cur_ilum_mode << std::endl;
	/* Illuminance setting */
	if (pDynaFlash->SetIlluminance(ilum_mode) != STATUS_SUCCESSFUL)
	{
		gracefully_close();
		return false;
	}

	print_led_values();
	// set_led_values();
	// print_led_values();

	/* Get frame buffer for projection */
	if (pDynaFlash->AllocFrameBuffer(alloc_frame_buffer) != STATUS_SUCCESSFUL)
	{
		gracefully_close();
		return false;
	}

	print_version();

	if (pDynaFlash->Start() != STATUS_SUCCESSFUL)
	{
		printf("Start Error\n");
		gracefully_close();
		return false;
	}
	initialized = true;
	return true;
}

void DynaFlashProjector::print_led_values()
{
	unsigned long nDaValue[4];
	pDynaFlash->ReadDACRegister(0x00, &nDaValue[0]);
	pDynaFlash->ReadDACRegister(0x01, &nDaValue[1]);
	pDynaFlash->ReadDACRegister(0x02, &nDaValue[2]);
	pDynaFlash->ReadDACRegister(0x03, &nDaValue[3]);
	double current = ((nDaValue[0]) * (5.0 / 1024.0)) / 0.75;
	std::cout << "green LED current: " << current << std::endl;
	pDynaFlash->ReadDACRegister(0x04, &nDaValue[0]);
	pDynaFlash->ReadDACRegister(0x05, &nDaValue[1]);
	pDynaFlash->ReadDACRegister(0x06, &nDaValue[2]);
	pDynaFlash->ReadDACRegister(0x07, &nDaValue[3]);
	current = ((nDaValue[0]) * (5.0 / 1024.0)) / 0.75;
	std::cout << "red LED current: " << current << std::endl;
	pDynaFlash->ReadDACRegister(0x08, &nDaValue[0]);
	current = ((nDaValue[0]) * (5.0 / 1024.0)) / 0.067;
	std::cout << "blue LED current: " << current << std::endl;
}

void DynaFlashProjector::set_led_values()
{
	double green_current = 0.1f;
	double Vadj = (10 * 0.075 * green_current);
	unsigned long write_value = (int)(Vadj / (5.0 / 1024.0));
	pDynaFlash->WriteDACRegister(0x00, write_value);
	pDynaFlash->WriteDACRegister(0x01, write_value);
	pDynaFlash->WriteDACRegister(0x02, write_value);
	pDynaFlash->WriteDACRegister(0x03, write_value);
	double red_current = 0.1f;
	Vadj = (10 * 0.075 * red_current);
	write_value = (int)(Vadj / (5.0 / 1024.0));
	pDynaFlash->WriteDACRegister(0x04, write_value);
	pDynaFlash->WriteDACRegister(0x05, write_value);
	pDynaFlash->WriteDACRegister(0x06, write_value);
	pDynaFlash->WriteDACRegister(0x07, write_value);
	double blue_current = 2.5f;
	Vadj = (10 * 0.0067 * blue_current);
	write_value = (int)(Vadj / (5.0 / 1024.0));
	pDynaFlash->WriteDACRegister(0x08, write_value);
}

void DynaFlashProjector::gracefully_close()
{
	if (initialized)
	{
		/* first set flag and sleep (let other threads finish) */
		// todo protect shared projector buffer with mutex instead
		initialized = false;
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		/* stop projection */
		pDynaFlash->Stop();

		/* release framebuffer */
		pDynaFlash->ReleaseFrameBuffer();

		/* Float the mirror device */
		pDynaFlash->Float(0);

		/* disconnect the DynaFlash */
		pDynaFlash->Disconnect();

		/* release instance of DynaFlash class */
		ReleaseDynaFlash(&pDynaFlash);
		std::cout << "dynaflash killed." << std::endl;
	}
}

void DynaFlashProjector::print_version()
{
	char DriverVersion[40];
	unsigned long nVersion;

	/* get a driver version */
	pDynaFlash->GetDriverVersion(DriverVersion);
	printf("DynaFlash driver Ver : %s\r\n", DriverVersion);

	/* get a HW version */
	pDynaFlash->GetHWVersion(&nVersion);
	printf("DynaFlash HW Ver     : %08x\r\n", nVersion);

	/* get a DLL version */
	pDynaFlash->GetDLLVersion(&nVersion);
	printf("DynaFlash DLL Ver    : %08x\r\n", nVersion);
}

void DynaFlashProjector::show(const cv::Mat frame)
{
	if (initialized)
	{
		// pFrameData = (char *)malloc(frame_size);
		// if (pFrameData == NULL){
		// 	std::cout << "frame buffer is NULL" << std::endl;
		// 	exit(0);
		// }
		// memcpy((void *)pFrameData, (void *)frame.data, frame_size);
		pDynaFlash->GetStatus(&stDynaFlashStatus);
		if ((stDynaFlashStatus.InputFrames - stDynaFlashStatus.OutputFrames) > 100)
		{
			std::cout << "input - output > 100!" << std::endl;
			return;
		}
		if (pDynaFlash->GetFrameBuffer(&pBuf, &nGetFrameCnt) != STATUS_SUCCESSFUL)
		{
			std::cout << "GetFrameBuffer Error\n";
			gracefully_close();
		}
		if ((pBuf != NULL) && (nGetFrameCnt != 0))
		{
			// std::cout << "frame count: " << nGetFrameCnt << std::endl;
			memcpy(pBuf, frame.data, frame_size);
			if (pDynaFlash->PostFrameBuffer(1) != STATUS_SUCCESSFUL)
			{
				std::cout << "PostFrameBuffer Error\n";
				gracefully_close();
			}
		}
		// free((void *)pFrameData);
	}
}

void DynaFlashProjector::show_buffer(const uint8_t *buffer)
{
	if (initialized)
	{
		pDynaFlash->GetStatus(&stDynaFlashStatus);
		if ((stDynaFlashStatus.InputFrames - stDynaFlashStatus.OutputFrames) > 100)
		{
			std::cout << "dropping frame, as (input buffer - output buffer) > 100" << std::endl;
			return;
		}
		if (pDynaFlash->GetFrameBuffer(&pBuf, &nGetFrameCnt) != STATUS_SUCCESSFUL)
		{
			std::cout << "GetFrameBuffer Error\n";
			gracefully_close();
		}
		if ((pBuf != NULL) && (nGetFrameCnt != 0))
		{
			// std::cout << "frame count: " << nGetFrameCnt << std::endl;
			memcpy(pBuf, buffer, frame_size);
			if (pDynaFlash->PostFrameBuffer(1) != STATUS_SUCCESSFUL)
			{
				std::cout << "PostFrameBuffer Error\n";
				gracefully_close();
			}
		}
		// free((void *)pFrameData);
	}
}

void DynaFlashProjector::show()
{
	if (initialized)
	{
		pDynaFlash->GetStatus(&stDynaFlashStatus);
		if ((stDynaFlashStatus.InputFrames - stDynaFlashStatus.OutputFrames) > 100)
		{
			return;
		}
		if (pDynaFlash->GetFrameBuffer(&pBuf, &nGetFrameCnt) != STATUS_SUCCESSFUL)
		{
			std::cout << "GetFrameBuffer Error\n";
			gracefully_close();
		}
		if ((pBuf != NULL) && (nGetFrameCnt != 0))
		{
			memcpy(pBuf, white_image.data, frame_size);
			if (pDynaFlash->PostFrameBuffer(1) != STATUS_SUCCESSFUL)
			{
				std::cout << "PostFrameBuffer Error\n";
				gracefully_close();
			}
		}
	}
}