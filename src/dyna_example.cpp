/********************************************************************************
 * File Name　：SimpleSample
 * Description：Sample application for DynaFlash
 *
 *--------------------------------------------------------------------------------
 * 2016/12/08 Create							Watase.h
 * 2017/02/28 Add:Supported OpenCV			Kiguchi.t
 * 2017/07/25 Fix:Supported API 1.02			Kiguchi.t
 * 2019/03/23 Fix:Supported API 1.20			Kiguchi.t
 * 2019/06/11    :							Watase.h
 ********************************************************************************/

#include "stdafx.h"

#include <windows.h>
#include <stdio.h>
#include <string.h>
#include <signal.h>
#include "opencv2/opencv.hpp"
#include "DynaFlash.h"

#define MONOCHROME (0) /* 0:モノクロ 1:カラー */

#if MONOCHROME
#define FRAME_MODE (FRAME_MODE_GRAY)
#define FRAME_SIZE (FRAME_BUF_SIZE_8BIT)
#else
#define FRAME_MODE (FRAME_MODE_RGB)
#define FRAME_SIZE (FRAME_BUF_SIZE_24BIT)
#endif

#define BOARD_INDEX (0)				  /* PCに接続されたDynaFlashのインデックス */
#define FRAME_RATE (946.0f)			  /* DynaFlashのフレームレート */
#define BIT_DEPTH (8)				  /* DynaFlashのBit深度 */
#define ALLOC_FRAME_BUFFER (16)		  /* 確保するフレームバッファ数 (ドライバ) */
#define ALLOC_SRC_FRAME_BUFFER (2000) /* 確保するフレームバッファ数 (アプリケーション) */

static CDynaFlash *pDynaFlash = NULL;					  /* I/Fクラスのインスタンス */
static char *pFrameData[ALLOC_SRC_FRAME_BUFFER] = {NULL}; /* 投影画像の読み込み先 */
bool signal_received = false;
/********************************************************************************
 * バージョン表示
 ********************************************************************************/
void signal_callback_handler(int signum)
{
	std::cout << "Caught signal " << signum << std::endl;
	signal_received = true;
}

void PrintVersion()
{
	// char DriverVersion[40];
	unsigned long nVersion;

	/* ドライババージョン取得 */
	//	pDynaFlash->GetDriverVersion(DriverVersion);
	//	printf("DRIVER Ver : %s\r\n", DriverVersion);

	/* DLLバージョン取得 */
	pDynaFlash->GetDLLVersion(&nVersion);
	printf("DLL Ver    : %08x\r\n", nVersion);

	/* HWバージョン取得 */
	pDynaFlash->GetHWVersion(&nVersion);
	printf("HW Ver     : %08x\r\n", nVersion);
}

/********************************************************************************
 * 動画の読み込み (フレーム単位でPCメモリに展開する)
 ********************************************************************************/
int LoadMovieFile(char *pFileName)
{
	// char FileName[MAX_PATH];
	cv::Mat FrameDataOrg;
	cv::Mat FrameDataResize;
#if MONOCHROME
	cv::Mat FrameDataGray;
#endif
	int nFrameCnt = 0;

	// WideCharToMultiByte(CP_ACP, 0, pFileName, -1, FileName, MAX_PATH, NULL, NULL);

	/* 動画ファイルを開く */
	cv::VideoCapture capture(pFileName);
	if (!capture.isOpened())
	{
		return nFrameCnt;
	}

	for (int i = 0; i < ALLOC_SRC_FRAME_BUFFER; i++)
	{
		/* 1フレーム読み込み */
		capture >> FrameDataOrg;
		if (FrameDataOrg.empty())
		{
			break;
		}

		/* 読み込んだ画像を保存するためのメモリを確保する */
		pFrameData[i] = (char *)malloc(FRAME_SIZE);
		if (pFrameData[i] == NULL)
		{
			break;
		}

		/* 画像サイズを1024 x 768にリサイズする */
		cv::resize(FrameDataOrg, FrameDataResize, cv::Size(), 1024.0f / FrameDataOrg.cols, 768.0f / FrameDataOrg.rows);

#if MONOCHROME
		/* モノクロ画像に変換する */
		cv::cvtColor(FrameDataResize, FrameDataGray, cv::COLOR_RGB2GRAY);

		/* 確保したメモリにコピー */
		memcpy((void *)pFrameData[i], (void *)FrameDataGray.data, FRAME_SIZE);
#else
		/* 確保したメモリにコピー */
		memcpy((void *)pFrameData[i], (void *)FrameDataResize.data, FRAME_SIZE);
#endif

		nFrameCnt++;
	}

	return nFrameCnt;
}

/********************************************************************************
 * 動画の破棄
 ********************************************************************************/
void ReleaseMovieFile()
{
	for (int i = 0; i < ALLOC_SRC_FRAME_BUFFER; i++)
	{
		if (pFrameData[i] != NULL)
		{
			free((void *)pFrameData[i]);
		}
	}
}

/********************************************************************************
 * サンプルメイン
 ********************************************************************************/
int _tmain(int argc, _TCHAR *argv[])
{
	signal(SIGINT, signal_callback_handler);
	DYNAFLASH_STATUS stDynaFlashStatus;
	char *pBuf = NULL;
	unsigned long nGetFrameCnt = 0;
	unsigned long nFrameCnt = 0;
	unsigned long nMaxFrameNum = 0;
	DYNAFLASH_PARAM tDynaFlash_Param = {0};
	// cv::Mat white_image(1024, 768, CV_8UC3, cv::Scalar(255, 255, 255));
	if (argc <= 1)
	{
		printf("usage dyna_example.exe filename\n");
		return -1;
	}

	if (!SetProcessWorkingSetSizeEx(::GetCurrentProcess(), (2000UL * 1024 * 1024), (3000UL * 1024 * 1024), QUOTA_LIMITS_HARDWS_MIN_ENABLE))
	{
		printf("SetProcessWorkingSetSize Failed!\n");
		return -1;
	}

	/* DynaFlashクラスのインスタンス生成 */
	pDynaFlash = CreateDynaFlash();
	if (pDynaFlash == NULL)
	{
		return -1;
	}

	/* DynaFlash接続 */
	if (pDynaFlash->Connect(BOARD_INDEX) != STATUS_SUCCESSFUL)
	{
		return -1;
	}

	/* 各種バージョン表示 */
	PrintVersion();

	/* DynaFlashリセット */
	if (pDynaFlash->Reset() != STATUS_SUCCESSFUL)
	{
		goto _exit;
	}

	/* 投影パラメーター設定 */
	//	if (pDynaFlash->SetParam(FRAME_RATE, FRAME_MODE_RGB, BIT_DEPTH, 0) != STATUS_SUCCESSFUL) {	// 20210406 ono.ys DLL変更に伴う異常終了対応
	tDynaFlash_Param.dFrameRate = FRAME_RATE;
	tDynaFlash_Param.dRProportion = (double)100 / (double)3;
	tDynaFlash_Param.dGProportion = (double)100 / (double)3;
	tDynaFlash_Param.dBProportion = (double)100 / (double)3;
	tDynaFlash_Param.nBinaryMode = FRAME_MODE_RGB;
	tDynaFlash_Param.nBitDepth = BIT_DEPTH;
	tDynaFlash_Param.nMirrorMode = 1;
	if (pDynaFlash->SetParam(&tDynaFlash_Param) != STATUS_SUCCESSFUL)
	{
		goto _exit;
	}
	// ILLUMINANCE_MODE cur_ilum_mode;
	// if (pDynaFlash->GetIlluminance(&cur_ilum_mode) != STATUS_SUCCESSFUL)
	// {
	// 	goto _exit;
	// }
	// std::cout << "DynaFlash current illuminance mode: " << cur_ilum_mode << std::endl;
	/* 投影照度設定 */
	if (pDynaFlash->SetIlluminance(HIGH_MODE) != STATUS_SUCCESSFUL)
	{
		goto _exit;
	}
	// if (pDynaFlash->GetIlluminance(&cur_ilum_mode) != STATUS_SUCCESSFUL)
	// {
	// 	goto _exit;
	// }
	// std::cout << "DynaFlash current illuminance mode: " << cur_ilum_mode << std::endl;
	/* 投影データ転送用フレームバッファ確保 */
	if (pDynaFlash->AllocFrameBuffer(ALLOC_FRAME_BUFFER) != STATUS_SUCCESSFUL)
	{
		goto _exit;
	}

	/* 動画の読み込み (フレーム単位でPCメモリに展開する) */

	printf("Load Movie File...\n");
	nMaxFrameNum = LoadMovieFile(argv[1]);
	printf("%dFrames\n", nMaxFrameNum);
	if (nMaxFrameNum == 0)
	{
		goto _exit;
	}

	/* 投影開始 */
	if (pDynaFlash->Start() != STATUS_SUCCESSFUL)
	{
		printf("Start Error\n");
		goto _exit;
	}

	printf("Please the ESC key to exit.\n");

	for (;;)
	{
		/* ESCキーが押されたら終了する */
		if ((GetAsyncKeyState(VK_ESCAPE) & 0x80000000) != 0)
		{
			break;
		}
		if (signal_received)
		{
			break;
		}

		// auto start = std::chrono::system_clock::now();
		/* ステータスの取得 */
		pDynaFlash->GetStatus(&stDynaFlashStatus);
		int dropped = stDynaFlashStatus.InputFrames - stDynaFlashStatus.OutputFrames;
		if ((dropped % 1000 == 0) && (dropped > 0))
		{
			// should happen eventually since this program is much faster than the projector throughput
			std::cout << "frames transffered: " << stDynaFlashStatus.InputFrames << std::endl;
			std::cout << "frames projected: " << stDynaFlashStatus.OutputFrames << std::endl;
			std::cout << "dropped: " << dropped << std::endl;
		}

		/* 投影データ更新可能な投影用フレームバッファ取得 */
		if (pDynaFlash->GetFrameBuffer(&pBuf, &nGetFrameCnt) != STATUS_SUCCESSFUL)
		{
			printf("GetFrameBuffer Error\n");
			goto _exit;
		}

		/* フレームバッファの更新処理を記載する */
		if ((pBuf != NULL) && (nGetFrameCnt != 0))
		{
			// std::cout << "frame count: " << nGetFrameCnt << std::endl;
			/* メモリに読み込み済みの動画データをDynaFlash転送用のフレームバッファにコピーする */
			memcpy(pBuf, pFrameData[nFrameCnt], FRAME_SIZE);

			nFrameCnt++;
			if (nFrameCnt >= nMaxFrameNum)
			{
				nFrameCnt = 0;
			}

			if (pDynaFlash->PostFrameBuffer(1) != STATUS_SUCCESSFUL)
			{
				printf("PostFrameBuffer Error\n");
				goto _exit;
			}
		}
		// auto runtime = std::chrono::system_clock::now() - start;
		//         std::cout << "ms: "
		//         << (std::chrono::duration_cast<std::chrono::microseconds>(runtime)).count()*1.0/1000
		//         << "\n";
	}

_exit:
	/* 投影終了 */
	pDynaFlash->Stop();

	/* フレームバッファの破棄 */
	pDynaFlash->ReleaseFrameBuffer();

	/* ミラーデバイスをフロート状態にする */
	pDynaFlash->Float(0);

	/* DynaFlash切断 */
	pDynaFlash->Disconnect();

	/* DynaFlashクラスのインスタンス破棄 */
	ReleaseDynaFlash(&pDynaFlash);

	/* 動画の破棄 */
	ReleaseMovieFile();

	return 0;
}
