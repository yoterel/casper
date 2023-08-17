/********************************************************************************
 * ファイル名　：DynaFlash.h
 * ファイル説明：DynaFlashクラスの公開用定義
 *
 * インスタンスの生成には「CreateCommunicationConfig」関数を使用し、
 * インスタンスの破棄には「ReleaseCommunicationConfig」関数を使用します。
 *
 *--------------------------------------------------------------------------------
 * 2016/12/07 TED 渡瀬 新規作成
 ********************************************************************************/

#ifndef _DYNAFLASH_H_
#define _DYNAFLASH_H_

#ifdef DYNAFLASH_EXPORTS
#define DYNAFLASH_API __declspec(dllexport) 
#else
#define DYNAFLASH_API __declspec(dllimport) 
#endif

/********************************************************************************
* 関数戻り値定義
********************************************************************************/

#define STATUS_SUCCESSFUL		 		(0x0000)
#define	STATUS_INCOMPLETE				(0x0001)
#define	STATUS_INVALID_PARAM			(0x0002)
#define STATUS_INVALID_BOARDINDEX		(0x0004)
#define STATUS_ALLOC_FAILED				(0x0008)
#define STATUS_INVALID_DEVICEHANDLE		(0x0010)
#define STATUS_INVALID_BARNUM			(0x0020)
#define STATUS_LOCK_FAILED				(0x0040)
#define STATUS_UNLOCK_FAILED			(0x0080)
#define STATUS_FREE_FAILED				(0x0100)
#define STATUS_INVALID_CHINDEX			(0x0200)
#define STATUS_DMA_TIMEOUT				(0x0400)
#define STATUS_NO_TRIGIN				(0x0800)
#define	STATUS_FRAME_RATE_R_OVERFLOW	(0x8000)
#define	STATUS_FRAME_RATE_G_OVERFLOW	(0x8001)
#define	STATUS_FRAME_RATE_B_OVERFLOW	(0x8002)
#define	STATUS_FRAME_RATE_W_OVERFLOW	(0x8003)
#define	STATUS_FRAME_RATE_OVERFLOW		(0x8004)

/********************************************************************************
* 各種定義
********************************************************************************/


#define MAXIMUM_NUMBER_OF_DYNAFLASH	(4)				/* DynaFlash最大接続数 */
                                                    
#define FRAME_BUF_SIZE_8BIT		(1024 * 768)		/* 投影画像サイズ(Byte) */
#define FRAME_BUF_SIZE_BINARY	(1024 * 768 / 8)	/* 投影画像サイズ(Byte) */
#define FRAME_BUF_SIZE_24BIT	(1024 * 768 * 3)	/* 投影画像サイズ(Byte) */
#define FRAME_BUF_SIZE_32BIT	(1024 * 768 * 4)	/* 投影画像サイズ(Byte) */
                                                    
                                                    
#define FRAME_MODE_BINARY		 ( 0x00 )			/* Binary Mode  (1 bit) */
#define FRAME_MODE_GRAY			 ( 0x01 )			/* Gray   Mode  (8 bit) */
#define FRAME_MODE_RGB			 ( 0x02 )			/* RGB    Mode  (24 bit)*/
#define FRAME_MODE_RGBW			 ( 0x03 )			/* RGBW   Mode  (32 bit)*/
                                                    
                                                    
  /* 投影モード設定定義 */                        
#define MIRROR					(0x00000001)		/* 投影画像の左右反転 */
#define FLIP					(0x00000002)		/* 投影画像の上下反転 */
#define COMP					(0x00000004)		/* ピクセルデータのbit反転 */
#define ONESHOT					(0x00000008)		/* ワンショット投影モード */
#define BINARY					(0x00000010)		/* バイナリモード */
#define EXT_TRIGGER				(0x00000020)		/* 外部トリガーモード */
#define TRIGGER_SKIP			(0x00000040)		/* トリガースキップモード */
#define BLOCKNUM_14				(0x00000080)		/* DMD Block Number 14 / 16 */

/* 照度設定定義 */
typedef enum {
	LOW_MODE = 0,									/* 低照度 */
	HIGH_MODE										/* 高照度 */
} ILLUMINANCE_MODE;

/********************************************************************************
 * 構造体定義
 ********************************************************************************/

 /* システムパラメータ取得用構造体 */
typedef struct _tagDynaFlashStatus
{
	unsigned long	Error;							/* DynaFlasのエラー情報 */
	unsigned long	InputFrames;					/* DynaFlashへ転送したフレーム数 */
	unsigned long	OutputFrames;					/* 投影済みのフレーム数 */
} DYNAFLASH_STATUS, *PDYNAFLASH_STATUS;

/* 投影用パラメータ構造体 */

typedef struct _tagDynaFlashParam
{
	double	dFrameRate;			/* Binary Mode: 0 - 22000; GrayMode: 1 - 2840 RGB Mode : fps 0.03 fps - 946 fps; */
	double	dRProportion;		/* 0.00 - 100 Proportion*/
	double	dGProportion;		/* 0.00 - 100 Proportion*/
	double	dBProportion;		/* 0.00 - 100 Proportion*/
	double	dWProportion;		/* 0.00 - 100 Proportion*/

	ULONG	nBinaryMode;		/* 0: Binary 1: 8bit   2: RGB 3: RGBW*/
	ULONG	nBitDepth;			/* 8: [7:0], 7: [7:1], 6: [7:2], 5: [7:3], 4: [7:4], 3: [7:5], 2: [7:6], 1: [7:7] */
	ULONG	nMirrorMode;		/* 0: enable 1: disable */
	ULONG	nFlipMode;			/* 0: enable 1: disable */
	ULONG	nCompData;			/* 0: enable 1: disable */
	ULONG	nBlockNum;			/* 0: Block 16 1: Block 14 */
	ULONG	nTriggerSelect;		/* 0: Internal 1: External */
	ULONG	nTriggerSkip;		/* 0: disable 1: enable */

	ULONG	nTimingSel;			/* 0: auto calculate 1: csv file */
	ULONG	nTimingMode;		/* 0: Normal Mode    1: Timing Mode */

	ULONG	nSWCount[4][20];	/*RGBW LED ON/OFF Timing,read from csv file*/
	ULONG	nGRSTOffset[4][8];	/*RGBW Global Reset Timing,read from csv file*/
} DYNAFLASH_PARAM, *PDYNAFLASH_PARAM;

/********************************************************************************
 * DynaFlashクラスの定義
 ********************************************************************************/
class CDynaFlash
{
public:
	explicit CDynaFlash() {}
	virtual ~CDynaFlash() {}

	virtual int Connect(unsigned int nDynaFlashIndex) = 0;
	virtual int Disconnect(void) = 0;
	virtual int PowerOff(void) = 0;
	virtual unsigned int GetIndex(void) = 0;

	virtual int GetDriverVersion(char nVersion[40]) = 0;
	virtual int GetHWVersion(unsigned long *pVersion) = 0;
	virtual int GetDLLVersion(unsigned long *pVersion) = 0;
	virtual int GetDynaFlashType(unsigned long * pDynaFlashType) = 0;

	virtual int Reset(void) = 0;
	virtual int Start(void) = 0;
	virtual int Stop(void) = 0;
	virtual int Float(unsigned int isPowerFloat) = 0;

	virtual int SetParam(PDYNAFLASH_PARAM pDynaFlashParam) = 0;
	virtual int GetStatus(PDYNAFLASH_STATUS pDynaFlashStatus) = 0;

	virtual int SetIlluminance(int nIlluminanceLevel) = 0;
	virtual int GetIlluminance(int *pIlluminanceLevel) = 0;
	virtual int SetIlluminance(ILLUMINANCE_MODE eIlluminanceMode) = 0;
	virtual int GetIlluminance(ILLUMINANCE_MODE *pIlluminanceMode) = 0;

	virtual int AllocFrameBuffer(unsigned long nFrameCnt) = 0;
	virtual int ReleaseFrameBuffer(void) = 0;

	virtual int GetFrameBuffer(char **ppBuffer, unsigned long *pFrameCnt) = 0;
	virtual int PostFrameBuffer(unsigned long nFrameCnt) = 0;

	virtual int GetFpgaInfo(float *pTemp, float *pVaux, float *pVint) = 0;
	virtual int GetFanInfo(unsigned long *pData) = 0;
	virtual int GetSysInfo(unsigned long *pData) = 0;
	virtual int GetLedTemp(unsigned long nLedIndex, float *nTemp) = 0;

	virtual int GetLedEnable(unsigned long *pLedEn) = 0;
	virtual int SetLedEnable(unsigned long nLedEn) = 0;

	/* 以下製品版では未公開の関数 */
	virtual int WriteRegister(unsigned int nBar, unsigned int nOffset, unsigned long nData) = 0;
	virtual int ReadRegister(unsigned int nBar, unsigned int nOffset, unsigned long *pData) = 0;

	virtual int WriteDACRegister(unsigned long nIndex, unsigned long nData) = 0;
	virtual int ReadDACRegister(unsigned long nIndex, unsigned long *pData) = 0;
	/********************************************************************************
 * ファイル名　：DynaFlash.h
 * ファイル説明：DynaFlashクラスの公開用定義
 *
 * インスタンスの生成には「CreateCommunicationConfig」関数を使用し、
 * インスタンスの破棄には「ReleaseCommunicationConfig」関数を使用します。
 *
 *--------------------------------------------------------------------------------
 * 2016/12/07 TED 渡瀬 新規作成
 ********************************************************************************/

#ifndef _DYNAFLASH_H_
#define _DYNAFLASH_H_

#ifdef DYNAFLASH_EXPORTS
#define DYNAFLASH_API __declspec(dllexport) 
#else
#define DYNAFLASH_API __declspec(dllimport) 
#endif

 /********************************************************************************
  * 関数戻り値定義
  ********************************************************************************/

#define STATUS_SUCCESSFUL		 		(0x0000)
#define	STATUS_INCOMPLETE				(0x0001)
#define	STATUS_INVALID_PARAM			(0x0002)
#define STATUS_INVALID_BOARDINDEX		(0x0004)
#define STATUS_ALLOC_FAILED				(0x0008)
#define STATUS_INVALID_DEVICEHANDLE		(0x0010)
#define STATUS_INVALID_BARNUM			(0x0020)
#define STATUS_LOCK_FAILED				(0x0040)
#define STATUS_UNLOCK_FAILED			(0x0080)
#define STATUS_FREE_FAILED				(0x0100)
#define STATUS_INVALID_CHINDEX			(0x0200)
#define STATUS_DMA_TIMEOUT				(0x0400)
#define STATUS_NO_TRIGIN				(0x0800)
#define	STATUS_FRAME_RATE_R_OVERFLOW	(0x8000)
#define	STATUS_FRAME_RATE_G_OVERFLOW	(0x8001)
#define	STATUS_FRAME_RATE_B_OVERFLOW	(0x8002)
#define	STATUS_FRAME_RATE_W_OVERFLOW	(0x8003)
#define	STATUS_FRAME_RATE_OVERFLOW		(0x8004)

  /********************************************************************************
   * 各種定義
   ********************************************************************************/

#define MAXIMUM_NUMBER_OF_DYNAFLASH	(4)				/* DynaFlash最大接続数 */

#define FRAME_BUF_SIZE_8BIT		(1024 * 768)		/* 投影画像サイズ(Byte) */
#define FRAME_BUF_SIZE_BINARY	(1024 * 768 / 8)	/* 投影画像サイズ(Byte) */
#define FRAME_BUF_SIZE_24BIT	(1024 * 768 * 3)	/* 投影画像サイズ(Byte) */
#define FRAME_BUF_SIZE_32BIT	(1024 * 768 * 4)	/* 投影画像サイズ(Byte) */


#define FRAME_MODE_BINARY		 ( 0x00 )			/* Binary Mode  (1 bit) */
#define FRAME_MODE_GRAY			 ( 0x01 )			/* Gray   Mode  (8 bit) */
#define FRAME_MODE_RGB			 ( 0x02 )			/* RGB    Mode  (24 bit)*/
#define FRAME_MODE_RGBW			 ( 0x03 )			/* RGBW   Mode  (32 bit)*/


   /* 投影モード設定定義 */
#define MIRROR					(0x00000001)		/* 投影画像の左右反転 */
#define FLIP					(0x00000002)		/* 投影画像の上下反転 */
#define COMP					(0x00000004)		/* ピクセルデータのbit反転 */
#define ONESHOT					(0x00000008)		/* ワンショット投影モード */
#define BINARY					(0x00000010)		/* バイナリモード */
#define EXT_TRIGGER				(0x00000020)		/* 外部トリガーモード */
#define TRIGGER_SKIP			(0x00000040)		/* トリガースキップモード */
#define BLOCKNUM_14				(0x00000080)		/* DMD Block Number 14 / 16 */

/* 照度設定定義 */
	typedef enum {
		LOW_MODE = 0,									/* 低照度 */
		HIGH_MODE										/* 高照度 */
	} ILLUMINANCE_MODE;

	/********************************************************************************
	 * 構造体定義
	 ********************************************************************************/

	 /* システムパラメータ取得用構造体 */
	typedef struct _tagDynaFlashStatus
	{
		unsigned long	Error;							/* DynaFlasのエラー情報 */
		unsigned long	InputFrames;					/* DynaFlashへ転送したフレーム数 */
		unsigned long	OutputFrames;					/* 投影済みのフレーム数 */
	} DYNAFLASH_STATUS, *PDYNAFLASH_STATUS;

	/* 投影用パラメータ構造体 */
	typedef struct _tagDynaFlashParam
	{
		double	dFrameRate;			/* Binary Mode: 0 - 22000; GrayMode: 1 - 2840 RGB Mode : fps 0.03 fps - 946 fps; */
		double	dRProportion;		/* 0.00 - 100 Proportion モノクロ無効*/
		double	dGProportion;		/* 0.00 - 100 Proportion モノクロ無効*/
		double	dBProportion;		/* 0.00 - 100 Proportion モノクロ無効*/
		double	dWProportion;		/* 0.00 - 100 Proportion モノクロ無効*/

		ULONG	nBinaryMode;		/* 0: Binary 1: 8bit   2: RGB 3: RGBW*/
		ULONG	nBitDepth;			/* 8: [7:0], 7: [7:1], 6: [7:2], 5: [7:3], 4: [7:4], 3: [7:5], 2: [7:6], 1: [7:7] */
		ULONG	nMirrorMode;		/* 0: enable 1: disable */
		ULONG	nFlipMode;			/* 0: enable 1: disable */
		ULONG	nCompData;			/* 0: enable 1: disable */
		ULONG	nTriggerSelect;		/* 0: Internal 1: External */
		ULONG	nTriggerSkip;		/* 0: disable 1: enable */

		ULONG	nBlockNum;			/* 0: Block 16 1: Block 14  モノクロ無効*/
		LONG	nTimingSel;			/* 0: auto calculate 1: csv file  モノクロ無効*/
		ULONG	nTimingMode;		/* 0: Normal Mode    1: Timing Mode  モノクロ無効*/

		ULONG	nSWCount[4][20];	/*RGBW LED ON/OFF Timing,read from csv file*/
		ULONG	nGRSTOffset[4][8];	/*RGBW Global Reset Timing,read from csv file*/
	} DYNAFLASH_PARAM, *PDYNAFLASH_PARAM;

	/********************************************************************************
	 * DynaFlashクラスの定義
	 ********************************************************************************/
	class CDynaFlash
	{
	public:
		explicit CDynaFlash() {}
		virtual ~CDynaFlash() {}

		virtual int Connect(unsigned int nDynaFlashIndex) = 0;
		virtual int Disconnect(void) = 0;
		virtual int PowerOff(void) = 0;
		virtual unsigned int GetIndex(void) = 0;

		virtual int GetDriverVersion(char nVersion[40]) = 0;
		virtual int GetHWVersion(unsigned long *pVersion) = 0;
		virtual int GetDLLVersion(unsigned long *pVersion) = 0;
		virtual int GetDynaFlashType(unsigned long * pDynaFlashType) = 0;

		virtual int Reset(void) = 0;
		virtual int Start(void) = 0;
		virtual int Stop(void) = 0;
		virtual int Float(unsigned int isPowerFloat) = 0;

		virtual int SetParam(PDYNAFLASH_PARAM pDynaFlashParam) = 0;
		virtual int GetStatus(PDYNAFLASH_STATUS pDynaFlashStatus) = 0;

		virtual int SetIlluminance(int nIlluminanceLevel) = 0;
		virtual int GetIlluminance(int *pIlluminanceLevel) = 0;
		virtual int SetIlluminance(ILLUMINANCE_MODE eIlluminanceMode) = 0;
		virtual int GetIlluminance(ILLUMINANCE_MODE *pIlluminanceMode) = 0;

		virtual int AllocFrameBuffer(unsigned long nFrameCnt) = 0;
		virtual int ReleaseFrameBuffer(void) = 0;

		virtual int GetFrameBuffer(char **ppBuffer, unsigned long *pFrameCnt) = 0;
		virtual int PostFrameBuffer(unsigned long nFrameCnt) = 0;

		virtual int GetFpgaInfo(float *pTemp, float *pVaux, float *pVint) = 0;
		virtual int GetFanInfo(unsigned long *pData) = 0;
		virtual int GetSysInfo(unsigned long *pData) = 0;
		virtual int GetLedTemp(unsigned long nLedIndex, float *nTemp) = 0;

		virtual int GetLedEnable(unsigned long *pLedEn) = 0;
		virtual int SetLedEnable(unsigned long nLedEn) = 0;

		/* 以下製品版では未公開の関数 */
		virtual int WriteRegister(unsigned int nBar, unsigned int nOffset, unsigned long nData) = 0;
		virtual int ReadRegister(unsigned int nBar, unsigned int nOffset, unsigned long *pData) = 0;

		virtual int WriteDACRegister(unsigned long nIndex, unsigned long nData) = 0;
		virtual int ReadDACRegister(unsigned long nIndex, unsigned long *pData) = 0;
		virtual int GetParameter(unsigned long* u32GRSTOffset, unsigned long *SWCount) = 0;

	};

	/********************************************************************************
	 * インスタンス生成関数
	 ********************************************************************************/
	DYNAFLASH_API CDynaFlash * _stdcall CreateDynaFlash(void);

	/********************************************************************************
	 * インスタンス破棄関数
	 ********************************************************************************/
	DYNAFLASH_API bool _stdcall ReleaseDynaFlash(CDynaFlash **pDynaFlash);

#endif

	virtual int GetParameter(unsigned long* u32GRSTOffset, unsigned long *SWCount) = 0;

};

/********************************************************************************
 * インスタンス生成関数
 ********************************************************************************/
DYNAFLASH_API CDynaFlash * _stdcall CreateDynaFlash(void);

/********************************************************************************
 * インスタンス破棄関数
 ********************************************************************************/
DYNAFLASH_API bool _stdcall ReleaseDynaFlash(CDynaFlash **pDynaFlash);

#endif
