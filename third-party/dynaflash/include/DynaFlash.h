/********************************************************************************
 * File name: DynaFlash.h
 * File description: Public definition of DynaFlash class
 *
 * Use the "CreateCommunicationConfig" function to generate an instance
 * ReleaseCommunicationConfig" function to destroy the instance
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
 * Function return value definition
 ********************************************************************************/

#define STATUS_SUCCESSFUL (0x0000)
#define STATUS_INCOMPLETE (0x0001)
#define STATUS_INVALID_PARAM (0x0002)
#define STATUS_INVALID_BOARDINDEX (0x0004)
#define STATUS_ALLOC_FAILED (0x0008)
#define STATUS_INVALID_DEVICEHANDLE (0x0010)
#define STATUS_INVALID_BARNUM (0x0020)
#define STATUS_LOCK_FAILED (0x0040)
#define STATUS_UNLOCK_FAILED (0x0080)
#define STATUS_FREE_FAILED (0x0100)
#define STATUS_INVALID_CHINDEX (0x0200)
#define STATUS_DMA_TIMEOUT (0x0400)
#define STATUS_NO_TRIGIN (0x0800)
#define STATUS_FRAME_RATE_R_OVERFLOW (0x8000)
#define STATUS_FRAME_RATE_G_OVERFLOW (0x8001)
#define STATUS_FRAME_RATE_B_OVERFLOW (0x8002)
#define STATUS_FRAME_RATE_W_OVERFLOW (0x8003)
#define STATUS_FRAME_RATE_OVERFLOW (0x8004)

/********************************************************************************
 * Various definitions
 ********************************************************************************/

#define MAXIMUM_NUMBER_OF_DYNAFLASH (4) /* Maximum number of DynaFlash connections */

#define FRAME_BUF_SIZE_8BIT (1024 * 768)	   /* Projection image size (Byte) */
#define FRAME_BUF_SIZE_BINARY (1024 * 768 / 8) /* Projection image size (Byte) */
#define FRAME_BUF_SIZE_24BIT (1024 * 768 * 3)  /* Projection image size (Byte) */
#define FRAME_BUF_SIZE_32BIT (1024 * 768 * 4)  /* Projection image size (Byte) */

#define FRAME_MODE_BINARY (0x00) /* Binary Mode  (1 bit) */
#define FRAME_MODE_GRAY (0x01)	 /* Gray   Mode  (8 bit) */
#define FRAME_MODE_RGB (0x02)	 /* RGB    Mode  (24 bit)*/
#define FRAME_MODE_RGBW (0x03)	 /* RGBW   Mode  (32 bit)*/

/* Projection mode setting definition */
#define MIRROR (0x00000001)		  /* Flip the projected image horizontally */
#define FLIP (0x00000002)		  /* Flip the projected image vertically */
#define COMP (0x00000004)		  /* Bit inversion of pixel data */
#define ONESHOT (0x00000008)	  /* One-shot projection mode */
#define BINARY (0x00000010)		  /* Binary mode */
#define EXT_TRIGGER (0x00000020)  /* External trigger mode */
#define TRIGGER_SKIP (0x00000040) /* Trigger skip mode */
#define BLOCKNUM_14 (0x00000080)  /* DMD Block Number 14 / 16 */

/* Illuminance setting definition */
typedef enum
{
	LOW_MODE = 0, /* Low illumination */
	HIGH_MODE	  /* High illumination */
} ILLUMINANCE_MODE;

/********************************************************************************
 * structure definitions
 ********************************************************************************/

/* Structure for obtaining system parameters */
typedef struct _tagDynaFlashStatus
{
	unsigned long Error;		/* Error information */
	unsigned long InputFrames;	/* Number of actual projected frames by the projector */
	unsigned long OutputFrames; /* Number of frames transferred to the projector */
} DYNAFLASH_STATUS, *PDYNAFLASH_STATUS;

/* Projection parameter structure */

typedef struct _tagDynaFlashParam
{
	double dFrameRate;	 /* Binary Mode: 0 - 22000; GrayMode: 1 - 2840 RGB Mode : fps 0.03 fps - 946 fps; */
	double dRProportion; /* 0.00 - 100 Proportion*/
	double dGProportion; /* 0.00 - 100 Proportion*/
	double dBProportion; /* 0.00 - 100 Proportion*/
	double dWProportion; /* 0.00 - 100 Proportion*/

	unsigned long nBinaryMode;	  /* 0: Binary 1: 8bit   2: RGB 3: RGBW*/
	unsigned long nBitDepth;	  /* 8: [7:0], 7: [7:1], 6: [7:2], 5: [7:3], 4: [7:4], 3: [7:5], 2: [7:6], 1: [7:7] */
	unsigned long nMirrorMode;	  /* 0: enable 1: disable */
	unsigned long nFlipMode;	  /* 0: enable 1: disable */
	unsigned long nCompData;	  /* 0: enable 1: disable */
	unsigned long nBlockNum;	  /* 0: Block 16 1: Block 14 */
	unsigned long nTriggerSelect; /* 0: Internal 1: External */
	unsigned long nTriggerSkip;	  /* 0: disable 1: enable */

	unsigned long nTimingSel;  /* 0: auto calculate 1: csv file */
	unsigned long nTimingMode; /* 0: Normal Mode    1: Timing Mode */

	unsigned long nSWCount[4][20];	 /*RGBW LED ON/OFF Timing,read from csv file*/
	unsigned long nGRSTOffset[4][8]; /*RGBW Global Reset Timing,read from csv file*/
} DYNAFLASH_PARAM, *PDYNAFLASH_PARAM;

/********************************************************************************
 * DynaFlash class definition
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
	virtual int GetDynaFlashType(unsigned long *pDynaFlashType) = 0;

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

	/* The following functions are not released in the product version */
	virtual int WriteRegister(unsigned int nBar, unsigned int nOffset, unsigned long nData) = 0;
	virtual int ReadRegister(unsigned int nBar, unsigned int nOffset, unsigned long *pData) = 0;

	virtual int WriteDACRegister(unsigned long nIndex, unsigned long nData) = 0;
	virtual int ReadDACRegister(unsigned long nIndex, unsigned long *pData) = 0;

	virtual int GetParameter(unsigned long *u32GRSTOffset, unsigned long *SWCount) = 0;
};

/********************************************************************************
 * Instance generation function
 ********************************************************************************/
DYNAFLASH_API CDynaFlash *_stdcall CreateDynaFlash(void);

/********************************************************************************
 * Instance destruction function
 ********************************************************************************/
DYNAFLASH_API bool _stdcall ReleaseDynaFlash(CDynaFlash **pDynaFlash);

#endif
