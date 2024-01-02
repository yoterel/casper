/********************************************************************************
 * �t�@�C�����@�FDynaFlash.h
 * �t�@�C�������FDynaFlash�N���X�̌��J�p��`
 *
 * �C���X�^���X�̐����ɂ́uCreateCommunicationConfig�v�֐����g�p���A
 * �C���X�^���X�̔j���ɂ́uReleaseCommunicationConfig�v�֐����g�p���܂��B
 *
 *--------------------------------------------------------------------------------
 * 2016/12/07 TED �n�� �V�K�쐬
 ********************************************************************************/

#ifndef _DYNAFLASH_H_
#define _DYNAFLASH_H_

#ifdef DYNAFLASH_EXPORTS
#define DYNAFLASH_API __declspec(dllexport) 
#else
#define DYNAFLASH_API __declspec(dllimport) 
#endif

/********************************************************************************
* �֐��߂�l��`
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
* �e���`
********************************************************************************/


#define MAXIMUM_NUMBER_OF_DYNAFLASH	(4)				/* DynaFlash�ő�ڑ��� */
                                                    
#define FRAME_BUF_SIZE_8BIT		(1024 * 768)		/* ���e�摜�T�C�Y(Byte) */
#define FRAME_BUF_SIZE_BINARY	(1024 * 768 / 8)	/* ���e�摜�T�C�Y(Byte) */
#define FRAME_BUF_SIZE_24BIT	(1024 * 768 * 3)	/* ���e�摜�T�C�Y(Byte) */
#define FRAME_BUF_SIZE_32BIT	(1024 * 768 * 4)	/* ���e�摜�T�C�Y(Byte) */
                                                    
                                                    
#define FRAME_MODE_BINARY		 ( 0x00 )			/* Binary Mode  (1 bit) */
#define FRAME_MODE_GRAY			 ( 0x01 )			/* Gray   Mode  (8 bit) */
#define FRAME_MODE_RGB			 ( 0x02 )			/* RGB    Mode  (24 bit)*/
#define FRAME_MODE_RGBW			 ( 0x03 )			/* RGBW   Mode  (32 bit)*/
                                                    
                                                    
  /* ���e���[�h�ݒ��` */                        
#define MIRROR					(0x00000001)		/* ���e�摜�̍��E���] */
#define FLIP					(0x00000002)		/* ���e�摜�̏㉺���] */
#define COMP					(0x00000004)		/* �s�N�Z���f�[�^��bit���] */
#define ONESHOT					(0x00000008)		/* �����V���b�g���e���[�h */
#define BINARY					(0x00000010)		/* �o�C�i�����[�h */
#define EXT_TRIGGER				(0x00000020)		/* �O���g���K�[���[�h */
#define TRIGGER_SKIP			(0x00000040)		/* �g���K�[�X�L�b�v���[�h */
#define BLOCKNUM_14				(0x00000080)		/* DMD Block Number 14 / 16 */

/* �Ɠx�ݒ��` */
typedef enum {
	LOW_MODE = 0,									/* ��Ɠx */
	HIGH_MODE										/* ���Ɠx */
} ILLUMINANCE_MODE;

/********************************************************************************
 * �\���̒�`
 ********************************************************************************/

 /* �V�X�e���p�����[�^�擾�p�\���� */
typedef struct _tagDynaFlashStatus
{
	unsigned long	Error;							/* DynaFlas�̃G���[��� */
	unsigned long	InputFrames;					/* DynaFlash�֓]�������t���[���� */
	unsigned long	OutputFrames;					/* ���e�ς݂̃t���[���� */
} DYNAFLASH_STATUS, *PDYNAFLASH_STATUS;

/* ���e�p�p�����[�^�\���� */

typedef struct _tagDynaFlashParam
{
	double	dFrameRate;			/* Binary Mode: 0 - 22000; GrayMode: 1 - 2840 RGB Mode : fps 0.03 fps - 946 fps; */
	double	dRProportion;		/* 0.00 - 100 Proportion*/
	double	dGProportion;		/* 0.00 - 100 Proportion*/
	double	dBProportion;		/* 0.00 - 100 Proportion*/
	double	dWProportion;		/* 0.00 - 100 Proportion*/

	unsigned long	nBinaryMode;		/* 0: Binary 1: 8bit   2: RGB 3: RGBW*/
	unsigned long	nBitDepth;			/* 8: [7:0], 7: [7:1], 6: [7:2], 5: [7:3], 4: [7:4], 3: [7:5], 2: [7:6], 1: [7:7] */
	unsigned long	nMirrorMode;		/* 0: enable 1: disable */
	unsigned long	nFlipMode;			/* 0: enable 1: disable */
	unsigned long	nCompData;			/* 0: enable 1: disable */
	unsigned long	nBlockNum;			/* 0: Block 16 1: Block 14 */
	unsigned long	nTriggerSelect;		/* 0: Internal 1: External */
	unsigned long	nTriggerSkip;		/* 0: disable 1: enable */

	unsigned long	nTimingSel;			/* 0: auto calculate 1: csv file */
	unsigned long	nTimingMode;		/* 0: Normal Mode    1: Timing Mode */

	unsigned long	nSWCount[4][20];	/*RGBW LED ON/OFF Timing,read from csv file*/
	unsigned long	nGRSTOffset[4][8];	/*RGBW Global Reset Timing,read from csv file*/
} DYNAFLASH_PARAM, *PDYNAFLASH_PARAM;

/********************************************************************************
 * DynaFlash�N���X�̒�`
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

	/* �ȉ����i�łł͖����J�̊֐� */
	virtual int WriteRegister(unsigned int nBar, unsigned int nOffset, unsigned long nData) = 0;
	virtual int ReadRegister(unsigned int nBar, unsigned int nOffset, unsigned long *pData) = 0;

	virtual int WriteDACRegister(unsigned long nIndex, unsigned long nData) = 0;
	virtual int ReadDACRegister(unsigned long nIndex, unsigned long *pData) = 0;
	/********************************************************************************
 * �t�@�C�����@�FDynaFlash.h
 * �t�@�C�������FDynaFlash�N���X�̌��J�p��`
 *
 * �C���X�^���X�̐����ɂ́uCreateCommunicationConfig�v�֐����g�p���A
 * �C���X�^���X�̔j���ɂ́uReleaseCommunicationConfig�v�֐����g�p���܂��B
 *
 *--------------------------------------------------------------------------------
 * 2016/12/07 TED �n�� �V�K�쐬
 ********************************************************************************/

#ifndef _DYNAFLASH_H_
#define _DYNAFLASH_H_

#ifdef DYNAFLASH_EXPORTS
#define DYNAFLASH_API __declspec(dllexport) 
#else
#define DYNAFLASH_API __declspec(dllimport) 
#endif

 /********************************************************************************
  * �֐��߂�l��`
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
   * �e���`
   ********************************************************************************/

#define MAXIMUM_NUMBER_OF_DYNAFLASH	(4)				/* DynaFlash�ő�ڑ��� */

#define FRAME_BUF_SIZE_8BIT		(1024 * 768)		/* ���e�摜�T�C�Y(Byte) */
#define FRAME_BUF_SIZE_BINARY	(1024 * 768 / 8)	/* ���e�摜�T�C�Y(Byte) */
#define FRAME_BUF_SIZE_24BIT	(1024 * 768 * 3)	/* ���e�摜�T�C�Y(Byte) */
#define FRAME_BUF_SIZE_32BIT	(1024 * 768 * 4)	/* ���e�摜�T�C�Y(Byte) */


#define FRAME_MODE_BINARY		 ( 0x00 )			/* Binary Mode  (1 bit) */
#define FRAME_MODE_GRAY			 ( 0x01 )			/* Gray   Mode  (8 bit) */
#define FRAME_MODE_RGB			 ( 0x02 )			/* RGB    Mode  (24 bit)*/
#define FRAME_MODE_RGBW			 ( 0x03 )			/* RGBW   Mode  (32 bit)*/


   /* ���e���[�h�ݒ��` */
#define MIRROR					(0x00000001)		/* ���e�摜�̍��E���] */
#define FLIP					(0x00000002)		/* ���e�摜�̏㉺���] */
#define COMP					(0x00000004)		/* �s�N�Z���f�[�^��bit���] */
#define ONESHOT					(0x00000008)		/* �����V���b�g���e���[�h */
#define BINARY					(0x00000010)		/* �o�C�i�����[�h */
#define EXT_TRIGGER				(0x00000020)		/* �O���g���K�[���[�h */
#define TRIGGER_SKIP			(0x00000040)		/* �g���K�[�X�L�b�v���[�h */
#define BLOCKNUM_14				(0x00000080)		/* DMD Block Number 14 / 16 */

/* �Ɠx�ݒ��` */
	typedef enum {
		LOW_MODE = 0,									/* ��Ɠx */
		HIGH_MODE										/* ���Ɠx */
	} ILLUMINANCE_MODE;

	/********************************************************************************
	 * �\���̒�`
	 ********************************************************************************/

	 /* �V�X�e���p�����[�^�擾�p�\���� */
	typedef struct _tagDynaFlashStatus
	{
		unsigned long	Error;							/* DynaFlas�̃G���[��� */
		unsigned long	InputFrames;					/* DynaFlash�֓]�������t���[���� */
		unsigned long	OutputFrames;					/* ���e�ς݂̃t���[���� */
	} DYNAFLASH_STATUS, *PDYNAFLASH_STATUS;

	/* ���e�p�p�����[�^�\���� */
	typedef struct _tagDynaFlashParam
	{
		double	dFrameRate;			/* Binary Mode: 0 - 22000; GrayMode: 1 - 2840 RGB Mode : fps 0.03 fps - 946 fps; */
		double	dRProportion;		/* 0.00 - 100 Proportion ���m�N������*/
		double	dGProportion;		/* 0.00 - 100 Proportion ���m�N������*/
		double	dBProportion;		/* 0.00 - 100 Proportion ���m�N������*/
		double	dWProportion;		/* 0.00 - 100 Proportion ���m�N������*/

		ULONG	nBinaryMode;		/* 0: Binary 1: 8bit   2: RGB 3: RGBW*/
		ULONG	nBitDepth;			/* 8: [7:0], 7: [7:1], 6: [7:2], 5: [7:3], 4: [7:4], 3: [7:5], 2: [7:6], 1: [7:7] */
		ULONG	nMirrorMode;		/* 0: enable 1: disable */
		ULONG	nFlipMode;			/* 0: enable 1: disable */
		ULONG	nCompData;			/* 0: enable 1: disable */
		ULONG	nTriggerSelect;		/* 0: Internal 1: External */
		ULONG	nTriggerSkip;		/* 0: disable 1: enable */

		ULONG	nBlockNum;			/* 0: Block 16 1: Block 14  ���m�N������*/
		LONG	nTimingSel;			/* 0: auto calculate 1: csv file  ���m�N������*/
		ULONG	nTimingMode;		/* 0: Normal Mode    1: Timing Mode  ���m�N������*/

		ULONG	nSWCount[4][20];	/*RGBW LED ON/OFF Timing,read from csv file*/
		ULONG	nGRSTOffset[4][8];	/*RGBW Global Reset Timing,read from csv file*/
	} DYNAFLASH_PARAM, *PDYNAFLASH_PARAM;

	/********************************************************************************
	 * DynaFlash�N���X�̒�`
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

		/* �ȉ����i�łł͖����J�̊֐� */
		virtual int WriteRegister(unsigned int nBar, unsigned int nOffset, unsigned long nData) = 0;
		virtual int ReadRegister(unsigned int nBar, unsigned int nOffset, unsigned long *pData) = 0;

		virtual int WriteDACRegister(unsigned long nIndex, unsigned long nData) = 0;
		virtual int ReadDACRegister(unsigned long nIndex, unsigned long *pData) = 0;
		virtual int GetParameter(unsigned long* u32GRSTOffset, unsigned long *SWCount) = 0;

	};

	/********************************************************************************
	 * �C���X�^���X�����֐�
	 ********************************************************************************/
	DYNAFLASH_API CDynaFlash * _stdcall CreateDynaFlash(void);

	/********************************************************************************
	 * �C���X�^���X�j���֐�
	 ********************************************************************************/
	DYNAFLASH_API bool _stdcall ReleaseDynaFlash(CDynaFlash **pDynaFlash);

#endif

	virtual int GetParameter(unsigned long* u32GRSTOffset, unsigned long *SWCount) = 0;

};

/********************************************************************************
 * �C���X�^���X�����֐�
 ********************************************************************************/
DYNAFLASH_API CDynaFlash * _stdcall CreateDynaFlash(void);

/********************************************************************************
 * �C���X�^���X�j���֐�
 ********************************************************************************/
DYNAFLASH_API bool _stdcall ReleaseDynaFlash(CDynaFlash **pDynaFlash);

#endif
