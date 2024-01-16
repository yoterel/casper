#pragma once

// �R�����g�����Fdll�쐬���[�h�i�v���W�F�N�g���v���p�e�B���\���̎�ށ�dll�ɐݒ�j
// �R�����g�L���Fcpp���s���[�h�i�v���W�F�N�g���v���p�e�B���\���̎�ށ�exe�ɐݒ�j
//  #define UNITY_MODE

#ifndef UNITY_MODE
#define UNITY_INTERFACE_API
#define UNITY_INTERFACE_EXPORT
#else
#define UNITY_INTERFACE_API __stdcall
#define UNITY_INTERFACE_EXPORT __declspec(dllexport)
#endif

// �o�͂���֐�
#ifdef UNITY_MODE
extern "C"
{
#endif

    // API for Unity
    UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API OpenMidiController();
    UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API CloseMidiController();

    UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API GetSliderValues(float *out_array, int first_idx = -1, int last_idx = -1);
    UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API GetKnobValues(float *out_array, int first_idx = -1, int last_idx = -1);
    UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API GetButtonStates(bool *out_array, int first_idx = -1, int last_idx = -1);
    UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API GetSystemButtonStates(bool *out_array, int first_idx = -1, int last_idx = -1);

    UNITY_INTERFACE_EXPORT float UNITY_INTERFACE_API Get1SliderValue(int id);
    UNITY_INTERFACE_EXPORT float UNITY_INTERFACE_API Get1KnobValue(int id);
    UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API Get1ButtonState(int id);
    UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API Get1SystemButtonState(const char *name);

    UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API SetButtonOnInterval(int msec);

#ifdef UNITY_MODE
}
#endif