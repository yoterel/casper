#include <windows.h>
#include <mmsystem.h>
#include <string>
#include <map>
#include "MidiControllerAPI.h"
#pragma comment(lib, "winmm.lib")

#ifdef UNITY_MODE
extern "C"
{
#endif // !DEBUGMODE

// midi controller information
#define CONTROLLER_NAME "WORLDE easy control"
#define SLIDER_BIAS 3
#define SLIDER_NUM 9
#define KNOB_BIAS 14
#define KNOB_NUM 9
#define BUTTON_BIAS 23
#define BUTTON_NUM 9
#define SYSTEM_BUTTON_BIAS 44
#define SYSTEM_BUTTON_NUM 6
#define OTHER_BUTTON_NUM 4

    class MidiController
    {
    private:
        bool enable = true;
        int button_on_interval = 100;

        std::map<std::string, int> system_buttons_map = {{"RECORD", 0}, {"PLAY", 1}, {"STOP", 2}, {"REWIND", 3}, {"FORWARD", 4}, {"UPDATE", 5}};

        float sliders[SLIDER_NUM] = {-1};
        float knobs[KNOB_NUM] = {-1};

        float buttons[BUTTON_NUM] = {-1};
        bool app_buttons[BUTTON_NUM] = {false};
        unsigned int buttons_lastontimes[BUTTON_NUM] = {0};

        float system_buttons[SYSTEM_BUTTON_NUM] = {-1};
        bool app_system_buttons[SYSTEM_BUTTON_NUM] = {false};
        unsigned int system_buttons_lastontimes[SYSTEM_BUTTON_NUM] = {0};

    public:
        MidiController(bool enable = true)
        {
            this->enable = enable;
            Initializer();
        }

        void Initializer()
        {
            for (int i = 0; i < SLIDER_NUM; i++)
                sliders[i] = -1;
            for (int i = 0; i < KNOB_NUM; i++)
                knobs[i] = -1;
            for (int i = 0; i < BUTTON_NUM; i++)
            {
                buttons[i] = -1;
                app_buttons[i] = false;
                buttons_lastontimes[i] = 0;
            }
            for (int i = 0; i < SYSTEM_BUTTON_NUM; i++)
            {
                system_buttons[i] = -1;
                app_system_buttons[i] = false;
                system_buttons_lastontimes[i] = 0;
            }
        }

        void DecodeMidiEvent(unsigned char name, unsigned char value, unsigned int time)
        {
            // slider
            if (name >= SLIDER_BIAS && name < (SLIDER_BIAS + SLIDER_NUM))
            {
                sliders[name - SLIDER_BIAS] = value / 127.0;
            }

            // knob
            else if (name >= KNOB_BIAS && name < (KNOB_BIAS + KNOB_NUM))
            {
                knobs[name - KNOB_BIAS] = value / 127.0;
            }

            // button
            else if (name >= BUTTON_BIAS && name < (BUTTON_BIAS + BUTTON_NUM))
            {
                int idx = name - BUTTON_BIAS;
                buttons[idx] = value / 127.0;
                if (value == 127)
                {
                    if (time - buttons_lastontimes[idx] > button_on_interval)
                    {
                        app_buttons[idx] = true;
                        buttons_lastontimes[idx] = time;
                    }
                    // else {
                    //     printf("chattering����?\n");
                    // }
                }
                else
                {
                    app_buttons[idx] = false;
                }
            }

            // system_button
            else if (name >= SYSTEM_BUTTON_BIAS && name < (SYSTEM_BUTTON_BIAS + SYSTEM_BUTTON_NUM))
            {
                int idx = name - SYSTEM_BUTTON_BIAS;
                system_buttons[idx] = value / 127.0;
                if (value == 127)
                {
                    if (time - system_buttons_lastontimes[idx] > button_on_interval)
                    {
                        app_system_buttons[idx] = true;
                        system_buttons_lastontimes[idx] = time;
                    }
                    // else {
                    //     printf("chattering����?\n");
                    // }
                }
            }

            // ���̃{�^�����g�������Ȃ�A�����ɋL�q���Ă�������
            else
            {
                printf("this program does not handle the buttons\nbutton id = %u, button_value = %u\n", name, value);
            }
        }

        bool GetSliderValue(float *out_array, int first_idx, int last_idx) const
        {
            if (!enable)
                return false;
            if (first_idx == -1 && last_idx == -1)
            {
                // �S�ẴX���C�_�[�̏���񋟂���
                first_idx = 0;
                last_idx = SLIDER_NUM;
            }
            for (int i = first_idx; i < last_idx; i++)
                out_array[i] = sliders[i];
            return true;
        }
        bool GetKnobValue(float *out_array, int first_idx, int last_idx) const
        {
            if (!enable)
                return false;
            if (first_idx == -1 && last_idx == -1)
            {
                // �S�Ẵm�u�̏���񋟂���
                first_idx = 0;
                last_idx = KNOB_NUM;
            }
            for (int i = first_idx; i < last_idx; i++)
                out_array[i] = knobs[i];
            return true;
        }
        bool GetButtonState(bool *out_array, int first_idx, int last_idx)
        {
            if (!enable)
                return false;
            if (first_idx == -1 && last_idx == -1)
            {
                // �S�Ẵ{�^���̏���񋟂���
                first_idx = 0;
                last_idx = BUTTON_NUM;
            }
            for (int i = first_idx; i < last_idx; i++)
            {
                out_array[i] = (app_buttons[i]) ? true : false;
                app_buttons[i] = false;
            }
            return true;
        }
        bool GetSystemButtonState(bool *out_array, int first_idx, int last_idx)
        {
            if (!enable)
                return false;
            if (first_idx == -1 && last_idx == -1)
            {
                // �S�ẴV�X�e���{�^���̏���񋟂���
                first_idx = 0;
                last_idx = SYSTEM_BUTTON_NUM;
            }
            for (int i = first_idx; i < last_idx; i++)
            {
                out_array[i] = (app_system_buttons[i]) ? true : false;
                app_system_buttons[i] = false;
            }
            return true;
        }
        bool GetSystemButtonState(bool *out_array, const char *name)
        {
            if (!enable)
                return false;
            if (system_buttons_map.count(name))
            {
                int idx = system_buttons_map[name];
                *out_array = (app_system_buttons[idx]) ? true : false;
                app_system_buttons[idx] = false;
                return true;
            }
            return false;
        }
        bool SetButtonOnInterval(int msec)
        {
            if (!enable)
                return false;
            button_on_interval = msec;
            return true;
        }
        bool Enable() const
        {
            return enable;
        }
    };

    HMIDIIN midi_in_handle;          // midi�R���g���[���[�̃n���h��
    MidiController *midi_controller; // �R���g���[���[�̏��

    void CALLBACK MidiInProc(HMIDIIN midi_in_handle, UINT wMsg, DWORD dwInstance, DWORD dwParam1, DWORD dwParam2)
    {
        switch (wMsg)
        {
        case MIM_OPEN:
            printf("MIDI device was opened\n");
            break;
        case MIM_CLOSE:
            printf("MIDI device was closed\n");
            break;
        case MIM_DATA:
        {
            unsigned char status_byte = (dwParam1 & 0x000000ff);
            if (status_byte != 0xb0)
                break;

            unsigned char name = (dwParam1 & 0x0000ff00) >> 8;
            unsigned char value = (dwParam1 & 0x00ff0000) >> 16;
            unsigned int time = dwParam2;
            midi_controller->DecodeMidiEvent(name, value, time);
        }
        break;

        case MIM_LONGDATA:
        case MIM_ERROR:
        case MIM_LONGERROR:
        case MIM_MOREDATA:
        default:
            break;
        }
    }

    // API for Unity

    UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API OpenMidiController()
    {
        // PC�ɐڑ�����Ă���MIDI�R���g���[���̖��O����A�^�[�Q�b�g�R���g���[���[��ID����肷��

        unsigned int target_dev_id = -1;
        unsigned int num_devices = midiInGetNumDevs();

        MMRESULT result;
        MIDIINCAPS midi_in_caps;
        char device_name_buff[32];

        // for (unsigned int dev_id = 0; dev_id < num_devices; ++dev_id)
        // {
        //     UINT cbMidiInCaps;
        //     result = midiInGetDevCaps(
        //         dev_id,
        //         &midi_in_caps,
        //         sizeof(midi_in_caps));
        //     if (result != MMSYSERR_NOERROR)
        //     {
        //         continue;
        //     }

        //     errno_t error = wcstombs_s(
        //         NULL, device_name_buff, 32,
        //         midi_in_caps.szPname, sizeof(midi_in_caps.szPname));
        //     if (error != 0)
        //     {
        //         continue;
        //     }

        //     if (strcmp(device_name_buff, CONTROLLER_NAME) == 0)
        //     {
        //         target_dev_id = dev_id;
        //     }
        // }

        // if (target_dev_id == -1)
        // {
        //     printf("\nDevice is not connected!!!\n");
        //     midi_controller = new MidiController(false);
        //     return false;
        // }
        target_dev_id = 0;
        MMRESULT res;
        WCHAR errmsg[MAXERRORLENGTH];
        char errmsg_buff[MAXERRORLENGTH];

        res = midiInOpen(&midi_in_handle, target_dev_id, (DWORD_PTR)MidiInProc, 0, CALLBACK_FUNCTION);
        if (res != MMSYSERR_NOERROR)
        {
            printf("Cannot open MIDI input device %u", target_dev_id);
            midi_controller = new MidiController(false);
            return false;
        }

        //    printf("Successfully opened a MIDI input device %u.\n", target_dev_id);
        midi_controller = new MidiController();
        midiInStart(midi_in_handle);
        return true;
    }

    UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API CloseMidiController()
    {
        if (!midi_controller->Enable())
        {
            delete midi_controller;
            return false;
        }
        midiInStop(midi_in_handle);
        midiInReset(midi_in_handle);
        midiInClose(midi_in_handle);
        delete midi_controller;
        return true;
    }

    UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API GetSliderValues(float *out_array, int first_idx, int last_idx)
    {
        return midi_controller->GetSliderValue(out_array, first_idx, last_idx);
    }
    UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API GetKnobValues(float *out_array, int first_idx, int last_idx)
    {
        return midi_controller->GetKnobValue(out_array, first_idx, last_idx);
    }
    UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API GetButtonStates(bool *out_array, int first_idx, int last_idx)
    {
        return midi_controller->GetButtonState(out_array, first_idx, last_idx);
    }
    UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API GetSystemButtonStates(bool *out_array, int first_idx, int last_idx)
    {
        return midi_controller->GetSystemButtonState(out_array, first_idx, last_idx);
    }
    UNITY_INTERFACE_EXPORT float UNITY_INTERFACE_API Get1SliderValue(int id)
    {
        float ret = -1;
        midi_controller->GetSliderValue(&ret, id, id + 1);
        return ret;
    }
    UNITY_INTERFACE_EXPORT float UNITY_INTERFACE_API Get1KnobValue(int id)
    {
        float ret = -1;
        midi_controller->GetKnobValue(&ret, id, id + 1);
        return ret;
    }
    UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API Get1ButtonState(int id)
    {
        bool ret = false;
        midi_controller->GetButtonState(&ret, id, id + 1);
        return ret;
    }
    UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API Get1SystemButtonState(const char *name)
    {
        bool ret = false;
        midi_controller->GetSystemButtonState(&ret, name);
        return ret;
    }
    UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API SetButtonOnInterval(int msec)
    {
        return midi_controller->SetButtonOnInterval(msec);
    }

#ifdef UNITY_MODE
}
#endif // !DEBUGMODE