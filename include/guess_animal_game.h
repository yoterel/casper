#ifndef GUESS_ANIMAL_GAME_H
#define GUESS_ANIMAL_GAME_H
#include "timer.h"

class GuessAnimalGame
{
public:
    GuessAnimalGame();
    bool isInitialized();
    int getState();
    void resetState();

private:
    bool initialized = false;
    int state = 0;
    float countdown_expired = 10.0f; // minimum seconds to wait before starting a new bake
    Timer bake_countdown;
};

enum class GuessAnimalGameState
{
    COUNTDOWN = 0,
    BAKE = 1,
};
#endif // GUESS_ANIMAL_GAME