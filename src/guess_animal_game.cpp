#include "guess_animal_game.h"

GuessAnimalGame::GuessAnimalGame()
{
    initialized = true;
    bake_countdown.start();
};

bool GuessAnimalGame::isInitialized()
{
    return initialized;
};

int GuessAnimalGame::getState()
{
    switch (state)
    {
    case static_cast<int>(GuessAnimalGameState::COUNTDOWN):
    {
        double seconds = bake_countdown.getElapsedTimeInSec();
        if (seconds > countdown_expired)
        {
            state = static_cast<int>(GuessAnimalGameState::BAKE);
        }
        break;
    }
    default:
        break;
    }

    return state;
};
void GuessAnimalGame::resetState()
{
    state = static_cast<int>(GuessAnimalGameState::COUNTDOWN);
    bake_countdown.start();
};
