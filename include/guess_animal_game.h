#ifndef GUESS_ANIMAL_GAME_H
#define GUESS_ANIMAL_GAME_H
class GuessAnimalGame
{
public:
    GuessAnimalGame();
    bool isInitialized() { return initialized; };

private:
    bool initialized;
};
#endif // GUESS_ANIMAL_GAME