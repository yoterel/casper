#ifndef GUESS_CHAR_GAME_H
#define GUESS_CHAR_GAME_H
class GuessAnimalGame
{
public:
    GuessAnimalGame();
    bool isInitialized() { return initialized; };

private:
    bool initialized;
};
#endif // GUESS_CHAR_GAME