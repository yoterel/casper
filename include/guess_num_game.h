#ifndef GUESS_NUM_GAME_H
#define GUESS_NUM_GAME_H

#include <vector>
#include "timer.h"
#include <glm/glm.hpp>
#include <algorithm>
#include <random>

class GuessNumGame
{
public:
    GuessNumGame();
    // void setPoses(std::vector<std::vector<glm::mat4>> required_poses);
    void setBonesVisible(bool visible) { bonesVisible = visible; };
    // bool isCountingDown() { return countDownInProgress; };
    int getState();
    int getCountdownTime();
    void setRandomChars();
    int getRandomChars(std::string &chars);
    void setResponse(bool playerCorrect);
    void setAllExtended(bool all_extended);
    glm::vec2 getRandomLocation();
    // std::vector<glm::mat4> getPose();
    void setScore(float score);
    void printScore();
    void reset(bool shuffle = true);

private:
    void setState(int state) { curState = state; };
    int curState;
    bool countDownInProgress;
    bool bonesVisible;
    bool allExtended;
    bool roundFinished;
    // int curPoseIndex;
    Timer countDownTimer;
    Timer totalTime;
    // std::vector<std::vector<glm::mat4>> poses;
    std::vector<float> cur_scores;
    float breakTime;
    std::string curChars;
    int curCorrectIndex;
    int gameMode;
    std::mt19937 rng;
};

enum class GuessNumGameState
{
    WAIT_FOR_USER = 0,
    COUNTDOWN = 1,
    PLAY = 2,
    END = 3,
};

#endif // GUESS_NUM_GAME