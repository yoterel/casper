#ifndef GUESS_CHAR_GAME_H
#define GUESS_CHAR_GAME_H

#include <vector>
#include "timer.h"
#include <glm/glm.hpp>
#include <algorithm>
#include <random>
#include <unordered_map>

class GuessCharGame
{
public:
    GuessCharGame();
    // void setPoses(std::vector<std::vector<glm::mat4>> required_poses);
    void setBonesVisible(bool visible) { bonesVisible = visible; };
    // bool isCountingDown() { return countDownInProgress; };
    int getState();
    int getCountdownTime();
    void setNewChars();
    int getRandomChars(std::string &chars);
    void setResponse(bool playerCorrect);
    void setAllExtended(bool all_extended);
    glm::vec2 getRandomLocation();
    std::unordered_map<std::string, glm::vec2> getNumberLocations(bool frontView = false);
    int getTotalSessionCounter() { return totalSessionCounter; };
    void setScore(float score);
    void printScore();
    void reset(bool randomSession, std::string comment = "");
    void hardReset();
    bool isInitialized() { return initialized; };

private:
    void setState(int state) { curState = state; };
    int curState;
    bool initialized;
    bool countDownInProgress;
    bool bonesVisible;
    bool allExtended;
    bool roundFinished;
    bool curSessionRandom;
    int totalSessionCounter;
    // int curPoseIndex;
    Timer countDownTimer;
    Timer delayTimer;
    Timer totalTime;
    // std::vector<std::vector<glm::mat4>> poses;
    std::vector<float> cur_scores;
    std::unordered_map<std::string, glm::vec2> fingerLocationsUV;
    std::unordered_map<std::string, glm::vec2> curFingerLocationsUV;
    float breakTime;
    std::string curChars;
    std::vector<std::string> selectedChars;
    std::vector<int> selectedIndices;
    int curCorrectIndex;
    int gameMode;
    std::mt19937 rng;
};

enum class GuessCharGameState
{
    WAIT_FOR_USER = 0,
    COUNTDOWN = 1,
    PLAY = 2,
    WAIT = 3,
    END = 4,
};

#endif // GUESS_CHAR_GAME