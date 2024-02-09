#ifndef GAME_H
#define GAME_H

#include <vector>
#include "timer.h"
#include <glm/glm.hpp>
#include <algorithm>
#include <random>

class Game
{
public:
    Game();
    void setPoses(std::vector<std::vector<glm::mat4>> required_poses);
    void setBonesVisible(bool visible) { bonesVisible = visible; };
    // bool isCountingDown() { return countDownInProgress; };
    int getState();
    int getCountdownTime();
    std::vector<glm::mat4> getPose();
    void setScore(float score);
    void printScore();
    void reset(bool shuffle = true);

private:
    void setState(int state) { curState = state; };
    int curState;
    bool countDownInProgress;
    bool bonesVisible;
    int curPoseIndex;
    Timer timer;
    std::vector<std::vector<glm::mat4>> poses;
    std::vector<float> cur_scores;
    std::vector<float> score_per_pose;
    int gameMode;
    std::mt19937 rng;
};

enum class GameState
{
    WAIT_FOR_USER = 0,
    COUNTDOWN = 1,
    PLAY = 2,
    END = 3,
};

enum class GameMode
{
    AVG_SCORE_OVER_DURATION = 0,
    TIME_UNTIL_THRESHOLD = 1,
};

#endif // GAME_H