#include "guess_num_game.h"
#include <iostream>

GuessNumGame::GuessNumGame()
{
    rng = std::default_random_engine{};
    // reset();
}

int GuessNumGame::getState()
{
    switch (curState)
    {
    case static_cast<int>(GuessNumGameState::WAIT_FOR_USER):
    {

        countDownInProgress = false;
        timer.stop();
        break;
    }
    case static_cast<int>(GuessNumGameState::COUNTDOWN):
    {
        if (!countDownInProgress)
        {
            countDownInProgress = true;
            timer.start();
        }
        else
        {
            if (timer.getElapsedTimeInSec() >= 3)
            {
                countDownInProgress = false;
                timer.stop();
                setState(static_cast<int>(GuessNumGameState::PLAY));
            }
        }
        break;
    }
    case static_cast<int>(GuessNumGameState::PLAY):
    {
        break;
    }
    case static_cast<int>(GuessNumGameState::END):
    {
        reset();
        setState(static_cast<int>(GuessNumGameState::WAIT_FOR_USER));
        break;
    }
    default:
        break;
    }
    return curState;
}

int GuessNumGame::getCountdownTime()
{
    return static_cast<int>(timer.getElapsedTimeInSec());
}

void GuessNumGame::setScore(float score)
{
    cur_scores.push_back(score);
}

void GuessNumGame::reset(bool shuffle)
{
    curState = 0;
    countDownInProgress = false;
    bonesVisible = false;
    cur_scores.clear();
    score_per_pose.clear();
    gameMode = 1;
}

void GuessNumGame::printScore()
{
}