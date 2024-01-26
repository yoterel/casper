#include "game.h"
#include <iostream>

Game::Game()
{
    reset();
}

int Game::getState()
{
    switch (curState)
    {
    case static_cast<int>(GameState::WAIT_FOR_USER):
    {
        if (bonesVisible)
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
                    setState(static_cast<int>(GameState::COUNTDOWN));
                }
            }
        }
        else
        {
            countDownInProgress = false;
            timer.stop();
        }
        break;
    }
    case static_cast<int>(GameState::COUNTDOWN):
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
                setState(static_cast<int>(GameState::PLAY));
            }
        }
        break;
    }
    case static_cast<int>(GameState::PLAY):
    {
        if (!countDownInProgress)
        {
            countDownInProgress = true;
            timer.start();
        }
        if (timer.getElapsedTimeInSec() >= 10)
        {
            countDownInProgress = false;
            timer.stop();
            curPoseIndex += 1;
            float avgScore = 0.0f;
            for (auto score : cur_scores)
            {
                avgScore += score;
            }
            avgScore /= cur_scores.size();
            score_per_pose.push_back(avgScore);
            cur_scores.clear();
        }
        if (curPoseIndex >= poses.size())
        {
            printScore();
            setState(static_cast<int>(GameState::END));
        }
        break;
    }
    case static_cast<int>(GameState::END):
    {
        break;
    }
    default:
        break;
    }
    return curState;
}

std::vector<glm::mat4> Game::getPose()
{
    if (curPoseIndex >= poses.size())
        return poses[poses.size() - 1];
    return poses[curPoseIndex];
}

void Game::setScore(float score)
{
    cur_scores.push_back(score);
}

void Game::reset()
{
    curState = 0;
    curPoseIndex = 0;
    countDownInProgress = false;
    bonesVisible = false;
    cur_scores.clear();
    score_per_pose.clear();
    poses.clear();
}

void Game::printScore()
{
    float avgScore = 0.0f;
    std::cout << "------------------------" << std::endl;
    for (auto score : score_per_pose)
    {
        std::cout << "Score: " << score << std::endl;
        avgScore += score;
    }
    std::cout << std::endl
              << "------------------------" << std::endl;
    avgScore /= score_per_pose.size();
    std::cout << "Average Score: " << avgScore << std::endl;
}