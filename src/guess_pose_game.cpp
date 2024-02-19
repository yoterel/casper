#include "guess_pose_game.h"
#include <iostream>

GuessPoseGame::GuessPoseGame()
{
    rng = std::default_random_engine{};
    // reset();
}

int GuessPoseGame::getState()
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
        switch (gameMode)
        {
        case static_cast<int>(GameMode::AVG_SCORE_OVER_DURATION):
        {
            if (!countDownInProgress)
            {
                if (bonesVisible)
                {
                    countDownInProgress = true;
                    timer.start();
                }
            }
            else
            {
                if (!bonesVisible)
                {
                    timer.stop();
                }
                else
                {
                    timer.resume();
                }
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
            break;
        }
        case static_cast<int>(GameMode::TIME_UNTIL_THRESHOLD):
        {
            if (!countDownInProgress)
            {
                if (bonesVisible)
                {
                    countDownInProgress = true;
                    timer.start();
                }
            }
            else
            {
                if (!bonesVisible)
                {
                    timer.stop();
                }
                else
                {
                    timer.resume();
                }
            }
            bool passedTreashold = false;
            float threshold = 0.8f;
            for (int i = 0; i < cur_scores.size(); i++)
            {
                if (cur_scores[i] >= threshold)
                {
                    float time_until_threshold = static_cast<float>(timer.getElapsedTimeInSec());
                    countDownInProgress = false;
                    score_per_pose.push_back(time_until_threshold);
                    curPoseIndex += 1;
                    setState(static_cast<int>(GameState::COUNTDOWN));
                    break;
                }
            }
            cur_scores.clear();
            break;
        }
        default:
            break;
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
        reset();
        setState(static_cast<int>(GameState::WAIT_FOR_USER));
        break;
    }
    default:
        break;
    }
    return curState;
}

int GuessPoseGame::getCountdownTime()
{
    return static_cast<int>(timer.getElapsedTimeInSec());
}
std::vector<glm::mat4> GuessPoseGame::getPose()
{
    if (curPoseIndex >= poses.size())
        return poses[poses.size() - 1];
    return poses[curPoseIndex];
}

void GuessPoseGame::setScore(float score)
{
    cur_scores.push_back(score);
}

void GuessPoseGame::setPoses(std::vector<std::vector<glm::mat4>> required_poses)
{
    poses = required_poses;
}

void GuessPoseGame::reset(bool shuffle)
{
    curState = 0;
    curPoseIndex = 0;
    countDownInProgress = false;
    bonesVisible = false;
    cur_scores.clear();
    score_per_pose.clear();
    if (shuffle)
        std::shuffle(std::begin(poses), std::end(poses), rng);
    // poses.clear();
    gameMode = 1;
}

void GuessPoseGame::printScore()
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