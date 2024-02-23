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

        if (bonesVisible)
        {
            if (!countDownInProgress)
            {
                countDownInProgress = true;
                countDownTimer.start();
            }
            else
            {
                if (countDownTimer.getElapsedTimeInSec() >= 3)
                {
                    countDownInProgress = false;
                    countDownTimer.stop();
                    setState(static_cast<int>(GuessNumGameState::COUNTDOWN));
                }
            }
        }
        else
        {
            countDownInProgress = false;
            countDownTimer.stop();
        }
        break;
    }
    case static_cast<int>(GuessNumGameState::COUNTDOWN):
    {
        if (!countDownInProgress)
        {
            countDownInProgress = true;
            countDownTimer.start();
        }
        else
        {
            if (countDownTimer.getElapsedTimeInSec() >= 3)
            {
                countDownInProgress = false;
                countDownTimer.stop();
                setRandomChars();
                totalTime.start();
                setState(static_cast<int>(GuessNumGameState::PLAY));
            }
        }
        break;
    }
    case static_cast<int>(GuessNumGameState::PLAY):
    {
        if (!bonesVisible)
        {
            if (!countDownInProgress)
            {
                countDownTimer.start();
                countDownInProgress = true;
            }
        }
        else
        {
            if (countDownInProgress)
            {
                countDownInProgress = false;
                countDownTimer.stop();
                breakTime += countDownTimer.getElapsedTimeInSec();
            }
        }

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
    return static_cast<int>(countDownTimer.getElapsedTimeInSec());
}

void GuessNumGame::setRandomChars()
{
    std::string allChars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&?";
    // shuffle the string
    std::shuffle(allChars.begin(), allChars.end(), rng);
    // get the first 4 characters
    std::string randomChars = allChars.substr(0, 4);
    curChars = randomChars;
    // select a random index from 0 to 3
    curCorrectIndex = std::uniform_int_distribution<int>(0, 3)(rng);
}

int GuessNumGame::getRandomChars(std::string &chars)
{
    chars = curChars;
    return curCorrectIndex;
}

void GuessNumGame::setAllExtended(bool all_extended)
{
    allExtended = all_extended;
    if (allExtended && roundFinished)
    {
        roundFinished = false;
    }
}

void GuessNumGame::setResponse(bool playerCorrect)
{
    if (bonesVisible)
    {
        if (!roundFinished)
        {
            if (playerCorrect)
            {
                // std::cout << "CORRECT" << std::endl;
                setScore(1.0f);
            }
            else
            {
                // std::cout << "MISTAKE" << std::endl;
                setScore(0.0f);
            }
            if (cur_scores.size() >= 20)
            {
                totalTime.stop();
                printScore();
                setState(static_cast<int>(GuessNumGameState::END));
            }
            else
            {
                countDownInProgress = true;
                countDownTimer.start();
                setRandomChars();
            }
            roundFinished = true;
        }
    }
}

void GuessNumGame::setScore(float score)
{
    cur_scores.push_back(score);
}

glm::vec2 GuessNumGame::getRandomLocation()
{
    std::uniform_real_distribution<float> dist(-0.02f, 0.02f);
    return glm::vec2(dist(rng), dist(rng));
}

void GuessNumGame::reset(bool shuffle)
{
    curState = 0;
    countDownInProgress = false;
    bonesVisible = false;
    cur_scores.clear();
    curChars = "";
    allExtended = false;
    roundFinished = false;
    curCorrectIndex = 0;
    breakTime = 0.0f;
    gameMode = 1;
}

void GuessNumGame::printScore()
{
    float avgScore = 0.0f;
    for (auto &score : cur_scores)
    {
        avgScore += score;
    }
    avgScore /= cur_scores.size();
    std::cout << "Average score: " << avgScore << std::endl;
    std::cout << "Total time: " << totalTime.getElapsedTimeInSec() - breakTime << " seconds" << std::endl;
}