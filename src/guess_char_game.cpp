#include "guess_char_game.h"
#include <iostream>

GuessCharGame::GuessCharGame()
{
    rng = std::default_random_engine{};
    // reset();
}

int GuessCharGame::getState()
{
    switch (curState)
    {
    case static_cast<int>(GuessCharGameState::WAIT_FOR_USER):
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
                    setState(static_cast<int>(GuessCharGameState::COUNTDOWN));
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
    case static_cast<int>(GuessCharGameState::COUNTDOWN):
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
                setState(static_cast<int>(GuessCharGameState::PLAY));
            }
        }
        break;
    }
    case static_cast<int>(GuessCharGameState::PLAY):
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
    case static_cast<int>(GuessCharGameState::WAIT):
    {
        if (!countDownInProgress)
        {
            if (roundFinished)
            {
                countDownInProgress = true;
                countDownTimer.start();
            }
        }
        else
        {
            if (allExtended)
            {
                if (countDownTimer.getElapsedTimeInSec() >= 1)
                {
                    roundFinished = false;
                    countDownInProgress = false;
                    countDownTimer.stop();
                    breakTime += countDownTimer.getElapsedTimeInSec();
                    setState(static_cast<int>(GuessCharGameState::PLAY));
                }
            }
        }
        break;
    }
    case static_cast<int>(GuessCharGameState::END):
    {
        reset();
        setState(static_cast<int>(GuessCharGameState::WAIT_FOR_USER));
        break;
    }
    default:
        break;
    }
    return curState;
}

int GuessCharGame::getCountdownTime()
{
    return static_cast<int>(countDownTimer.getElapsedTimeInSec());
}

void GuessCharGame::setRandomChars()
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

int GuessCharGame::getRandomChars(std::string &chars)
{
    chars = curChars;
    return curCorrectIndex;
}

void GuessCharGame::setAllExtended(bool all_extended)
{
    allExtended = all_extended;
}

void GuessCharGame::setResponse(bool playerCorrect)
{
    if (bonesVisible)
    {
        if (!roundFinished)
        {
            std::cout << cur_scores.size() << "/"
                      << "20";
            if (playerCorrect)
            {
                std::cout << ", CORRECT" << std::endl;
                setScore(1.0f);
            }
            else
            {
                std::cout << ", MISTAKE" << std::endl;
                setScore(0.0f);
            }
            if (cur_scores.size() >= 20)
            {
                totalTime.stop();
                printScore();
                setState(static_cast<int>(GuessCharGameState::END));
            }
            else
            {
                setRandomChars();
                setState(static_cast<int>(GuessCharGameState::WAIT));
            }
            roundFinished = true;
        }
    }
}

void GuessCharGame::setScore(float score)
{
    cur_scores.push_back(score);
}

glm::vec2 GuessCharGame::getRandomLocation()
{
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
    return glm::vec2(dist(rng), dist(rng));
}

std::unordered_map<std::string, glm::vec2> GuessCharGame::getNumberLocations()
{
    if (!delayTimer.isRunning())
        delayTimer.start();
    if (delayTimer.getElapsedTimeInSec() >= 1.0)
    {
        curFingerLocationsUV["index"] = fingerLocationsUV["index"] + getRandomLocation();
        curFingerLocationsUV["middle"] = fingerLocationsUV["middle"] + getRandomLocation();
        curFingerLocationsUV["ring"] = fingerLocationsUV["ring"] + getRandomLocation();
        curFingerLocationsUV["pinky"] = fingerLocationsUV["pinky"] + getRandomLocation();
        delayTimer.start();
    }
    return curFingerLocationsUV;
}

void GuessCharGame::reset()
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
    // glm::vec2 palm_ndc = glm::vec2(-0.66f, -0.683f);
    // glm::vec2 index_ndc = glm::vec2(-0.425f, 0.847f);
    // glm::vec2 middle_ndc = glm::vec2(0.822f, 0.729f);
    // glm::vec2 ring_ndc = glm::vec2(-0.966f, 0.282f);
    // glm::vec2 pinky_ndc = glm::vec2(0.14f, 0.894);
    glm::vec2 palm_ndc2 = glm::vec2(-0.551f, -0.579f);
    glm::vec2 index_ndc2 = glm::vec2(0.59f, 0.407f);
    glm::vec2 middle_ndc2 = glm::vec2(-0.029f, 0.464f);
    glm::vec2 ring_ndc2 = glm::vec2(0.502f, -0.565f);
    glm::vec2 pinky_ndc2 = glm::vec2(0.449f, -0.131f);
    fingerLocationsUV["palm"] = palm_ndc2;
    fingerLocationsUV["index"] = index_ndc2;
    fingerLocationsUV["middle"] = middle_ndc2;
    fingerLocationsUV["ring"] = ring_ndc2;
    fingerLocationsUV["pinky"] = pinky_ndc2;
    curFingerLocationsUV = fingerLocationsUV;
}

void GuessCharGame::printScore()
{
    float avgScore = 0.0f;
    for (auto &score : cur_scores)
    {
        avgScore += score;
    }
    avgScore /= cur_scores.size();
    std::cout << "##### Results #####" << std::endl;
    std::cout << "Average score: " << avgScore << std::endl;
    std::cout << "Total time: " << totalTime.getElapsedTimeInSec() - breakTime << " seconds" << std::endl;
    std::cout << "###################" << std::endl;
}