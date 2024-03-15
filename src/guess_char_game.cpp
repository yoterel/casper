#include "guess_char_game.h"
#include <iostream>

GuessCharGame::GuessCharGame()
{
    rng = std::default_random_engine{};
    totalSessionCounter = 0;
    initialized = false;
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
                setNewChars();
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
        // reset();
        // setState(static_cast<int>(GuessCharGameState::WAIT_FOR_USER));
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

void GuessCharGame::setNewChars()
{
    if (!curSessionRandom)
    {
        curChars = selectedChars[cur_scores.size()];
        curCorrectIndex = selectedIndices[cur_scores.size()];
    }
    else
    {
        std::string allChars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ012345678!@#$%^&?";
        // shuffle the string
        std::shuffle(allChars.begin(), allChars.end(), rng);
        // get the first 4 characters
        std::string randomChars = allChars.substr(0, 4);
        curChars = randomChars;

        selectedChars.push_back(curChars);
        // select a random index from 0 to 3
        curCorrectIndex = std::uniform_int_distribution<int>(0, 3)(rng);
        selectedIndices.push_back(curCorrectIndex);
    }
    // std::cout << "New characters: " << curChars << std::endl;
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
            // std::cout << cur_scores.size() << "/"
            //           << "20";
            if (playerCorrect)
            {
                // std::cout << ", CORRECT" << std::endl;
                setScore(1.0f);
            }
            else
            {
                // std::cout << ", MISTAKE" << std::endl;
                setScore(0.0f);
            }
            // auto tTime = totalTime.getElapsedTimeInSec();
            // std::cout << "total time: " << tTime << std::endl;
            // std::cout << "break time: " << breakTime << std::endl;
            // std::cout << "play time: " << tTime - breakTime << std::endl;
            if (cur_scores.size() >= 20)
            {
                totalTime.stop();
                printScore();
                setState(static_cast<int>(GuessCharGameState::END));
            }
            else
            {
                setNewChars();
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

std::unordered_map<std::string, glm::vec2> GuessCharGame::getNumberLocations(bool frontView)
{
    if (!delayTimer.isRunning())
        delayTimer.start();
    if (delayTimer.getElapsedTimeInSec() >= 1.0)
    {
        if (frontView)
        {
            fingerLocationsUV["palm"] = glm::vec2(-0.554f, -0.679f);
            fingerLocationsUV["index"] = glm::vec2(0.648f, 0.390f);
            fingerLocationsUV["middle"] = glm::vec2(0.083, 0.461f);
            fingerLocationsUV["ring"] = glm::vec2(0.5f, -0.521f);
            fingerLocationsUV["pinky"] = glm::vec2(0.47f, -0.117);
        }
        else
        {
            fingerLocationsUV["palm"] = glm::vec2(-0.551f, -0.579f);
            fingerLocationsUV["index"] = glm::vec2(0.59f, 0.407f);
            fingerLocationsUV["middle"] = glm::vec2(-0.029f, 0.464f);
            fingerLocationsUV["ring"] = glm::vec2(0.502f, -0.565f);
            fingerLocationsUV["pinky"] = glm::vec2(0.449f, -0.131f);
        }
        curFingerLocationsUV["index"] = fingerLocationsUV["index"] + getRandomLocation();
        curFingerLocationsUV["middle"] = fingerLocationsUV["middle"] + getRandomLocation();
        curFingerLocationsUV["ring"] = fingerLocationsUV["ring"] + getRandomLocation();
        curFingerLocationsUV["pinky"] = fingerLocationsUV["pinky"] + getRandomLocation();
        delayTimer.start();
    }
    return curFingerLocationsUV;
}

void GuessCharGame::reset(bool randomSession, std::string comment)
{
    initialized = true;
    totalSessionCounter++;
    std::cout << std::endl
              << "Resetting game..." << std::endl;
    std::cout << "Session Number: " << totalSessionCounter << std::endl;
    std::cout << "Type: " << comment << ", RandomChars: " << randomSession << std::endl;
    curState = 0;
    countDownInProgress = false;
    bonesVisible = false;
    cur_scores.clear();
    curChars = "";
    curSessionRandom = randomSession;
    if (curSessionRandom)
    {
        selectedIndices.clear();
        selectedChars.clear();
    }
    allExtended = false;
    roundFinished = false;
    curCorrectIndex = 0;
    breakTime = 0.0f;
    gameMode = 1;
    // palm
    // glm::vec2 palm_ndc = glm::vec2(-0.5f, -0.731f);
    // glm::vec2 index_ndc = glm::vec2(0.648f, 0.390f);
    // glm::vec2 middle_ndc = glm::vec2(0.083, 0.461f);
    // glm::vec2 ring_ndc = glm::vec2(0.5f, -0.521f);
    // glm::vec2 pinky_ndc = glm::vec2(0.47f, -0.117);
    // backhand
    glm::vec2 palm_ndc = glm::vec2(-0.551f, -0.579f);
    glm::vec2 index_ndc = glm::vec2(0.59f, 0.407f);
    glm::vec2 middle_ndc = glm::vec2(-0.029f, 0.464f);
    glm::vec2 ring_ndc = glm::vec2(0.502f, -0.565f);
    glm::vec2 pinky_ndc = glm::vec2(0.449f, -0.131f);
    fingerLocationsUV["palm"] = palm_ndc;
    fingerLocationsUV["index"] = index_ndc;
    fingerLocationsUV["middle"] = middle_ndc;
    fingerLocationsUV["ring"] = ring_ndc;
    fingerLocationsUV["pinky"] = pinky_ndc;
    curFingerLocationsUV = fingerLocationsUV;
}

void GuessCharGame::hardReset()
{
    initialized = false;
    totalSessionCounter = 0;
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