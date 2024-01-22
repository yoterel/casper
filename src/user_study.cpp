#include "user_study.h"
#include <iostream>
#include <random>

UserStudy::UserStudy()
{
    reset();
}

void UserStudy::reset()
{
    wasHumanSucessfull = false;
    trialIsBaseline = false;
    isFirstTrial = true;
    trialEnded = false;
    baseStep = 1.6f;
    minBaseStep = 0.01f;
    minLatency = 0.1f;
    curLatency = 10.f;
    attempts = 0;
    pair_attempts = 0;
    reversals = 0;
    maxReversals = 10;
    jnd = 0.0f;
    latencies.clear();
}

float UserStudy::getCurrentLatency()
{
    if (!trialEnded)
        latencies.push_back(curLatency);
    return curLatency;
}

bool UserStudy::randomize01()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(0, 1);
    if (distr(gen) == 0)
    {
        return true;
    }
    else
    {
        return false;
    }
}
float UserStudy::randomTrial(int humanChoice)
{
    if (attempts == 0)
    {
        trialIsBaseline = randomize01();
    }
    else
    {
        if ((attempts % 2 == 0))
        {
            if (humanChoice == 1)
            {
                if (trialIsBaseline)
                {
                    std::cout << "succees. human: first, gt: first" << std::endl;
                    trial(true);
                }
                else
                {
                    std::cout << "fail. human: first, gt: second" << std::endl;
                    trial(false);
                }
            }
            else
            {
                if (trialIsBaseline)
                {
                    std::cout << "fail. human: second, gt: first" << std::endl;
                    trial(false);
                }
                else
                {
                    std::cout << "sucess. human: first, gt: first" << std::endl;
                    trial(true);
                }
            }
            trialIsBaseline = randomize01();
        }
        else
        {
            trialIsBaseline = !trialIsBaseline;
        }
    }
    attempts++;
    if (trialIsBaseline)
    {
        return 0.0f;
    }
    else
    {
        return getCurrentLatency();
    }
}

void UserStudy::trial(bool successfullHuman)
{

    if (trialEnded)
    {
        return;
    }
    if (isFirstTrial)
    {
        isFirstTrial = false;
        if (successfullHuman)
        {
            curLatency -= baseStep;
        }
        else
        {
            curLatency += 3 * baseStep;
        }
    }
    else
    {
        if (successfullHuman)
        {
            if (!wasHumanSucessfull)
            {
                std::cout << "reversal occured" << std::endl;
                reversals++;
                baseStep /= 2;
                if (baseStep <= minBaseStep)
                {
                    baseStep = minBaseStep;
                }
            }
            curLatency -= baseStep;
            if (curLatency <= minLatency)
            {
                curLatency = minLatency;
            }
        }
        else
        {
            if (wasHumanSucessfull)
            {
                std::cout << "reversal occured" << std::endl;
                reversals++;
                baseStep /= 2;
                if (baseStep <= minBaseStep)
                {
                    baseStep = minBaseStep;
                }
            }
            curLatency += 3 * baseStep;
        }
        if (reversals >= maxReversals)
        {
            trialEnded = true;
            jnd = curLatency;
        }
    }
    std::cout << "new latency: " << curLatency << std::endl;
    wasHumanSucessfull = successfullHuman;
    pair_attempts++;
}

void UserStudy::printStats()
{
    std::cout << "JND: " << jnd << std::endl;

    std::cout << "Latencies: ";
    float accumulator_last_five = 0.0f;
    for (int i = 0; i < latencies.size(); i++)
    {
        std::cout << latencies[i] << " ";
        if (i >= latencies.size() - 5)
        {
            accumulator_last_five += latencies[i];
        }
    }
    std::cout << "JND, last 5: " << accumulator_last_five / 5 << std::endl;
}