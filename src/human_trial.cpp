#include "human_trial.h"
#include <iostream>
#include <random>

HumanTrial::HumanTrial()
{
    reset();
}

float HumanTrial::getCurrentLatency()
{
    if (!trialEnded)
        latencies.push_back(curLatency);
    return curLatency;
}

bool HumanTrial::randomize01()
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
float HumanTrial::randomTrial(bool humanThinksFirstIsBaseline)
{
    if ((attempts % 2 == 0) && (attempts != 0))
    {
        if (trialIsBaseline)
        {
            trial(false);
        }
        else
        {
            trial(true);
        }
        trialIsBaseline = randomize01();
    }
    else
    {
        trialIsBaseline = !trialIsBaseline;
    }
    if (trialIsBaseline)
    {
        return 0.0f;
    }
    else
    {
        return getCurrentLatency();
    }
    attempts += 1;
}

void HumanTrial::trial(bool successfullHuman)
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
            curLatency += 3 * baseStep;
        }
        else
        {
            curLatency -= baseStep;
        }
    }
    else
    {
        if (successfullHuman)
        {
            curLatency += 3 * baseStep;
            if (!wasHumanSucessfull)
            {
                reversals++;
                baseStep /= 2;
                if (baseStep >= minBaseStep)
                {
                    baseStep = minBaseStep;
                }
            }
        }
        else
        {
            curLatency -= baseStep;
            if (curLatency <= minLatency)
            {
                curLatency = minLatency;
            }
            if (wasHumanSucessfull)
            {
                reversals++;
                baseStep /= 2;
                if (baseStep >= minBaseStep)
                {
                    baseStep = minBaseStep;
                }
            }
        }
        if (reversals >= 10)
        {
            printStats();
            trialEnded = true;
            return;
        }
    }
    wasHumanSucessfull = successfullHuman;
    pair_attempts++;
}

void HumanTrial::printStats()
{
    std::cout << "JND: " << jnd << std::endl;
    std::cout << "Latencies: ";
    for (int i = 0; i < latencies.size(); i++)
    {
        std::cout << latencies[i] << " ";
    }
    std::cout << std::endl;
}

void HumanTrial::reset()
{
    trialIsBaseline = false;
    isFirstTrial = true;
    trialEnded = false;
    wasHumanSucessfull = false;
    baseStep = 1.6f;
    minLatency = 0.1f;
    minBaseStep = 0.1f;
    curLatency = 40.f;
    latencies.clear();
    attempts = 0;
    pair_attempts = 0;
    reversals = 0;
}