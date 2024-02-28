#include "user_study.h"
#include <filesystem>

namespace fs = std::filesystem;

UserStudy::UserStudy()
{
    rng = std::default_random_engine{};
    rng.seed(std::random_device{}());
    reset();
}

void UserStudy::reset(float initialLatency)
{
    wasHumanSucessfull = false;
    trialIsBaseline = false;
    isFirstTrial = true;
    trialEnded = false;
    resultFile = "";
    baseStep = 1.6f;
    minBaseStep = 0.01f;
    minLatency = 0.0f;
    curLatency = initialLatency;
    attempts = 0;
    pair_attempts = 0;
    reversals = 0;
    successStreak = 0;
    maxSuccessStreak = 10;
    maxReversals = 10;
    jnd = 0.0f;
    latencies.clear();
    trials.clear();
    sessionTimer.start();
    allowedLatencies = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    allowedMotionTypes = std::vector<int>{static_cast<int>(UserStudyMotionModel::TRANSLATION),
                                          static_cast<int>(UserStudyMotionModel::ROTATION),
                                          static_cast<int>(UserStudyMotionModel::DEFORMATION),
                                          static_cast<int>(UserStudyMotionModel::ALL)};
    allowedPairs = std::vector<std::pair<int, int>>{{0, 1}, {0, 2}, {1, 2}};
    for (int i = 0; i < allowedLatencies.size(); i++)
    {
        for (int j = 0; j < allowedMotionTypes.size(); j++)
        {
            for (int k = 0; k < allowedPairs.size(); k++)
            {
                trials.push_back(std::make_tuple(i, j, k, false, 0));
            }
        }
    }
    std::shuffle(trials.begin(), trials.end(), rng);
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

void UserStudy::printTrialInfo(int attempt, bool withUserResponse)
{
    if (attempt < trials.size())
    {
        std::tuple<int, int, int, bool, int> trial = trials[attempt];
        int latencyIndex = std::get<0>(trial);
        int motionTypeIndex = std::get<1>(trial);
        int pairIndex = std::get<2>(trial);
        bool swapPair = std::get<3>(trial);
        float latency = allowedLatencies[latencyIndex];
        int motionType = allowedMotionTypes[motionTypeIndex];
        std::pair<int, int> mypair = allowedPairs[pairIndex];
        if (swapPair)
            mypair = std::make_pair(mypair.second, mypair.first);
        std::vector<std::string> pairNames = {"GT", "vanilla", "ours"};
        std::cout << "latency: " << latency << ", ";
        switch (motionType)
        {
        case 0:
            std::cout << "motionType: translation, ";
            break;
        case 1:
            std::cout << "motionType: rotation, ";
            break;
        case 2:
            std::cout << "motionType: deformation, ";
            break;
        case 3:
            std::cout << "motionType: all, ";
            break;
        default:
            std::cout << "motionType: unknown, ";
            break;
        }
        std::cout << "pair: " << pairNames[mypair.first] << " " << pairNames[mypair.second];
        if (withUserResponse)
        {
            int response = std::get<4>(trial);
            std::cout << ", response: " << pairNames[response];
        }
        std::cout << std::endl;
    }
}

void UserStudy::setSubjectResponse(int response)
{
    // to be called only after randomTrial was called
    // response must be 1 or 2 corresponding to one of the video in the presented pair
    std::tuple<int, int, int, bool, int> trial = trials[attempts - 1];
    int pairIndex = std::get<2>(trial);
    bool swapPair = std::get<3>(trial);
    std::pair<int, int> mypair = allowedPairs[pairIndex];
    if (swapPair)
        mypair = std::make_pair(mypair.second, mypair.first);
    if (response == 1) // user decided first video has more latency
    {
        std::get<4>(trial) = mypair.first;
        trials[attempts - 1] = trial;
    }
    else
    {
        if (response == 2)
        {
            std::get<4>(trial) = mypair.second;
            trials[attempts - 1] = trial;
        }
        else
        {
            std::cout << "unknown response" << std::endl;
            exit(1);
        }
    }
    printTrialInfo(attempts - 1, true);
    saveAttempt(attempts - 1);
    if (attempts >= trials.size())
    {
        trialEnded = true;
        sessionTimer.stop();
    }
}

void UserStudy::randomTrial(float &latency, int &motionType, std::pair<int, int> &pair)
{
    if (attempts < trials.size())
    {
        std::tuple<int, int, int, bool, int> trial = trials[attempts];
        // select a random combination with no repetition
        int latencyIndex = std::get<0>(trial);
        int motionTypeIndex = std::get<1>(trial);
        int pairIndex = std::get<2>(trial);
        latency = allowedLatencies[latencyIndex];
        motionType = allowedMotionTypes[motionTypeIndex];
        std::pair<int, int> mypair = allowedPairs[pairIndex];
        std::uniform_int_distribution<> distr(0, 1);
        if (distr(rng) == 0)
        {
            pair = mypair;
        }
        else
        {
            pair = std::make_pair(mypair.second, mypair.first);
            std::get<3>(trial) = true;
            trials[attempts] = trial;
        }
        printTrialInfo(attempts, false);
    }
    attempts++;
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
        std::cout << "Current video is baseline" << std::endl;
        return 0.0f;
    }
    else
    {
        std::cout << "Current video is not baseline" << std::endl;
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
            successStreak += 1;
            if (successStreak >= maxSuccessStreak)
            {
                std::cout << "Success streak occured, increasing base step" << std::endl;
                baseStep *= 2;
                successStreak = 0;
            }
            else
            {
                if (!wasHumanSucessfull)
                {
                    std::cout << "reversal occured" << std::endl;
                    reversals++;
                    std::cout << "reversals: " << reversals << std::endl;
                    baseStep /= 2;
                    if (baseStep <= minBaseStep)
                    {
                        baseStep = minBaseStep;
                    }
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
            successStreak = 0;
            if (wasHumanSucessfull)
            {
                std::cout << "reversal occured" << std::endl;
                reversals++;
                std::cout << "reversals: " << reversals << std::endl;
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
            sessionTimer.stop();
            jnd = curLatency;
            return;
        }
    }
    std::cout << "New latency: " << curLatency << std::endl;
    std::cout << "Total session time [m]: " << sessionTimer.getElapsedTimeInSec() / 60 << std::endl;
    wasHumanSucessfull = successfullHuman;
    pair_attempts++;
}

void UserStudy::printRandomSessionStats()
{
    std::cout << "Report: " << std::endl;
    std::cout << "Total session time [m]: " << sessionTimer.getElapsedTimeInSec() / 60 << std::endl;
    for (int i = 0; i < trials.size(); i++)
    {
        printTrialInfo(i, true);
    }
}

void UserStudy::setResultFilePath(std::string resultFilePath)
{
    fs::path filePath{resultFilePath};
    if (!fs::exists(filePath))
    {
        resultFile = resultFilePath;
    }
    else
    {
        std::cout << "File already exists, please provide a new file name" << std::endl;
    }
}

void UserStudy::saveAttempt(int attempt)
{
    std::ofstream outfile;
    outfile.open(resultFile, std::ios_base::app);
    std::tuple<int, int, int, bool, int> trial = trials[attempt];
    int latencyIndex = std::get<0>(trial);
    int motionTypeIndex = std::get<1>(trial);
    int pairIndex = std::get<2>(trial);
    // bool pairInverted = std::get<3>(trial);
    int subjectChoice = std::get<4>(trial);
    float latency = allowedLatencies[latencyIndex];
    int motionType = allowedMotionTypes[motionTypeIndex];
    std::pair<int, int> pair = allowedPairs[pairIndex];
    outfile << latency << "," << motionType << "," << pair.first << "," << pair.second << "," << subjectChoice << "\n";
}

void UserStudy::printStats()
{
    std::cout << "Total session time [m]: " << sessionTimer.getElapsedTimeInSec() / 60 << std::endl;
    std::cout << "JND: " << jnd << std::endl;
    std::cout << "Latencies: " << std::endl;
    float accumulator_last_five = 0.0f;
    for (int i = 0; i < latencies.size(); i++)
    {
        std::cout << latencies[i] << " ";
        if (i >= latencies.size() - 5)
        {
            accumulator_last_five += latencies[i];
        }
    }
    std::cout << std::endl;
    std::cout << "Avg last 5: " << accumulator_last_five / 5 << std::endl;
}