#ifndef USER_STUDY_H
#define USER_STUDY_H

#include <vector>
#include "timer.h"
#include <random>

class UserStudy
{
public:
    UserStudy();
    void reset(float initialLatency = 10.0f);
    void randomTrial(float &latency, int &motionType, std::pair<int, int> &pair);
    float randomTrial(int humanChoice);
    void trial(bool successfullHuman);
    void printStats();
    void printRandomSessionStats();
    void saveStats();
    bool getTrialFinished() { return trialEnded; };
    void setSubjectResponse(int response);
    int getAttempts() { return attempts; };
    int getPairAttempts() { return pair_attempts; };

private:
    bool randomize01();
    float getCurrentLatency();
    bool wasHumanSucessfull;
    bool trialIsBaseline;
    bool isFirstTrial;
    bool trialEnded;
    float baseStep;
    float minBaseStep;
    float minLatency;
    float curLatency;
    int attempts;
    int pair_attempts;
    int reversals;
    int maxReversals;
    int successStreak;
    int maxSuccessStreak;
    float jnd;
    Timer sessionTimer;
    std::vector<float> latencies;
    std::vector<float> allowedLatencies;
    std::vector<int> allowedMotionTypes;
    std::vector<std::pair<int, int>> allowedPairs;
    std::vector<std::tuple<int, int, int, bool, bool>> trials; // latency, motionType, pair, pairinverted, subjectChoice
    std::mt19937 rng;
};

enum class UserStudyMotionModel
{
    TRANSLATION = 0,
    ROTATION = 1,
    DEFORMATION = 2,
    ALL = 3,
};

#endif // USER_STUDY_H