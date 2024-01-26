#ifndef USER_STUDY_H
#define USER_STUDY_H

#include <vector>
class UserStudy
{
public:
    UserStudy();
    void reset();
    float randomTrial(int humanChoice);
    void trial(bool successfullHuman);
    void printStats();
    bool getTrialFinished() { return trialEnded; };
    int getAttempts() { return attempts; };

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
    float jnd;
    std::vector<float> latencies;
};
#endif // USER_STUDY_H