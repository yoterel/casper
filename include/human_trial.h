#include <vector>
class HumanTrial
{
public:
    HumanTrial();
    float randomTrial(bool humanThinksFirstIsBaseline);
    bool randomize01();
    void trial(bool successfullHuman);
    void printStats();
    void reset();

private:
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
    float jnd;
    std::vector<float> latencies;
};