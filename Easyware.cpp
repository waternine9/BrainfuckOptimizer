#include <iostream>
#include <vector>
#include <random>
#include <Windows.h>
#include <chrono>
#include <array>
#include <thread>
#include <omp.h>

std::random_device RandDevice;
std::mt19937 RandGen(RandDevice());
std::uniform_real_distribution<float> RandDist(-1.0f, 1.0f);

struct Layer
{
    std::vector<float> Nodes;
    std::vector<float> Biases;
    std::vector<std::vector<float>> Weights;
};

std::vector<float> RandVec(int size)
{
    std::vector<float> Output;
    while (size--)
    {
        Output.push_back(RandDist(RandGen));
    }
    return Output;
}

std::vector<std::vector<float>> RandVec2D(int size_deep, int size_shallow)
{
    std::vector<std::vector<float>> Output;
    while (size_shallow--)
    {
        Output.push_back(RandVec(size_deep));
    }
    return Output;
}

void RandVecIze(std::vector<float>* vec, float mag)
{
    for (float& n : *vec)
    {
        n += RandDist(RandGen) * mag;
    }
}

void RandVecIze2D(std::vector<std::vector<float>>* vec, float mag)
{
    for (std::vector<float>& n : *vec)
    {
        RandVecIze(&n, mag);
    }
}

Layer RandLayer(int CurSize, int PrevSize)
{
    Layer Out;
    Out.Nodes = std::vector<float>(CurSize, 0.0f);
    Out.Biases = RandVec(CurSize);
    Out.Weights = RandVec2D(PrevSize, CurSize);
    return Out;
}

Layer NewLayer(int CurSize, int PrevSize)
{
    Layer Out;
    Out.Nodes = std::vector<float>(CurSize, 0.0f);
    return Out;
}

float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

void LayerForward(Layer* Prev, Layer* Cur)
{
    for (int i = 0; i < Cur->Nodes.size(); i++)
    {
        float Val = Cur->Biases[i];
        for (int j = 0; j < Cur->Weights[0].size();j++)
        {
            Val += Prev->Nodes[j] * Cur->Weights[i][j];
        }
        Cur->Nodes[i] = sigmoid(Val);
    }
}

class Network
{
private:
    std::vector<Layer> _Layers;
public:
    float fitness;
    void AddLayer(int Size)
    {
        if (_Layers.size() == 0)
        {
            _Layers.push_back(NewLayer(Size, 1));
        }
        else
        {
            _Layers.push_back(RandLayer(Size, _Layers[_Layers.size() - 1].Nodes.size()));
        }
    }
    void Forward(std::vector<float> Inputs)
    {
        _Layers[0].Nodes = Inputs;
        for (int i = 1; i < _Layers.size(); i++)
        {
            LayerForward(&_Layers[i - 1], &_Layers[i]);
        }
    }
    std::vector<float> Output()
    {
        return _Layers[_Layers.size() - 1].Nodes;
    }
    std::vector<std::vector<float>> GetWeights(int LayerIdx)
    {
        return _Layers[LayerIdx].Weights;
    }
    std::vector<float> GetBiases(int LayerIdx)
    {
        return _Layers[LayerIdx].Biases;
    }
    void Randomize(float mag)
    {
        for (int i = 1;i < _Layers.size();i++)
        {
            Layer& L = _Layers[i];
            RandVecIze(&L.Biases, mag / L.Weights[0].size());
            RandVecIze2D(&L.Weights, mag);
        }
    }
};

struct BFInterpSave
{
    bool Valid = false;
    char Cells[200];
    int datap;
};

class BFInterp
{
public:
    char Cells[200];
    int datap;
    std::pair<std::string, double> Run(std::string Code, BFInterpSave Save, std::string Expected = "")
    {
        auto start = std::chrono::high_resolution_clock::now();

        std::string output;
        for (int i = 0; i < 100; i++)
        {
            output = "";
            int stop = 0;
            if (Save.Valid)
            {
                memcpy(Cells, Save.Cells, 200);
                datap = Save.datap;
            }
            else
            {
                datap = 0;
                memset(Cells, 0, 200);
            }
            for (int ip = 0; ip < Code.size(); ip++)
            {
                stop++;
                if (stop > 500)
                {
                    output = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA!24102348919023481902384902385";
                    break;
                }
                if (datap < 0) break;
                if (Code[ip] == '+')
                {
                    Cells[datap]++;
                }
                else if (Code[ip] == '-')
                {
                    Cells[datap]--;
                }
                else if (Code[ip] == '>')
                {
                    datap++;
                }
                else if (Code[ip] == '<')
                {
                    datap--;
                }
                else if (Code[ip] == '.')
                {
                    output.push_back(Cells[datap]);
                }
                else if (Code[ip] == '[')
                {
                    if (Cells[datap] == 0)
                    {
                        while (Code[ip] != ']' && ip < Code.size()) ip++;
                    }
                }
                else if (Code[ip] == ']')
                {
                    if (Cells[datap] > 0)
                    {
                        while (Code[ip] != '[' && ip > 0) ip--;
                    }
                    if (ip == 0) break;
                }
                else if (Code[ip] == ' ') break;
            }
            if (Expected.size() != 0) if (output != Expected) break;
        }
        
        auto end = std::chrono::high_resolution_clock::now();


        return { output, (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() };
    }
    BFInterpSave Save()
    {
        BFInterpSave out;
        memcpy(out.Cells, Cells, 200);
        out.datap = datap;
        out.Valid = true;
        return out;
    }
};

char BFCharLUT[4] = { '+', '-', '<', '>' };

class Optimizer
{
private:
    std::vector<Network> Networks;
    char MLOutToChar_BF(std::vector<float> MLOut)
    {
        uint8_t Bit0 = MLOut[0] > 0.5f ? 1 : 0;
        uint8_t Bit1 = MLOut[1] > 0.5f ? 2 : 0;
        return BFCharLUT[Bit0 | Bit1];
    }
    int MLOutToInt_BF(std::vector<float> MLOut)
    {
        int Result = 0;
        int Mult = 1;
        for (float f : MLOut)
        {
            if (f > 0.5f) Result += Mult;
            Mult *= 2;
        }
        return Result;
    }
    std::vector<float> CodeToMLIn_BF(std::string Code)
    {
        std::vector<float> Out;
        for (char C : Code)
        {
            int idx = 0;
            for (; C != BFCharLUT[idx]; idx++);
            Out.push_back((idx % 2) * 2.0f - 1.0f);
            Out.push_back(((idx >> 1) % 2) * 2.0f - 1.0f);
        }
        return Out;
    }

    int InSize_BF;
    Network NewNet_BF()
    {
        Network Net;
        Net.AddLayer(InSize_BF * 3);
        Net.AddLayer(16);
        Net.AddLayer(8);
        Net.AddLayer(8);
        Net.AddLayer(1);
        return Net;
    }
    std::string GetNetResult_BF(Network& Net, std::string Input)
    {
        std::string Out = Input;
        std::string TrueOut;
        for (int i = 0; i < Input.size(); i++)
        {
            std::string NewOut = Out;
            if (InSize_BF != NewOut.size()) NewOut.erase(NewOut.begin() + InSize_BF, NewOut.end());
            Net.Forward(CodeToMLIn_BF(NewOut));
            int Remove = MLOutToInt_BF(Net.Output());
            if (Remove > 0)
            {
                TrueOut.push_back(Out[0]);
            }
            Out.push_back(Out[0]);
            Out.erase(Out.begin());
        }
        return TrueOut;
    }
    float GetScore_BF(std::string Code)
    {
        float Final = Code.size() * 2.0f;
        int Count = 0;
        for (char C : Code)
        {
            if (C == '[' || C == ']') Final += 10.0f;
            if (C == ']')
            {
                if (Count == 0) return INFINITY;
                Count--;
            }
            if (C == '[') Count++;
        }
        if (Count != 0) return INFINITY;
        return Final;
    }
    bool CellsEqual_BF(char* Cells0, char* Cells1)
    {
        for (int i = 0; i < 200; i++)
        {
            if (Cells0[i] != Cells1[i]) return false;
        }
        return true;
    }
public:
    std::pair<BFInterpSave, std::string> OptimizePartialBF(std::string InBF, int Units, int OptimizationLevel, BFInterpSave Save)
    {
        InSize_BF = min(InBF.size() - 1, 7);
        
        Networks.erase(Networks.begin(), Networks.end());
        for (int i = 0; i < Units; i++)
        {
            Networks.push_back(NewNet_BF());
        }

        BFInterp InterpTop;
        std::pair<std::string, double> BestResult = InterpTop.Run(InBF, Save);
        std::pair<std::string, double> InitResult = BestResult;



        float LearningRate = 1.0f;
        for (int i = 0; i < OptimizationLevel; i++)
        {
            bool Passed = false;

#pragma omp parallel for
            for (int j = 0;j < Networks.size();j++)
            {
                BFInterp Interp;
                Network& Net = Networks[j];
                std::string Out = GetNetResult_BF(Net, InBF);
                std::pair<std::string, double> Result = Interp.Run(Out, Save, InitResult.first);
                if (!CellsEqual_BF(InterpTop.Cells, Interp.Cells) || Result.first != InitResult.first || Interp.datap != InterpTop.datap || GetScore_BF(Out) == INFINITY)
                {
                    Net.fitness = INFINITY;
                }
                else
                {
                    Net.fitness = GetScore_BF(Out);
                }
            }
            std::sort(Networks.begin(), Networks.end(), [](const Network& lhs, const Network& rhs) {
                return lhs.fitness < rhs.fitness;
            });

            int NFailed = 0;
            for (int i = 0; i < Networks.size(); i++)
            {
                if (Networks[i].fitness == INFINITY)
                {
                    NFailed++;
                }
            }
            
            if (NFailed < Networks.size()) LearningRate *= 0.95f;
#pragma omp parallel for
            for (int i = 1;i < Networks.size();i++)
            {
                if (Networks[i].fitness == INFINITY) 
                {
                    Networks[i] = NewNet_BF();
                }
                else
                {
                    Networks[i] = Networks[0];
                    Networks[i].Randomize(LearningRate);
                }
            }

            
            if (NFailed == Networks.size()) OptimizationLevel++;

        }
        std::string Final = GetNetResult_BF(Networks[0], InBF);
        return { InterpTop.Save(), Final };
    }
    std::string OptimizeBF(std::string Code)
    {
        BFInterpSave CurSave = { false };
        std::string Out;
        int i = 0;
        while (i < Code.size())
        {
            std::string Sample;
            while (Code[i] != '.' && i < Code.size())
            {
                Sample.push_back(Code[i]);
                i++;
            }
            if (Sample.size() > 0)
            {
                Sample.push_back(Code[i]);
                std::pair<BFInterpSave, std::string> OptimalSample = OptimizePartialBF(Sample, 500, 20, CurSave); 
                CurSave = OptimalSample.first;
                Out += OptimalSample.second;
            }
            else
            {
                Out += Code[i];
            }
            i++;
            std::cout << "Completed sub-sample `" + Sample + "`\n";
        }
        std::cout << "Saved " << Code.size() - Out.size() << " characters!\n";
        return Out;
    }
};

int main()
{
    Optimizer Optim;
    std::string InCode = "+++++++++[>+++++++++-+<-]>.";
    std::cout << "Input: " << InCode << "\n";
    std::cout << "AI Optimized: " << Optim.OptimizeBF(InCode) << "\n";
}