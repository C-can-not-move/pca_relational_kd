// src/pca_relational_kd.cpp —— 数据管线冒烟测试
#include "pca_relational_kd.h"

int main(int argc, char** argv)
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    try
    {
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}