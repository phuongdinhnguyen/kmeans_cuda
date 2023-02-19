#include "kmeans_sequential.h"
#include <time.h>
#include <chrono>
#include <ctime>
#include <thread>

int main(int argc, char** argv)
{
    KMeans kmeans(10, 3072);
    kmeans.readDataFromFile();
    kmeans.initCentroid();

    auto startTime = std::chrono::steady_clock::now();  // start count running time

    kmeans.process();

    auto endTime = std::chrono::steady_clock::now();
    auto encTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    cout << "Total run time: " << encTime / 1000.0 << " sec." << endl;

    kmeans.checkAccuracy();
    kmeans.DaviesBouldinIndex();
    cout << "program finished!";
    return 0;
}