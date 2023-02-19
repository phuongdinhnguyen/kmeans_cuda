#pragma once
#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <string>
#include <cstring>
#include <stdlib.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>

using namespace std;
using namespace thrust;

string CIFAR10_file[5] = {"./cifar-10/data_batch_1.bin","./cifar-10/data_batch_2.bin","./cifar-10/data_batch_3.bin",
                          "./cifar-10/data_batch_4.bin","./cifar-10/data_batch_5.bin"};

#define THRESHOLD 0.001

class KMeans
{
public:
    int numClusters;            //number of centroids
    int dim;                    //dim of a data point
    int numPoints;

    ofstream fileO;

    float* m_dataPoints;
    float* m_centrePoints;
    float* m_clusterData;
    int* m_clusterSizes;
    int* m_clusterIds;
    int* m_labels;
    int* m_ord;

    // _k as kmeans, _dim in dimension
    KMeans(int _k, int _dim);
    void readCIFAR10();
    void initCentroid();
    void process();
    void printData();
    void printCentre();
    void printResult();
    void checkAccuracy();
    void evaluateResult();

private:
    float threshold = 0.01;
    void reset();
};

KMeans::KMeans(int _k, int _dim)
{
    numClusters = _k;
    dim = _dim;
    numPoints = 0;
    cout << "Number of clusters: " << numClusters << endl;
    cout << "Data dimension: " << dim << endl;

    cudaMallocManaged(&m_centrePoints, sizeof(float)*dim*numClusters);
    cudaMallocManaged(&m_clusterSizes, sizeof(int)*numClusters);

    fileO.open("output.txt");
    fileO << numClusters << endl;
    fileO << dim << endl;
}

void KMeans::readCIFAR10()
{
    int numData = 10000;    // number of data per CIFAR10 bin files
    int numFiles = 5;       // number of CIFAR10 files
    numPoints = numData * numFiles;
    cudaMallocManaged(&m_dataPoints, sizeof(float)*numPoints*dim);
    cudaMallocManaged(&m_labels    , sizeof(int)*numPoints);

    ifstream file3;
    for (int file_idx = 0; file_idx < numFiles; file_idx++)
    {
        file3.open(CIFAR10_file[file_idx], ios::binary);
        if (file3.is_open())
        {
            cout << "file opened! \n";
            unsigned char label = 0;
            for (int cnt = file_idx * numData; cnt < (file_idx + 1) * numData; cnt++)
            {
                file3.read((char*)&label, sizeof(label));
                m_labels[cnt] = (int)label;
                for (int i = 0; i < dim; ++i)
                {
                    unsigned char pixel = 0;
                    file3.read((char*)&pixel, sizeof(pixel));
                    m_dataPoints[cnt*dim+i] = (int)pixel;
                }
            }
        }
        file3.close();
    }
    cout << "finished read file!\n";
}

void KMeans::initCentroid()
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    static std::default_random_engine generator(123);
    int end = numPoints - 1;
    std::uniform_int_distribution<int> distribution(0, end);

    vector<int> p(numPoints);
    int cnt = 0;
    cout << "generate centroids from 0 to " << end << endl;
    cout << "List of centroids:\n";
    while (cnt < numClusters)
    {
        int number = distribution(generator);
        if (!p[number])
        {
            cout << number << " ";
            p[number] = 1;
            std::copy(m_dataPoints + number*dim, m_dataPoints + (number+1)*dim, m_centrePoints + cnt*dim);
            
            cnt++;
        }

    }
    cout << endl;
}

__host__ __device__ 
float distance2(const float* point1,const float* point2, int dim)
{
    float totalDistance = 0;
    for (int i = 0; i < dim; i++)
    {
        totalDistance += (point1[i] - point2[i]) * (point1[i] - point2[i]);
    }

    return totalDistance;
}

__global__
void Kmeans_find_nearest_cluster(int numPoints, 
                                 int numClusters, 
                                 float* __restrict__ dataPoints, 
                                 float* __restrict__ centrePoints, 
                                 float* __restrict__ distanceToCentre,
                                 int dim)
{
    int pointIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int centreIdx = blockIdx.y * blockDim.y + threadIdx.y;
    if (pointIdx >= numPoints || centreIdx >= numClusters) return;
    float sum = 0;
    float tmp = 0;

    for (int i = 0 ; i < dim ; i++)
    {
        tmp = dataPoints[pointIdx * dim + i] - centrePoints[centreIdx * dim + i];
        sum += tmp*tmp;
    }
    distanceToCentre[centreIdx * numPoints + pointIdx] = sqrt(sum);
}

__global__
void assignCluster(int numPoints, 
                   int numClusters, 
                   float* __restrict__ dataPoints, 
                   int* __restrict__ clusterId, 
                   int* __restrict__ clusterSizes, 
                   float* __restrict__ distanceToCentre, 
                   float* __restrict__ newCentre, 
                   int dim)
{
    int pointIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pointIdx >= numPoints) return;
    float dmin = INT_MAX;
    int cen = 0;
    for (int centreIdx = 0 ; centreIdx < numClusters ; centreIdx++)
    {
        float tmp = distanceToCentre[centreIdx * numPoints + pointIdx];
        if (tmp < dmin)
        {
            dmin = tmp;
            cen = centreIdx;
        }
    }
    clusterId[pointIdx] = cen;
    atomicAdd(&clusterSizes[cen],1);
    for (int i = 0 ; i < dim ; i++)
    {
        atomicAdd(&newCentre[cen*dim + i], dataPoints[pointIdx*dim + i]);
    }
}

__global__
void recalculate_centre(int numClusters, 
                        int* __restrict__ clusterSizes, 
                        float* __restrict__ m_centrePoints, 
                        float* __restrict__ newCentre, 
                        int dim)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index ; i < dim ; i+=stride)
    {
        for (int j = 0 ; j < numClusters ; j++)
        {
            m_centrePoints[j*dim + i] = newCentre[j*dim+i] / clusterSizes[j];
        }
    }
}

void KMeans::process()
{
    cout << "Total number of points: " << numPoints << endl;
    cudaMallocManaged(&m_clusterIds, sizeof(int)*numPoints);
    
    memset(m_clusterIds, 0 , sizeof(int)*numPoints);
    
    int blockSize = 1024;
    int numBlocks = (numPoints + blockSize - 1) / blockSize;

    cout << "Block size: " << blockSize << endl;
    cout << "Number of blocks: " << numBlocks << endl;

    float* newCentre;
    cudaMallocManaged(&newCentre, sizeof(float)*numClusters*dim);

    int* oldClusterSize;
    cudaMallocManaged(&oldClusterSize, sizeof(int)*numClusters);
    memset(newCentre,0 ,sizeof(float)*numClusters*dim);

    int diff;

    float* distanceToCentre;
    cudaMallocManaged(&distanceToCentre, sizeof(float)*numClusters*numPoints);

    // define dim of a thread block
    int blkSize_x = 96;
    int blkSize_y = numClusters;
    dim3 blkSize(blkSize_x, blkSize_y, 1);
    int gridSize = numPoints / blkSize_x + 1;
    while (true)
    {
        diff = 0;
        thrust::copy(thrust::device, m_clusterSizes, m_clusterSizes + sizeof(int)*numClusters, oldClusterSize);
        thrust::fill(thrust::device, m_clusterSizes, m_clusterSizes + sizeof(int)*numClusters, 0);

        // cout << "reset newcentre:" << endl;
        memset(newCentre,0 ,sizeof(float)*numClusters*dim);
        
        // cout << "find clusters...\n";
        Kmeans_find_nearest_cluster<<<gridSize, blkSize>>>(numPoints, 
                                                           numClusters, 
                                                           m_dataPoints,
                                                           m_centrePoints, 
                                                           distanceToCentre,
                                                           dim);                                                          


        // cout << "assign clusters...\n";
        assignCluster<<<numBlocks,blockSize>>>(numPoints, 
                                              numClusters, 
                                              m_dataPoints, 
                                              m_clusterIds, 
                                              m_clusterSizes, 
                                              distanceToCentre, 
                                              newCentre,
                                              dim);
      
        // cout << "recal centre...\n";
        recalculate_centre<<<numBlocks, blockSize>>>(numClusters,
                                                     m_clusterSizes, 
                                                     m_centrePoints, 
                                                     newCentre, 
                                                     dim);
        cudaDeviceSynchronize();

        for (int i = 0 ; i < numClusters ; i++)
        {
            diff += abs(m_clusterSizes[i] - oldClusterSize[i]);
        }

        if (diff < 2) break;
    }
}

void KMeans::checkAccuracy()
{
    int cnt = 0;
    vector <int> clusterLabel(10);
    for (int i = 0 ; i < numClusters ; i++)
    {
        vector <int> clusterCount(10, 0);
        int max = INT_MIN;
        int label = -1;
        for (int j = 0 ; j < numPoints ; j++)
        {
            if (m_clusterIds[j] == i)
            {
                clusterCount[m_labels[j]]++;
                if (clusterCount[m_labels[j]] > max)
                {
                    max = clusterCount[m_labels[j]];
                    label = m_labels[j];
                }
            }
        }
        
        clusterLabel[i] = label;
    }

    cout << "count true labels..." << endl;
    cnt = 0;
    for (int i = 0 ; i < numClusters ; i++)
    {
        for (int j = 0 ; j < numPoints ; j++)
        if (m_clusterIds[j] == i && m_labels[j] == clusterLabel[i]) cnt++;      
    }

    for (int i = 0 ; i < numClusters ; i++)
    {
        cout << "Cluster " << i << " true label is: " << clusterLabel[i] << endl;
    }
    

    cout << "Accuracy: " << (float) cnt / numPoints * 100.0 << endl;
    cout << cnt << " over " << numPoints << endl;

    //printPoint(dataPoints[1]);
    cudaFree(m_dataPoints);
    cudaFree(m_centrePoints);
    cudaFree(m_labels);
    cudaFree(m_clusterIds);
    cudaFree(m_clusterSizes);
}

void KMeans::evaluateResult()
{
    vector <float> IntraClusterDispersion(numClusters);
    for (int i = 0; i < numClusters; i++)
    {
        int clusterIdx = i;
        for (int j = 0; j < numPoints; j++)
            if (m_clusterIds[j] == clusterIdx)
            {
                IntraClusterDispersion[clusterIdx] += distance2(m_dataPoints+j*dim, m_centrePoints+i*dim, dim);
            }
    }

    float ClusterSeperationMeasure[numClusters][numClusters];
    memset(ClusterSeperationMeasure, 0, sizeof(ClusterSeperationMeasure));

    for (int i = 0; i < numClusters; i++)
        for (int j = i; j < numClusters; j++)
            ClusterSeperationMeasure[i][j] = distance2(m_centrePoints+i*dim, m_centrePoints+j*dim, dim);

    float SimilarityBetweenClusters[numClusters][numClusters];
    for (int i = 0; i < numClusters; i++)
        for (int j = i; j < numClusters; j++)
        {
            SimilarityBetweenClusters[i][j] = (IntraClusterDispersion[i] + IntraClusterDispersion[j]) / ClusterSeperationMeasure[i][j];
        }

    vector <float> MostSimilarCluster(numClusters);
    for (int i = 0; i < numClusters - 1; i++)
    {
        float max = INT_MIN;
        for (int j = i + 1; j < numClusters; j++)
        {
            if (max < SimilarityBetweenClusters[i][j])
                max = SimilarityBetweenClusters[i][j];
        }

        MostSimilarCluster[i] = max;
    }

    float result = 0;
    for (int i = 0; i < numClusters; i++)
    {
        result += MostSimilarCluster[i];
    }

    result = result / numClusters;
    cout << "Davies - Bouldin Index: " << result << endl;
}