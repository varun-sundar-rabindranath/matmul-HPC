#include "timer.hpp"

#include <stdio.h>
#include <sys/time.h>

#include <iostream>
#include <vector>
using namespace std;

#define MICRO 1000000

vector<struct timeval> startTime(50);
vector<struct timeval> endTime(50);



/*
 * Returns true on success;
 * Returns false on error;
 */
bool startTimer(int id) {
  if(gettimeofday(&startTime[id], NULL)!=0) {
    return false;
  }
  return true;
}

/*
 * Returns true on success;
 * Returns false on error;
 */
bool endTimer(int id) {
  if(gettimeofday(&endTime[id], NULL)!=0) {
    return false;
  }
  return true;
}

/*
 * Returns elapsed time
 */
ULL getElapsedTime(int id) {
  ULL startTimeMicroSeconds = startTime[id].tv_sec*MICRO + startTime[id].tv_usec;
  ULL endTimeMicroSeconds = endTime[id].tv_sec*MICRO + endTime[id].tv_usec;
  return endTimeMicroSeconds - startTimeMicroSeconds;
}
