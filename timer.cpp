#include <stdio.h>
#include <sys/time.h>

#define MICRO 1000000

struct timeval startTime;
struct timeval endTime;

/*
 * Returns true on success;
 * Returns false on error;
 */
bool startTimer() {
  if(gettimeofday(&startTime, NULL)!=0) {
    return false;
  }
  return true;
}

/*
 * Returns true on success;
 * Returns false on error;
 */
bool endTimer() {
  if(gettimeofday(&endTime, NULL)!=0) {
    return false;
  }
  return true;
}

/*
 * Returns elapsed time
 */
unsigned long long int getElapsedTime() {
  unsigned long long int startTimeMicroSeconds = startTime.tv_sec*MICRO + startTime.tv_usec;
  unsigned long long int endTimeMicroSeconds = endTime.tv_sec*MICRO + endTime.tv_usec;
  return endTimeMicroSeconds - startTimeMicroSeconds;
}
