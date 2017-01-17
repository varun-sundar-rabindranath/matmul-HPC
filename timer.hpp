#ifndef __TIMER_HPP__
#define __TIMER_HPP__

#define ULL unsigned long long int

typedef enum {
  APP_TIMER    = 0,
  MATMUL_TIMER = 1,
  TIMER_ID_END = 2,
}timer_id;

bool startTimer(int id);

bool endTimer(int id);

ULL getElapsedTime(int id);

#endif // __TIMER_HPP__
