#ifdef _WIN32
#include <windows.h>
int gettimeofday(struct timeval *tp, void *tzp) {
    static const unsigned __int64 epoch = 116444736000000000ULL;
    FILETIME file_time;
    SYSTEMTIME system_time;
    unsigned __int64 time;

    GetSystemTime(&system_time);
    SystemTimeToFileTime(&system_time, &file_time);
    time = ((unsigned __int64)file_time.dwLowDateTime);
    time += ((unsigned __int64)file_time.dwHighDateTime) << 32;

    tp->tv_sec = (long)((time - epoch) / 10000000L);
    tp->tv_usec = (long)(system_time.wMilliseconds * 1000);
    return 0;
}
#else
#include <sys/time.h>
#include <time.h>
#endif

double get_sys_time_seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)((long long)tv.tv_sec * 1000000LL + (long long)tv.tv_usec) / 1000000.0; // Convert microseconds to seconds
}

void sleep_ms(int milliseconds) {
#ifdef _WIN32
    Sleep(milliseconds);
#else
    struct timespec ts;
    ts.tv_sec = milliseconds / 1000;
    ts.tv_nsec = (milliseconds % 1000) * 1000000;
    nanosleep(&ts, NULL);
#endif
}