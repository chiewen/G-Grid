#pragma once
#include <time.h>

struct Now {
	static int now() {
		time_t rawtime;
		time(&rawtime);
		struct tm * ptm = gmtime(&rawtime);
		return ptm->tm_sec;
	}
};
