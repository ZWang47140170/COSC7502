#include <stdio.h>
#ifdef _WIN32
#include <Windows.h>
#else
#include <sys/time.h>
#endif

#include "CTime.h"

CTime::CTime()
{
#ifdef _WIN32
	QueryPerformanceFrequency(&m_nFreq);
#endif
}

CTime::~CTime()
{

}

/**
 * A function to record begin time.
 * 
 */
void CTime::Tic()
{
#ifdef _WIN32
	QueryPerformanceCounter(&m_nBeginTime);
#else
	gettimeofday(&m_nBeginTime, NULL);
#endif
}

/**
 * A function to record end time and compute corresponding duration.
 * 
 * \param txt The text to print.
 */
void CTime::Toc(const char* txt)
{
#ifdef _WIN32
	QueryPerformanceCounter(&m_nEndTime);
	m_elaps_ms = (float)(m_nEndTime.QuadPart - m_nBeginTime.QuadPart) / (float)m_nFreq.QuadPart * 1000.0f;
#else
	gettimeofday(&m_nEndTime, NULL);
	m_elaps_ms = (float)((m_nEndTime.tv_sec - m_nBeginTime.tv_sec) * 1000000 + (m_nEndTime.tv_usec - m_nBeginTime.tv_usec)) / 1000.0f;
#endif
	printf("Run time of %s is: %.3f ms\n", txt, m_elaps_ms);
}

/**
 * A function to record end time and compute corresponding duration.
 * 
 */
void CTime::Toc()
{
#ifdef _WIN32
	QueryPerformanceCounter(&m_nEndTime);
	m_elaps_ms = (float)(m_nEndTime.QuadPart - m_nBeginTime.QuadPart) / (float)m_nFreq.QuadPart * 1000.0f;
#else
	gettimeofday(&m_nEndTime, NULL);
	m_elaps_ms = (float)((m_nEndTime.tv_sec - m_nBeginTime.tv_sec) * 1000000 + (m_nEndTime.tv_usec - m_nBeginTime.tv_usec)) / 1000.0f;
#endif
}

/**
 * A function to get duration.
 * 
 * \return duration.
 */
float CTime::GetElaps()
{
	return m_elaps_ms;
}
