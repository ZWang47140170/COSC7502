#pragma once
class CTime
{
public:
	CTime();
	~CTime();
	void Tic();
	void Toc(const char* txt);
	void Toc();
	float GetElaps();

private:
    float m_elaps_ms;
#ifdef _WIN32
    LARGE_INTEGER m_nFreq;
    LARGE_INTEGER m_nBeginTime;
    LARGE_INTEGER m_nEndTime;
#else
	struct timeval m_nBeginTime;
	struct timeval m_nEndTime;
#endif
};

