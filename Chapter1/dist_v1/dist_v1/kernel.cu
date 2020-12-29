#include <math.h>
#define N 64

float scale(int i, int n)
{
	return ((float)i) / (n - 1);
}

float distance(float x1, float x2)
{
	return fabsf(x2 - x1);
}

int main()	
{
	float out[N] = { 0.0f };

	float ref = .5f;

	for (int i = 0; i < N; ++i)
	{
		float x = scale(i, N);
		out[i] = distance(x, ref);
	}

	return 0;
}