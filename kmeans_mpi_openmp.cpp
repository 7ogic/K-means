
#include "kmeans.h"
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <omp.h>

void kmeans(int iteration_n, int class_n, int data_n, Point* centroids, Point* data, int* partitioned)
{
	// Loop indices for iteration, data and class
	int i, data_i, class_i;
	// Count number of data in each class
	int* count = (int*)malloc(sizeof(int) * class_n);	//각 분류 클래스의 데이터 수 저장할 변수

	// Temporal point value to calculate distance
	Point t;	// 거리 계산을 하기 위한 포인트 변수

	struct Point *sum_centroids;
	int *sum_count;

	// Iterate through number of interations
	for (i = 0; i < iteration_n; i++) {
/************************************ OpenMP ***************************************/
#pragma omp parallel num_threads(1)
		{
			int nthreads = omp_get_num_threads();
			int ithread = omp_get_thread_num();

			/* Assignment step */
			#pragma omp for private(t, class_i)
			for (data_i = 0; data_i < data_n; data_i++) {
				float min_dist = DBL_MAX;					// 최소 거리값 최대값으로

				/* 분류될 곳 찾기 */
				for (class_i = 0; class_i < class_n; class_i++) {	// 분류 그룹 수 iter
					t.x = data[data_i].x - centroids[class_i].x;		// 점.x - 분류된 곳 점.x
					t.y = data[data_i].y - centroids[class_i].y;		// 점.y - 분류된 곳 점.y

					float dist = t.x * t.x + t.y * t.y;

					if (dist < min_dist) {
						partitioned[data_i] = class_i;
						min_dist = dist;
					}
				}
			}/* end of Assignment step */

			/* Update step */
			// Clear sum buffer and class count
#pragma omp for
			for (class_i = 0; class_i < class_n; class_i++) {
				centroids[class_i].x = 0.0;
				centroids[class_i].y = 0.0;
				count[class_i] = 0;
			}
#pragma omp single
			{
				sum_centroids = (struct Point*)malloc(sizeof(struct Point)*nthreads*class_n);
				sum_count = (int*)malloc(sizeof(int)*nthreads*class_n);
			}
#pragma omp for
			for(class_i = 0; class_i < class_n; class_i++) {
				sum_centroids[ithread*class_n+class_i].x = 0.0;
				sum_centroids[ithread*class_n+class_i].y = 0.0;
				sum_count[ithread*class_n+class_i] = 0;
			}

			// Sum up and count data for each class
#pragma omp for
			for (data_i = 0; data_i < data_n; data_i++) {         
				sum_centroids[ithread*class_n+partitioned[data_i]].x += data[data_i].x;
				sum_centroids[ithread*class_n+partitioned[data_i]].y += data[data_i].y;
				sum_count[ithread*class_n+partitioned[data_i]]++;		//각 분류된 그룹 점 개수 카운트++
			}
#pragma omp for private(t)
			for (class_i = 0; class_i < class_n; class_i++) {
				for (int t = 0; t < nthreads; t++) {
					centroids[class_i].x += sum_centroids[t*class_n+class_i].x;
					centroids[class_i].y += sum_centroids[t*class_n+class_i].y;
					count[class_i] += sum_count[t*class_n+class_i];
				}
			}
#pragma omp for
			for (class_i = 0; class_i < class_n; class_i++) {
				centroids[class_i].x /= count[class_i];				//평균 구하기
				centroids[class_i].y /= count[class_i];
			}
		}
	}
	free(sum_centroids);
	free(sum_count);
}

