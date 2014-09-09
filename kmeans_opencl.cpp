#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kmeans.h"

#include <CL/cl.h>
#include "cl_util.h"

// 앞으로 할 것 : 빌드 시간 줄이기, MPI, assignment, update_count 시간 체크
// 
void kmeans(int iteration_n, int class_n, int data_n, Point* centroids, Point* data, int* partitioned)
{
	//OpenCL
	cl_platform_id			*platform;
	cl_device_type			dev_type = CL_DEVICE_TYPE_GPU;
	cl_device_id				*devs = NULL;
	cl_context					context;
	cl_command_queue		*cmd_queues;
	cl_program					program;
	cl_kernel						*kernels;
	cl_mem							*mem_centroids;
	cl_mem							*mem_data;
	cl_mem							*mem_partitioned;
	cl_uint							num_platforms;
	cl_uint							num_devs = 0;
	cl_int							err;

	//#2
	cl_kernel						*kernels_update_count;
	cl_mem						*mem_g_centroids;
	cl_mem						*mem_g_count;

	unsigned int i;

	// Platform
	err = clGetPlatformIDs(0, NULL, &num_platforms);
	CHECK_ERROR(err);
	if (num_platforms == 0	) {
		fprintf(stderr, "[%s:%d]  ERROR: No OpenCL platform\n", __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	printf("Number of platforms: %u\n", num_platforms);
	platform = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
	err = clGetPlatformIDs(num_platforms, platform, NULL);
	CHECK_ERROR(err);

	// Device
	for (i = 0; i < num_platforms; ++i ) {
		err = clGetDeviceIDs(platform[i], dev_type, 0, NULL, &num_devs);
		if (err != CL_DEVICE_NOT_FOUND) CHECK_ERROR(err);
		if (num_devs >= 1) {
			devs = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devs);

			err = clGetDeviceIDs(platform[i], dev_type, num_devs, devs, NULL);
			break;
		}
	}

	if (devs == NULL || num_devs < 1) 	{
		fprintf(stderr, "[%s:%d]  ERROR: No device\n", __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	// print current devices
	for( i = 0; i < num_devs; ++i ) {
		printf("dev[%d] : ", i);
		print_device_name(devs[i]);
	}

	// Context
	context = clCreateContext(NULL, num_devs, devs, NULL, NULL, &err);
	CHECK_ERROR(err);
	// Command queue
	cmd_queues = (cl_command_queue*)malloc(sizeof(cl_command_queue)*num_devs);
	for (i = 0; i < num_devs; ++i) {
		cmd_queues[i] = clCreateCommandQueue(context, devs[i], 0, &err);
		CHECK_ERROR(err);
	}

	// Create a program.
	char *source_code = get_source_code("./kernel_kmeans.cl");
	program = clCreateProgramWithSource(context,
																			1,
																			(const char **)&source_code,
																			NULL,
																			&err);
	free(source_code);
	CHECK_ERROR(err);

	// Build the program.
	char build_opts[200];
	sprintf(build_opts, "-DCLASS_N=%d -DDATA_N=%d", class_n, data_n / num_devs);
	err = clBuildProgram(program, num_devs, devs, build_opts, NULL, NULL);
	if (err != CL_SUCCESS) {
		print_build_log(program, devs[0]);
		CHECK_ERROR(err);
	}

	// Kernel
	kernels = (cl_kernel*)malloc(sizeof(cl_kernel)*num_devs);
	for (i = 0; i < num_devs; ++i) {
		kernels[i] = clCreateKernel(program, "assignment", NULL);
	}

	//#3 kernel for update_count
	kernels_update_count = (cl_kernel*)malloc(sizeof(cl_kernel)*num_devs);
	for (i = 0; i < num_devs; ++i) {
		kernels_update_count[i] = clCreateKernel(program, "update_count", NULL);
	}

 	//Buffers
	mem_centroids = (cl_mem*)malloc(sizeof(cl_mem)*num_devs);
	mem_data = (cl_mem*)malloc(sizeof(cl_mem)*num_devs);
	mem_partitioned = (cl_mem*)malloc(sizeof(cl_mem)*num_devs);

	for (i = 0; i < num_devs; ++i) {
		mem_centroids[i] = clCreateBuffer(context,
														CL_MEM_READ_WRITE,
														sizeof(struct Point)*class_n,
														NULL,
														NULL);
		mem_data[i] = clCreateBuffer(context,
														CL_MEM_READ_ONLY,
														sizeof(struct Point)*data_n/num_devs,
														NULL,
														NULL);
		mem_partitioned[i] = clCreateBuffer(context,
														CL_MEM_READ_WRITE,
														sizeof(int)*data_n/num_devs,
														NULL,
														NULL);
	}

	//Write to Buffers 
	for (i = 0; i < num_devs; ++i) {
/*
		clEnqueueWriteBuffer(cmd_queues[i], mem_centroids[i],
										CL_FALSE, 0,
										sizeof(struct Point)*class_n,
										centroids,
										0, NULL, NULL);
*/
		clEnqueueWriteBuffer(cmd_queues[i], mem_data[i],
										CL_FALSE, 0,
										sizeof(struct Point)*data_n/num_devs,
										&data[i*(data_n/num_devs)],
										0, NULL, NULL);
		clEnqueueWriteBuffer(cmd_queues[i], mem_partitioned[i],
										CL_FALSE, 0,
										sizeof(int)*data_n/num_devs,
										&partitioned[i*(data_n/num_devs)],
										0, NULL, NULL);
	}

	//Set the arguments.
	for (i = 0; i < num_devs; ++i) {
		clSetKernelArg(kernels[i], 0, sizeof(cl_mem), (void*) &mem_centroids[i]);
		clSetKernelArg(kernels[i], 1, sizeof(cl_mem), (void*) &mem_data[i]);
		clSetKernelArg(kernels[i], 2, sizeof(cl_mem), (void*) &mem_partitioned[i]);
	}
	
	// Set the workspace.
	size_t lws[2] = { 256, 1 };
	size_t gws[2] = { data_n / num_devs, 1 };

	//#6 set the workspace for kernels update_x, update_y, count
	size_t kuc_lws[2] = { 256, 1 };
	size_t kuc_gws[2] = { (size_t)ceil((double)(data_n/num_devs)/lws[0])*lws[0], 1 };
	size_t num_wg = kuc_gws[0] / kuc_lws[0];
	//@@
	


	//#7 buffers for kernels update_count
	mem_g_centroids = (cl_mem*)malloc(sizeof(cl_mem)*num_devs);
	mem_g_count = (cl_mem*)malloc(sizeof(cl_mem)*num_devs);
	for (i = 0; i < num_devs; ++i) {
		mem_g_centroids[i] = clCreateBuffer(context,
														CL_MEM_READ_WRITE,
														sizeof(struct Point) * class_n * num_wg,
														NULL, &err);
		CHECK_ERROR(err);

		mem_g_count[i] = clCreateBuffer(context,
													CL_MEM_READ_WRITE,
													sizeof(int) * num_wg * class_n,
													NULL, &err);
		CHECK_ERROR(err);
	}

	//#8 set the arguments for kernels_update_count
	for (i = 0; i < num_devs; ++i) {
		//for kernels_update_count
		clSetKernelArg(kernels_update_count[i], 0, sizeof(cl_mem), (void*) &mem_data[i]);
		clSetKernelArg(kernels_update_count[i], 1, sizeof(cl_mem), (void*) &mem_partitioned[i]);
		clSetKernelArg(kernels_update_count[i], 2, sizeof(cl_mem), (void*) &mem_g_centroids[i]);
		clSetKernelArg(kernels_update_count[i], 3, sizeof(cl_mem), (void*) &mem_g_count[i]);
		//local mem
		clSetKernelArg(kernels_update_count[i], 4, sizeof(struct Point) * kuc_lws[0], NULL);
		clSetKernelArg(kernels_update_count[i], 5, sizeof(int) * kuc_lws[0], NULL);
	}

	//#9 
	for (i = 0; i < num_devs; ++i) {
		clFinish(cmd_queues[i]);
	}

	// Count number of data in each class
	int* count = (int*)malloc(sizeof(int) * class_n);	//각 분류 클래스의 데이터 수 저장할 변수
	int iter_i;
	int class_i;
	unsigned int num_wg_i;
	//#11 g_count, g_centroids
	struct Point *g_centroids = (struct Point*)malloc(sizeof(struct Point)*num_wg*num_devs*class_n);
	int *g_count = (int*)malloc(sizeof(int)*num_wg*num_devs*class_n);

	// Iterate through number of interations
	for (iter_i = 0; iter_i < iteration_n; ++iter_i) {

		//매회 업데이트되는 centroids 다시write
		for (i = 0; i < num_devs; ++i) {
			clEnqueueWriteBuffer(cmd_queues[i], mem_centroids[i],
											CL_FALSE, 0,
											sizeof(struct Point)*class_n,
											centroids,
											0, NULL, NULL);
		}

		// Enqueue the kernel( Assignment step )
		for (i = 0; i < num_devs; ++i) {
			err = clEnqueueNDRangeKernel(cmd_queues[i],
														kernels[i],
														2, NULL,
														gws, lws,
														0, NULL, NULL);
			CHECK_ERROR(err);
		} 


		//#10 Update step
		// Clear sum buffer and class count
		// Enqueue the kernel_update_count
		for (i = 0; i < num_devs; ++i) {
			err = clEnqueueNDRangeKernel(cmd_queues[i],
														kernels_update_count[i],
														2, NULL,
														kuc_gws, kuc_lws,
														0, NULL, NULL);
			CHECK_ERROR(err);
		}

		//#12 read the result of kernels(update_count)
		for (i = 0; i < num_devs; ++i) {
			clEnqueueReadBuffer(cmd_queues[i],
											mem_g_centroids[i],
											CL_FALSE, 0,
											sizeof(struct Point)*num_wg*class_n,
											&g_centroids[i*num_wg*class_n],			///////////////QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ
											0, NULL, NULL);
			CHECK_ERROR(err);

			clEnqueueReadBuffer(cmd_queues[i],
											mem_g_count[i],
											CL_FALSE, 0,
											sizeof(int)*num_wg*class_n,
											&g_count[i*num_wg*class_n],
											0, NULL, NULL);
			CHECK_ERROR(err);
		}

		for (i = 0; i < num_devs; ++i) {
			clFinish(cmd_queues[i]);
		}

		// Clear sum buffer and class count
		for (class_i = 0; class_i < class_n; class_i++) {
			centroids[class_i].x = 0.0;
			centroids[class_i].y = 0.0;
			count[class_i] = 0;
	//	printf("%f %f\n", centroids[class_i].x, centroids[class_i].y);
		}

		//#13 sum up count and centroids for each class
		for (i = 0; i < num_devs; ++i) {
			for (num_wg_i = 0; num_wg_i < num_wg; ++num_wg_i) {
				for (class_i = 0; class_i < class_n; ++class_i ) {
					centroids[class_i].x += g_centroids[i*class_n*num_wg+class_i*num_wg+num_wg_i].x;
					centroids[class_i].y += g_centroids[i*class_n*num_wg+class_i*num_wg+num_wg_i].y;
					count[class_i] += g_count[i*class_n*num_wg+class_i*num_wg+num_wg_i];
				}
			}
		}
		

		// Divide the sum with number of class for mean point
		for (class_i = 0; class_i < class_n; class_i++) {
			centroids[class_i].x /= count[class_i];				//평균 구하기
			centroids[class_i].y /= count[class_i];
//			printf("%f %f\n", centroids[class_i].x, centroids[class_i].y);
		}
	}

	// Read the result of Assignment.
	for (i = 0; i < num_devs; ++i) {
			clEnqueueReadBuffer(cmd_queues[i],
					mem_partitioned[i],
					CL_TRUE, 0,
					sizeof(int)*data_n/num_devs,
					&partitioned[i*(data_n/num_devs)],
					0, NULL, NULL);
	}


	//ev_kernels = .. .. . . .

	// Release
	for (i = 0; i < num_devs; ++i) {
		clReleaseMemObject(mem_centroids[i]);
		clReleaseMemObject(mem_data[i]);
		clReleaseMemObject(mem_partitioned[i]);
		clReleaseKernel(kernels[i]);
		clReleaseCommandQueue(cmd_queues[i]);
		//		clReleaseEvent(ev_kernels[i]);

		//#4 
		clReleaseKernel(kernels_update_count[i]);
		clReleaseMemObject(mem_g_centroids[i]);
		clReleaseMemObject(mem_g_count[i]);
	}
	clReleaseProgram(program);
	clReleaseContext(context);
	free(platform);
	free(mem_centroids);
	free(mem_data);
	free(mem_partitioned);
	//
	free(cmd_queues);
	free(kernels);
	free(devs);
	//	free(ev_kernels);

	//#5
	free(kernels_update_count);
	free(mem_g_centroids);
	free(mem_g_count);
	free(g_centroids);
	free(g_count);

/*	
	// Loop indices for iteration, data and class
	int data_i, class_i;
	// Count number of data in each class
	int* count = (int*)malloc(sizeof(int) * class_n);	//각 분류 클래스의 데이터 수 저장할 변수

	// Temporal point value to calculate distance
	Point t;	// 거리 계산을 하기 위한 포인트 변수


	// Iterate through number of interations
	for (i = 0; i < iteration_n; i++) {

		// Assignment step
		for (data_i = 0; data_i < data_n; data_i++) {
			float min_dist = DBL_MAX;					// 최소 거리값 최대값으로

			// 분류될 곳 찾기
			for (class_i = 0; class_i < class_n; class_i++) {	// 분류 그룹 수 iter
				t.x = data[data_i].x - centroids[class_i].x;		// 점.x - 분류된 곳 점.x
				t.y = data[data_i].y - centroids[class_i].y;		// 점.y - 분류된 곳 점.y

				float dist = t.x * t.x + t.y * t.y;

				if (dist < min_dist) {
					partitioned[data_i] = class_i;
					min_dist = dist;
				}
			}
		}

		// Update step
		// Clear sum buffer and class count
		for (class_i = 0; class_i < class_n; class_i++) {
			centroids[class_i].x = 0.0;
			centroids[class_i].y = 0.0;
			count[class_i] = 0;
		}

		// Sum up and count data for each class
		for (data_i = 0; data_i < data_n; data_i++) {         
			centroids[partitioned[data_i]].x += data[data_i].x;								//분리를 x하나
			centroids[partitioned[data_i]].y += data[data_i].y;								// y하나
			count[partitioned[data_i]]++;		//각 분류된 그룹 점 개수 카운트++		// count 하나
		}

		// Divide the sum with number of class for mean point
		for (class_i = 0; class_i < class_n; class_i++) {
			centroids[class_i].x /= count[class_i];				//평균 구하기
			centroids[class_i].y /= count[class_i];
		}
	}
	*/
}
