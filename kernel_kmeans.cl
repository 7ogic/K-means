struct Point {
    float x, y;
};

__kernel void assignment(
										__global struct Point* centroids,
										__global struct Point* data,
										__global int* partitioned
										)
{
	int class_n = CLASS_N;
	int global_x = get_global_id(0);
	float min_dist = MAXFLOAT;
	float2 data_point = (float2)( data[global_x].x, data[global_x].y );
	float2 centroid_point;

	int i;
	for (i = 0; i < class_n; ++i) {
		centroid_point = (float2)(centroids[i].x, centroids[i].y);
		float dist = distance(data_point, centroid_point);
		
		if (dist < min_dist) {
			partitioned[global_x] = i;
			min_dist = dist;
		}
	}
}

/*
__kernel void update_count(
													__global struct Point* data,
													__global int* partitioned,
													__global struct Point* g_centroids,
													__global int* g_count,
													__local struct Point* l_centroids,	//sizeof(point)*num_wg
													__local int* l_count								//sizeof(int)*num_wg
													)
// 1 workitem do update 256 points
{
	int global_x = get_global_id(0);			// 1~data_n /4
	int l_i = get_local_id(0);					// 1~256
	int i, j;
	int p;
	int data_index = global_x*get_local_size(0);

	for (i = 0; i < CLASS_N; ++i) {			// 1 ~ 16
		//# 1. init local
		l_count[l_i] = 0;
		l_centroids[l_i].x = 0.0;
		l_centroids[l_i].y = 0.0;
		barrier(CLK_LOCAL_MEM_FENCE);

		//# 2. sum ... points as many as local size
		for (j = 0; j < get_local_size(0); ++j) {
			if (data_index+j < DATA_N && partitioned[data_index+j] == i ){
				l_count[l_i] += 1;
				l_centroids[l_i].x += data[data_index+j].x;
				l_centroids[l_i].y += data[data_index+j].y;
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		// 3. sum
		for (p = get_local_size(0) / 2; p >= 1; p = p >> 1) {
			if (l_i < p) {
				l_count[l_i] += l_count[l_i+p];
				l_centroids[l_i].x += l_centroids[l_i+p].x;
				l_centroids[l_i].y += l_centroids[l_i+p].y;
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}

		// 4. copy the result in global variable
		if (global_x < DATA_N/get_local_size(0) && l_i == 0) {
			g_centroids[i*get_num_groups(0)+get_group_id(0)].x = l_centroids[0].x;
			g_centroids[i*get_num_groups(0)+get_group_id(0)].y = l_centroids[0].y;
			g_count[i*get_num_groups(0)+get_group_id(0)] = l_count[0];			
		}
	}
}
*/

__kernel void update_count(
													__global struct Point* data,
													__global int* partitioned,
													__global struct Point* g_centroids,
													__global int* g_count,
													__local struct Point* l_centroids,	//sizeof(point)*num_wg
													__local int* l_count								//sizeof(int)*num_wg
													)
{
	int global_x = get_global_id(0);			// 1~data_n /4
	int l_i = get_local_id(0);					// 1~256
	int i;
	int p;

	for (i = 0; i < CLASS_N; ++i) {			// 1 ~ 16
		// 1. init local
		if (global_x < DATA_N && partitioned[global_x] == i) {			//
			l_count[l_i] = 1;
			l_centroids[l_i].x = data[global_x].x;
			l_centroids[l_i].y = data[global_x].y;
		} else {
			l_count[l_i] = 0;
			l_centroids[l_i].x = 0.0;
			l_centroids[l_i].y = 0.0;	
		}	
		barrier(CLK_LOCAL_MEM_FENCE);

		// 3. sum
		for (p = get_local_size(0) / 2; p >= 1; p = p >> 1) {
			if (l_i < p) {
				l_count[l_i] += l_count[l_i+p];
				l_centroids[l_i].x += l_centroids[l_i+p].x;
				l_centroids[l_i].y += l_centroids[l_i+p].y;
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		// 4. copy the result in global variable
		if (global_x < DATA_N && l_i == 0) {
			g_centroids[i*get_num_groups(0)+get_group_id(0)].x = l_centroids[0].x;
			g_centroids[i*get_num_groups(0)+get_group_id(0)].y = l_centroids[0].y;
			g_count[i*get_num_groups(0)+get_group_id(0)] = l_count[0];			
		}
	}
}
