void _build_voxels(const int grid_size, const float step_d, const float step_h, const float step_w,
                   const float min_d, const float min_h, const float min_w,
                   const int filter_h, const int filter_w, const int num_classes,
                   const int height, const int width, const int* grid_indexes, const int* labels,
                   const float* pmatrix, int* top_locations, int* top_labels, int device_id);
