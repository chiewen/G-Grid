#include "stdafx.h"
#include "hybrid_spatial_db.h"

#include "subnet_shortest_path.cuh"
//#include "check_device_info.cuh"
#include "index_.h"

#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include "z_order.h"

HySDB::HySDB::HySDB() {
	
}

int HySDB::HySDB::initial() {
	Index::Initialize();
	return 0;
}

int HySDB::HySDB::test() {
	int data[6] = {1, 0, 2, 2, 1, 3};
	thrust::inclusive_scan(thrust::host, data, data + 6, data); // in-place scan

	std::copy(data, data + 6, std::ostream_iterator<int>(std::cout, " "));

	return 0;
}

