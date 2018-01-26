#pragma once

#include "stdafx.h"
#include "gtest_prod.h"

#include "index_.h"
#include <list>
#include <vector>

struct Message {
	int oid = 0;
	int eid = 0;
	int rho = 0;
	int t = 0;
};

class MessageBucket {
public:
	static const int kRhoB = 32;

	Message A[kRhoB];
	int n;
	int t;
	MessageBucket* p = nullptr;
};

class MessageLists {
public:
	static const int kTimeDelta = 2;
	static MessageBucket* lists_[Index::kCellNum];

	static void MessageCleaning(std::vector<int> lists, int message_out_num, Message* messages);
};