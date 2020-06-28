/*
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * - Neither the name of prim nor the names of its contributors may be used to
 * endorse or promote products derived from this software without specific prior
 * written permission.
 *
 * See the NOTICE file distributed with this work for additional information
 * regarding copyright ownership.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#include <gtest/gtest.h>
#include <prim/prim.h>

#include <cmath>
#include <cstdio>

#include <map>
#include <random>

#include "mut/mut.h"

TEST(arithmeticMean, simple) {
  std::vector<u64> v2 = {1, 2, 3, 1, 2, 3, 1, 2, 3};
  ASSERT_EQ(mut::arithmeticMean(v2), 2.0);
  std::vector<f64> v3 = {1, 2, 3, 1, 2, 3};
  ASSERT_EQ(mut::arithmeticMean(v3), 2.0);
  std::vector<f64> v4 = {1, 2, 3, 1, 2, 3, 1, 2, 3};
  ASSERT_EQ(mut::arithmeticMean(v4), 2.0);
}

TEST(geometricMean, simple) {
  std::vector<u64> v1 = {20, 30, 1000};
  f64 product = 20 * 30 * 1000;
  f64 exp = std::pow(product, 1 / 3.0);
  ASSERT_NEAR(mut::geometricMean(v1), exp, 0.0001);
  std::vector<f64> v2 = {2.0, 8.0};
  ASSERT_NEAR(mut::geometricMean(v2), 4.0, 0.0001);
}

TEST(harmonicMean, simple) {
  std::vector<u64> v1 = {60, 30};
  ASSERT_NEAR(mut::harmonicMean(v1), 40.0, 0.0001);
}

TEST(variance, simple) {
  std::vector<u64> v2 = {1, 2, 3, 1, 2, 3, 1, 2, 3};
  f64 mean = mut::arithmeticMean(v2);
  ASSERT_NEAR(mut::variance(v2, mean), 0.6666667, 0.000001);
}

TEST(standardDeviation, simple) {
  std::vector<u64> v2 = {1, 2, 3, 1, 2, 3, 1, 2, 3};
  f64 mean = mut::arithmeticMean(v2);
  f64 vari = mut::variance(v2, mean);
  ASSERT_NEAR(mut::standardDeviation(vari), sqrt(vari), 0.000001);
}

TEST(slope, simple) {
  std::vector<f64> t1 = {0, 1, 2, 3, 4, 5};
  std::vector<f64> v1 = {1, 2, 3, 4, 5, 6};
  ASSERT_LT(std::abs(mut::slope(t1, v1) - 1.0), 0.01);
  std::vector<f64> v2 = {0, 2, 4, 6, 8, 10};
  ASSERT_LT(std::abs(mut::slope(t1, v2) - 2.0), 0.01);

  std::vector<f64> t3 = {3, 4, 5, 0, 1, 2};
  std::vector<f64> v3 = {4, 5, 6, 1, 2, 3};
  ASSERT_LT(std::abs(mut::slope(t3, v3) - 1.0), 0.01);
}

TEST(generateCumulativeDistribution, one_based) {
  std::vector<f64> pdist = {0.10, 0.15, 0.50, 0.25};
  std::vector<f64> cdist;
  mut::generateCumulativeDistribution(pdist, &cdist);
  ASSERT_EQ(cdist.size(), pdist.size());
  ASSERT_EQ(cdist[0], 0.00);
  ASSERT_EQ(cdist[1], 0.10);
  ASSERT_EQ(cdist[2], 0.25);
  ASSERT_EQ(cdist[3], 0.75);
}

TEST(generateCumulativeDistribution, two_based) {
  std::vector<f64> pdist = {0.20, 0.30, 1.00, 0.50};
  std::vector<f64> cdist;
  mut::generateCumulativeDistribution(pdist, &cdist);
  ASSERT_EQ(cdist.size(), pdist.size());
  ASSERT_EQ(cdist[0], 0.00);
  ASSERT_EQ(cdist[1], 0.10);
  ASSERT_EQ(cdist[2], 0.25);
  ASSERT_EQ(cdist[3], 0.75);
}

TEST(generateCumulativeDistribution, half_based) {
  std::vector<f64> pdist = {0.05, 0.075, 0.25, 0.125};
  std::vector<f64> cdist;
  mut::generateCumulativeDistribution(pdist, &cdist);
  ASSERT_EQ(cdist.size(), pdist.size());
  ASSERT_EQ(cdist[0], 0.00);
  ASSERT_EQ(cdist[1], 0.10);
  ASSERT_EQ(cdist[2], 0.25);
  ASSERT_EQ(cdist[3], 0.75);
}

TEST(searchCumulativeDistribution, dist) {
  // generate the cumulative distribution
  std::vector<f64> pdist = {0.10, 0.15, 0.50, 0.25};
  std::vector<f64> cdist;
  mut::generateCumulativeDistribution(pdist, &cdist);

  // initialize a random number generator and uniform real distribution
  std::mt19937_64 prng;
  std::seed_seq seed = {0xDEAFBEEF};
  prng.seed(seed);
  std::uniform_real_distribution<f64> dist;

  // perform many rounds and keep count
  const u64 ROUNDS = 40000000;
  std::vector<u64> counts(pdist.size(), 0);
  for (u64 round = 0; round < ROUNDS; round++) {
    f64 rnd = dist(prng);
    u64 loc = mut::searchCumulativeDistribution(cdist, rnd);
    counts.at(loc)++;
  }

  // verify the distribution matches the pdist
  for (u64 idx = 0; idx < pdist.size(); idx++) {
    f64 act = (f64)counts.at(idx) / ROUNDS;
    f64 exp = pdist.at(idx);
    ASSERT_NEAR(act, exp, 0.0001);
  }
}
