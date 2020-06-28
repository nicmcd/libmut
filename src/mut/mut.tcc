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
#ifndef MUT_MUT_TCC_
#define MUT_MUT_TCC_

#ifndef MUT_MUT_H_
#error "do not include this file, use the .h instead"
#else  // MUT_MUT_H_

#include <cassert>
#include <cmath>

#include <limits>
#include <numeric>
#include <vector>

namespace mut {

template <typename T>
f64 arithmeticMean(const std::vector<T>& _vals) {
  f64 sum = 0.0;
  for (T x : _vals) {
    sum += x;
  }
  return sum / _vals.size();
}

template <typename T>
f64 geometricMean(const std::vector<T>& _vals) {
  // modified from https://stackoverflow.com/a/19982259/2116585
  f64 m = 1.0;
  u64 ex = 0;
  double invN = 1.0 / _vals.size();
  for (T x : _vals) {
    int i;
    f64 f1 = std::frexp(x, &i);
    m *= f1;
    ex += i;
  }
  constexpr int radix = std::numeric_limits<f64>::radix;
  return (std::pow(radix, ex * invN) * std::pow(m, invN));
}

template <typename T>
f64 harmonicMean(const std::vector<T>& _vals) {
  f64 sum = 0.0;
  for (T x : _vals) {
    sum += 1.0 / x;
  }
  return _vals.size() / sum;
}

template <typename T>
f64 variance(const std::vector<T>& _vals, f64 _arithmetic_mean) {
  f64 diffSum = 0.0;
  for (auto it = _vals.cbegin(); it != _vals.cend(); ++it) {
    f64 diff = *it - _arithmetic_mean;
    diffSum += (diff * diff);
  }
  return diffSum / _vals.size();
}

template <typename T>
f64 standardDeviation(f64 _variance) {
  return sqrt(_variance);
}

template <typename T>
f64 slope(const std::vector<T>& _x, const std::vector<T>& _y) {
  assert(_x.size() == _y.size());

  // get a size
  const f64 n = _x.size();

  // sums the vectors
  const f64 xs = std::accumulate(_x.cbegin(), _x.cend(), 0.0);
  const f64 ys = std::accumulate(_y.cbegin(), _y.cend(), 0.0);

  // compute the inner products
  const f64 xx = std::inner_product(_x.cbegin(), _x.cend(), _x.cbegin(), 0.0);
  const f64 xy = std::inner_product(_x.cbegin(), _x.cend(), _y.cbegin(), 0.0);

  // compute the slope
  return (n * xy - xs * ys) / (n * xx - xs * xs);
}

template <typename T>
void generateCumulativeDistribution(const std::vector<T>& _pdist,
                                    std::vector<T>* _cdist) {
  // determine the sum
  T sum = std::accumulate(_pdist.cbegin(), _pdist.cend(), 0.0);
  assert(sum > 0.0);

  // generate the cumulative distribution
  _cdist->clear();
  _cdist->resize(_pdist.size(), 0.0);
  T csum = 0.0;
  for (u64 idx = 0; idx < _pdist.size(); idx++) {
    _cdist->at(idx) = csum / sum;
    csum += _pdist.at(idx);
  }
}

template <typename T>
u64 searchCumulativeDistribution(const std::vector<T>& _cdist, T _value) {
  assert(_value >= 0.0 && _value <= 1.0);
  u32 bot = 0;
  u32 top = _cdist.size();

  while (true) {
    assert(top > bot);
    u32 span = top - bot;
    u32 mid = (span / 2) + bot;
    if (span == 1) {
      // done! return the index
      return mid;
    } else if (_cdist.at(mid) < _value) {
      // raise the bottom
      bot = mid;
    } else {
      // lower the top
      top = mid;
    }
  }
}

}  // namespace mut

#endif  // MUT_MUT_H_
#endif  // MUT_MUT_TCC_
