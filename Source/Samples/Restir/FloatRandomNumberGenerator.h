//========================================================================
// Copyright (c) Logiciels Nebula Render, 2018 - All Rights Reserved
// Unauthorized copying of this file, via any medium is strictly prohibited
// Proprietary and confidential
//
//	Author					: Yann Clotioloman Yeo
//	E-Mail					: nebularender@gmail.com
//========================================================================

#pragma once

#include "Math/Math.h"
#include <atomic>
#include <random>

namespace Math { namespace Generator
{
	template <typename T>
	struct CommonRandomNumberGenerator
	{
		inline CommonRandomNumberGenerator(nbInt32 seed = -1)
		{
			if (seed < 0)
			{
				m_state = (nbUint32)time(0);
			}
			else
			{
				m_state = (nbUint32)seed;
			}
		}

		// Generate between [0...1]
		inline T generateUnsignedNormalized()
		{
			return static_cast <T> (xorshift32()) / static_cast <T> (s_uint32Max);
		}

		// Generate between [-1...1]
		inline T generateSignedNormalized()
		{
			return generateBeetween(static_cast <T>(-1), static_cast <T>(1));
		}

		// Generate between [a...b]
		inline T generateBeetween(T a, T b)
		{
			return a + static_cast <T> (xorshift32()) / (static_cast <T> (s_uint32Max / (b - a)));
		}

	private:
		inline nbUint32 xorshift32()
		{
			/* Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" */
			nbUint32 x = m_state;
			x ^= x << 13;
			x ^= x >> 17;
			x ^= x << 5;
			return m_state = x;
		}


		static const nbUint32 s_uint32Max = std::numeric_limits<nbUint32>::max();
		std::atomic<nbUint32> m_state;
	};



#if !defined(_WINDOWS)
	// https://social.msdn.microsoft.com/Forums/en-us/3263ab0b-da3f-4737-9d80-0788a5b426db/39randr39-identifier-not-found?forum=vcgeneral
	// https://selkie.macalester.edu/csinparallel/modules/MonteCarloSimulationExemplar/build/html/SeedingThreads/SeedEachThread.html
	#error rand is thread safe on Windows. Same on this platform?
#endif
	struct FastFloatRandomNumberGenerator
	{
		// Generate between [0...1]
		inline nbFloat32 generateUnsignedNormalized()
		{
			return static_cast <nbFloat32> (rand()) / static_cast <nbFloat32> (RAND_MAX);
		}

		// Generate between [-1...1]
		inline nbFloat32 generateSignedNormalized()
		{
			const float random = ((nbFloat32)rand()) / (nbFloat32)RAND_MAX;
			return (random * 2.0f) - 1.0f;
		}

		// Generate between [a...b]
		inline nbFloat32 generateBeetween(nbFloat32 a, nbFloat32 b)
		{
			const nbFloat32 random = ((nbFloat32)rand()) / (nbFloat32)RAND_MAX;
			const nbFloat32 range = b - a;
			return (random*range) + a;
		}
	};

}}

