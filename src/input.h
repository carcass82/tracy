/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2017-2021
 */
#pragma once

struct Input
{
	enum Key
	{ 
		ESC = 0x1b,
		A = 'A', B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
		a = 'a', b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z
	};

	enum MouseButton { Left, Middle, Right, NumButtons };
	
	enum KeyGroup { Movement, NumGroups };

	struct mousestatus
	{
		vec2 pos;
		bool buttonstatus[NumButtons];
	};

	bool GetKeyStatus(uint8_t c) const
	{
		return keystatus[c];
	}
	
	bool GetKeyStatus(KeyGroup k) const
	{
		switch (k)
		{
		case Movement:
			return keystatus[W] || keystatus[S] || keystatus[A] || keystatus[D] || keystatus[Q] || keystatus[E];

		default:
			TracyLog("keygroup %d not recognized\n", k);
			return false;
		}
	}

	void SetKeyStatus(uint8_t c, bool value)
	{
		keystatus[c] = value;
	}

	void SetKeyStatus(KeyGroup k, bool value)
	{
		switch (k)
		{
		case Movement:
			keystatus[W] = keystatus[S] = keystatus[A] = keystatus[D] = keystatus[Q] = keystatus[E] = value;
			break;

		default:
			TracyLog("keygroup %d not recognized\n", k);
			break;
		}
	}

	void ResetKeyStatus(uint8_t c)
	{
		SetKeyStatus(c, false);
	}

	void ResetKeyStatus(KeyGroup k)
	{
		SetKeyStatus(k, false);
	}

	bool keystatus[0xff];
	mousestatus mouse;
	bool pending = false;
};
