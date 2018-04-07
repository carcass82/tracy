#pragma once

#include <cstdint>
#include <limits>
#include <cassert>
#include <cmath>

namespace vutil
{
    //
    // useful functions
    //
    template<typename T>
    constexpr const T& min(const T& a, const T& b) { return !(b < a)? a : b; }

    template<typename T>
    constexpr const T& max(const T& a, const T& b) { return (a < b)? b : a; }

    template<typename T>
    constexpr const T& clamp(const T& a, T lower, T upper) { return min(max(a, lower), upper); }

    template<typename T>
    constexpr const T& saturate(const T& a) { return clamp(a, T(0), T(1)); }

    template<typename T, size_t N>
    constexpr uint32_t array_size(const T(&)[N]) { return N; }
}

namespace vmath
{
    //
    // math constants
    //
    constexpr float PI = 3.14159265f;
    constexpr float EPS = std::numeric_limits<float>::epsilon();


    //
    // conversion utils
    //
    constexpr float radians(float deg)                { return deg * PI / 180.0f; }
    constexpr float degrees(float rad)                { return rad * 180.0f / PI; }
    constexpr float lerp(float v0, float v1, float t) { return (1.0f - t) * v0 + t * v1; }

    constexpr float fastsqrtf(float x)
    {
        assert(x >= .0f && x < std::numeric_limits<float>::infinity());

        auto sqrtfimpl = [](float x, float curr, float prev) -> float
        {
            auto recursive = [](float x, float curr, float prev, const auto& lambda) -> float
            {
                return (curr == prev)? curr : lambda(x, 0.5 * (curr + x / curr), curr, lambda);
            };

            return recursive(x, curr, prev, recursive);
        };

        return sqrtfimpl(x, x, .0f);
    }




    //
    // useful types
    //
    struct vec2
    {
        union {
            float v[2];
            struct { float x, y; };
            struct { float s, t; };
            struct { float w, h; };
        };

        constexpr inline float& operator[](size_t i)             { assert(i < 2); return v[i]; }
        constexpr inline const float& operator[](size_t i) const { assert(i < 2); return v[i]; }

        constexpr vec2() : v{} {}
        constexpr vec2(float _v) : v{_v, _v} {}
        constexpr vec2(float _v1, float _v2) : v{ _v1, _v2 } {}
        constexpr vec2(std::initializer_list<float> _v) : v{*_v.begin(), *(_v.begin() + 1)} { assert(_v.size() == 2); }
    };

    struct vec3
    {
        union {
            float v[3];
            struct { float x, y, z; };
            struct { float r, g, b; };
        };

        constexpr inline float& operator[](size_t i)             { assert(i < 3); return v[i]; }
        constexpr inline const float& operator[](size_t i) const { assert(i < 3); return v[i]; }

        constexpr vec3() : v{} {}
        constexpr vec3(float _v) : v{_v, _v, _v} {}
        constexpr vec3(float _v1, float _v2, float _v3) : v{ _v1, _v2, _v3 } {}
        constexpr vec3(std::initializer_list<float> _v) : v{*_v.begin(), *(_v.begin() + 1), *(_v.begin() + 2)} { assert(_v.size() == 3); }
    };

    struct vec4
    {
        union {
            float v[4];
            struct { float x, y, z, w; };
            struct { float r, g, b, a; };
        };

        constexpr inline float& operator[](size_t i)             { assert(i < 4); return v[i]; }
        constexpr inline const float& operator[](size_t i) const { assert(i < 4); return v[i]; }

        constexpr vec4() : v{} {}
        constexpr vec4(float _v) : v{_v, _v, _v, _v} {}
        constexpr vec4(float _v1, float _v2, float _v3, float _v4) : v{ _v1, _v2, _v3, _v4 } {}
        constexpr vec4(std::initializer_list<float> _v) : v{*_v.begin(), *(_v.begin() + 1), *(_v.begin() + 2), *(_v.begin() + 3)} { assert(_v.size() == 4); }
    };

    struct mat4
    {
        union {
            vec4 m[4];
            struct {
                float _m00, _m10, _m20, _m30;
                float _m01, _m11, _m21, _m31;
                float _m02, _m12, _m22, _m32;
                float _m03, _m13, _m23, _m33;
            };
        };

        constexpr vec4& operator[](size_t i)             { assert(i < 4); return m[i]; }
        constexpr const vec4& operator[](size_t i) const { assert(i < 4); return m[i]; }

        constexpr mat4() : m{} {}
        constexpr mat4(float _i) : m{} { _m00 = _m11 =_m22 = _m33 = _i; }
        constexpr mat4(std::initializer_list<vec4> _m) : m{*_m.begin(), *(_m.begin() + 1), *(_m.begin() + 2), *(_m.begin() + 3)} { assert(_m.size() == 4); }
    };

    struct mat3
    {
        union {
            vec3 m[3];
            struct {
                float _m00, _m10, _m20;
                float _m01, _m11, _m21;
                float _m02, _m12, _m22;
            };
        };

        constexpr vec3& operator[](size_t i)             { assert(i < 3); return m[i]; }
        constexpr const vec3& operator[](size_t i) const { assert(i < 3); return m[i]; }

        constexpr mat3() : m{} {}
        constexpr mat3(float _i) : m{} { _m00 = _m11 =_m22 = _i; }
        constexpr mat3(const mat4& _m) : _m00(_m._m00), _m10(_m._m10), _m20(_m._m20), _m01(_m._m01), _m11(_m._m11), _m21(_m._m21), _m02(_m._m02), _m12(_m._m12), _m22(_m._m22) {}
        constexpr mat3(std::initializer_list<vec3> _m) : m{*_m.begin(), *(_m.begin() + 1), *(_m.begin() + 2)} { assert(_m.size() == 3); }
    };


    //
    // operators
    //
    inline constexpr vec2 operator+(const vec2& a, float b)         { return { a.x + b, a.y + b }; }
    inline constexpr vec2 operator+(float b, const vec2& a)         { return { a.x + b, a.y + b }; }
    inline constexpr vec2 operator+(const vec2& a, const vec2& b)   { return { a.x + b.x, a.y + b.y }; }
    inline constexpr vec2 operator-(const vec2& a, float b)         { return { a.x - b, a.y - b }; }
    inline constexpr vec2 operator-(float b, const vec2& a)         { return { a.x - b, a.y - b }; }
    inline constexpr vec2 operator-(const vec2& a, const vec2& b)   { return { a.x - b.x, a.y - b.y }; }
    inline constexpr vec2 operator*(const vec2& a, float b)         { return { a.x * b, a.y * b }; }
    inline constexpr vec2 operator*(float b, const vec2& a)         { return { a.x * b, a.y * b }; }
    inline constexpr vec2 operator*(const vec2& a, const vec2& b)   { return { a.x * b.x, a.y * b.y }; }
    inline constexpr vec2 operator/(const vec2& a, float b)         { return { a.x / b, a.y / b }; }
    inline constexpr vec2 operator/(float b, const vec2& a)         { return { a.x / b, a.y / b }; }
    inline constexpr vec2 operator/(const vec2& a, const vec2& b)   { return { a.x / b.x, a.y / b.y }; }
    inline constexpr vec2& operator+=(vec2& a, float b)             { a.x += b; a.y += b; return a; }
    inline constexpr vec2& operator-=(vec2& a, float b)             { a.x -= b; a.y -= b; return a; }
    inline constexpr vec2& operator*=(vec2& a, float b)             { a.x *= b; a.y *= b; return a; }
    inline constexpr vec2& operator/=(vec2& a, float b)             { a.x /= b; a.y /= b; return a; }
    inline constexpr bool operator==(const vec2& a, const vec2& b)  { return fabsf(a.x - b.x) < EPS && fabsf(a.y - b.y) < EPS; }
    inline constexpr bool operator!=(const vec2& a, const vec2& b)  { return !(a == b); }

    inline constexpr vec3 operator+(const vec3& a, float b)         { return { a.x + b, a.y + b, a.z + b }; }
    inline constexpr vec3 operator+(float b, const vec3& a)         { return { a.x + b, a.y + b, a.z + b }; }
    inline constexpr vec3 operator+(const vec3& a, const vec3& b)   { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
    inline constexpr vec3 operator-(const vec3& a, float b)         { return { a.x - b, a.y - b, a.z - b }; }
    inline constexpr vec3 operator-(float b, const vec3& a)         { return { a.x - b, a.y - b, a.z - b }; }
    inline constexpr vec3 operator-(const vec3& a, const vec3& b)   { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
    inline constexpr vec3 operator*(const vec3& a, float b)         { return { a.x * b, a.y * b, a.z * b }; }
    inline constexpr vec3 operator*(float b, const vec3& a)         { return { a.x * b, a.y * b, a.z * b }; }
    inline constexpr vec3 operator*(const vec3& a, const vec3& b)   { return { a.x * b.x, a.y * b.y, a.z * b.z }; }
    inline constexpr vec3 operator/(const vec3& a, float b)         { return { a.x / b, a.y / b, a.z / b }; }
    inline constexpr vec3 operator/(float b, const vec3& a)         { return { a.x / b, a.y / b, a.z / b }; }
    inline constexpr vec3 operator/(const vec3& a, const vec3& b)   { return { a.x / b.x, a.y / b.y, a.z / b.z }; }
    inline constexpr vec3& operator+=(vec3& a, const vec3& b)       { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
    inline constexpr vec3& operator-=(vec3& a, const vec3& b)       { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
    inline constexpr vec3& operator*=(vec3& a, const vec3& b)       { a.x *= b.x; a.y *= b.y; a.z *= b.z; return a; }
    inline constexpr vec3& operator/=(vec3& a, const vec3& b)       { a.x /= b.x; a.y /= b.y; a.z /= b.z; return a; }
    inline constexpr vec3& operator+=(vec3& a, float b)             { a.x += b; a.y += b; a.z += b; return a; }
    inline constexpr vec3& operator-=(vec3& a, float b)             { a.x -= b; a.y -= b; a.z -= b; return a; }
    inline constexpr vec3& operator*=(vec3& a, float b)             { a.x *= b; a.y *= b; a.z *= b; return a; }
    inline constexpr vec3& operator/=(vec3& a, float b)             { a.x /= b; a.y /= b; a.z /= b; return a; }
    inline constexpr bool operator==(const vec3& a, const vec3& b)  { return fabsf(a.x - b.x) < EPS && fabsf(a.y - b.y) < EPS && fabsf(a.z - b.z) < EPS; }
    inline constexpr bool operator!=(const vec3& a, const vec3& b)  { return !(a == b); }

    inline constexpr vec4 operator+(const vec4& a, float b)         { return { a.x + b, a.y + b, a.z + b, a.w + b }; }
    inline constexpr vec4 operator+(float b, const vec4& a)         { return { a.x + b, a.y + b, a.z + b, a.w + b }; }
    inline constexpr vec4 operator+(const vec4& a, const vec4& b)   { return { a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w }; }
    inline constexpr vec4 operator-(const vec4& a, float b)         { return { a.x - b, a.y - b, a.z - b, a.w - b }; }
    inline constexpr vec4 operator-(float b, const vec4& a)         { return { a.x - b, a.y - b, a.z - b, a.w - b }; }
    inline constexpr vec4 operator-(const vec4& a, const vec4& b)   { return { a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w }; }
    inline constexpr vec4 operator*(const vec4& a, float b)         { return { a.x * b, a.y * b, a.z * b, a.w * b }; }
    inline constexpr vec4 operator*(float b, const vec4& a)         { return { a.x * b, a.y * b, a.z * b, a.w * b }; }
    inline constexpr vec4 operator*(const vec4& a, const vec4& b)   { return { a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w }; }
    inline constexpr vec4 operator/(const vec4& a, float b)         { return { a.x / b, a.y / b, a.z / b, a.w / b }; }
    inline constexpr vec4 operator/(float b, const vec4& a)         { return { a.x / b, a.y / b, a.z / b, a.w / b }; }
    inline constexpr vec4 operator/(const vec4& a, const vec4& b)   { return { a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w }; }
    inline constexpr vec4& operator+=(vec4& a, const vec4& b)       { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; return a; }
    inline constexpr vec4& operator-=(vec4& a, const vec4& b)       { a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w; return a; }
    inline constexpr vec4& operator*=(vec4& a, const vec4& b)       { a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w; return a; }
    inline constexpr vec4& operator/=(vec4& a, const vec4& b)       { a.x /= b.x; a.y /= b.y; a.z /= b.z; a.w /= b.w; return a; }
    inline constexpr vec4& operator+=(vec4& a, float b)             { a.x += b; a.y += b; a.z += b; a.w += b; return a; }
    inline constexpr vec4& operator-=(vec4& a, float b)             { a.x -= b; a.y -= b; a.z -= b; a.w -= b; return a; }
    inline constexpr vec4& operator*=(vec4& a, float b)             { a.x *= b; a.y *= b; a.z *= b; a.w *= b; return a; }
    inline constexpr vec4& operator/=(vec4& a, float b)             { a.x /= b; a.y /= b; a.z /= b; a.w /= b; return a; }
    inline constexpr bool operator==(const vec4& a, const vec4& b)  { return fabsf(a.x - b.x) < EPS && fabsf(a.y - b.y) < EPS && fabsf(a.z - b.z) < EPS && fabsf(a.w - b.w) < EPS; }
    inline constexpr bool operator!=(const vec4& a, const vec4& b)  { return !(a == b); }

    inline constexpr mat3 operator*(const mat3& a, const mat3& b)
    {
        return
        {
            { a[0] * b[0].x + a[1] * b[0].y + a[2] * b[0].z },
            { a[0] * b[1].x + a[1] * b[1].y + a[2] * b[1].z },
            { a[0] * b[2].x + a[1] * b[2].y + a[2] * b[2].z }
        };
    }

    inline constexpr mat4 operator*(const mat4& a, const mat4& b)
    {
        return
        {
            { a[0] * b[0].x + a[1] * b[0].y + a[2] * b[0].z + a[3] * b[0].w },
            { a[0] * b[1].x + a[1] * b[1].y + a[2] * b[1].z + a[3] * b[1].w },
            { a[0] * b[2].x + a[1] * b[2].y + a[2] * b[2].z + a[3] * b[2].w },
            { a[0] * b[3].x + a[1] * b[3].y + a[2] * b[3].z + a[3] * b[3].w }
        };
    }


    //
    // trig functions
    //
    constexpr float dot(const vec3& a, const vec3& b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    constexpr float length2(const vec3& a)
    {
        return dot(a, a);
    }

    float length(const vec3& a)
    {
        return fastsqrtf(length2(a));
    }

    constexpr vec3 cross(const vec3& a, const vec3& b)
    {
        return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
    }

    vec3 normalize(const vec3& a)
    {
        return a * (1.0f / length(a));
    }

    constexpr vec3 reflect(const vec3& I, const vec3& N)
    {
        return I - N * dot(N, I) * 2.f;
    }

    constexpr vec3 refract(const vec3& I, const vec3& N, float eta)
    {
        const float NdotI = dot(N, I);
        const float k = 1.f - eta * eta * (1.f - NdotI * NdotI);

        return (k >= .0f)? vec3(eta * I - (eta * NdotI + fastsqrtf(k)) * N) : vec3();
    }

    constexpr mat4 translate(const mat4& m, const vec3& v)
    {
        return
        {
            m[0],
            m[1],
            m[2],
            m[0] * v[0] + m[1] * v[1] + m[2] * v[2] + m[3]
        };
    }

    mat4 rotate(const mat4& m, float angle, const vec3& axis)
    {
        vec3 axis_n = normalize(axis);

        /*
        [1             0              0 0]
        [0 cos(-X Angle) -sin(-X Angle) 0]
        [0 sin(-X Angle)  cos(-X Angle) 0]
        [0             0              0 1]
        */
        const float cx = cosf(angle * axis_n.x);
        const float sx = sinf(angle * axis_n.x);
        mat4 rotX
        {
            { 1,   0,   0,  0 },
            { 0,  cx,  sx,  0 },
            { 0, -sx,  cx,  0 },
            { 0,   0,   0,  1 }
        };

        /*
        [cos(-Y Angle) 0 sin(-Y Angle) 0]
        [0 1 0 0]
        [-sin(-Y Angle) 0 cos(-Y Angle) 0]
        [0 0 0 1]
        */
        const float cy = cosf(angle * axis_n.y);
        const float sy = sinf(angle * axis_n.y);
        mat4 rotY
        {
            { cy,  0, -sy,  0 },
            {  0,  1,   0,  0 },
            { sy,  0,  cy,  0 },
            {  0,  0,   0,  1 }
        };

        /*
        [cos(-Z Angle) -sin(-Z Angle) 0 0]
        [sin(-Z Angle) cos(-Z Angle) 0 0]
        [0 0 1 0]
        [0 0 0 1]
        */
        const float cz = cosf(angle * axis_n.z);
        const float sz = sinf(angle * axis_n.z);
        mat4 rotZ
        {
            {  cz,  sz,  0,  0 },
            { -sz,  cz,  0,  0 },
            {   0,   0,  1,  0 },
            {   0,   0,  0,  1 }
        };

        return m * rotX * rotY * rotZ;
    }

    constexpr mat4 scale(const mat4& m, const vec3& v)
    {
        return
        {
            m[0] * v.x,
            m[1] * v.y,
            m[2] * v.z,
            m[3]
        };
    }

    mat4 lookAt(const vec3& eye, const vec3& center, const vec3& up)
    {
        const vec3 f(normalize(center - eye));
        const vec3 s(normalize(cross(up, f)));
        const vec3 u(cross(f, s));

        return
        {
            {          s.x,          u.x,          f.x,  0.0f },
            {          s.y,          u.y,          f.y,  0.0f },
            {          s.z,          u.z,          f.z,  0.0f },
            { -dot(s, eye), -dot(u, eye), -dot(f, eye),  1.0f }
        };
    }

    constexpr mat4 perspective(float fovy, float aspect, float znear, float zfar)
    {
        const float F = 1.0f / tanf(fovy / 2.0f);
        const float delta = zfar - znear;

        return
        {
            { F / aspect,   0.0f,                           0.0f,  0.0f },
            {       0.0f,      F,                           0.0f,  0.0f },
            {       0.0f,   0.0f,         (zfar + znear) / delta,  1.0f },
            {       0.0f,   0.0f, -(2.0f * zfar * znear) / delta,  0.0f }
        };
    }
}

namespace vgfx
{
    //
    // gfx related helpers
    //
    struct bbox
    {
        vmath::vec3 vmin{};
        vmath::vec3 vmax{};

        void Add(const vmath::vec3& v)
        {
            vmin.x = vutil::min(vmin.x, v.x);
            vmin.y = vutil::min(vmin.y, v.y);
            vmin.z = vutil::min(vmin.z, v.z);
            vmax.x = vutil::max(vmax.x, v.x);
            vmax.y = vutil::max(vmax.y, v.y);
            vmax.z = vutil::max(vmax.z, v.z);
        }
        vmath::vec3 Size() const   { return { vmax.x - vmin.x, vmax.y - vmin.y, vmax.z - vmin.z };  }
        vmath::vec3 Center() const { return { (vmax.x + vmin.x) / 2.f, (vmax.y + vmin.y) / 2.f, (vmax.z + vmin.z) / 2.f }; }
    };
}
