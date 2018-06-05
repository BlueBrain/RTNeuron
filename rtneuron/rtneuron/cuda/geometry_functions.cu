//////////////////////////////////////////////////////////////////////
// RTNeuron
//
// Copyright (c) 2006-2014 Cajal Blue Brain, BBP/EPFL
// All rights reserved. Do not distribute without permission.
//
// Responsible Author: Juan Hernando Vieites (JHV)
// contact: jhernando@fi.upm.es
//////////////////////////////////////////////////////////////////////

#ifndef GEOMETRY_FUNCTIONS_CU
#define GEOMETRY_FUNCTIONS_CU

/* A simple Matrix type declaration
   The notation used is the same as OSG (left multiplication: v x M) */
struct Matrix4x4
{
    float _m[4][4];
};

__device__ float3 operator^(const float3 &u, const float3 &v)
{
    return make_float3(u.y * v.z - u.z * v.y,
                       u.z * v.x - u.x * v.z,
                       u.x * v.y - u.y * v.x);
}

__device__ float operator*(const float3 &u, const float3 &v)
{
    return u.x * v.x + u.y * v.y + u.z * v.z;
}

__device__ float3 operator+(const float3 &u, const float3 &v)
{
    return make_float3(u.x + v.x, u.y + v.y, u.z + v.z);
}

__device__ float3 &operator+=(float3 &u, const float3 v)
{
    u.x += v.x;
    u.y += v.y;
    u.z += v.z;
    return u;
}

__device__ float3 operator-(const float3 &u, const float3 &v)
{
    return make_float3(u.x - v.x, u.y - v.y, u.z - v.z);
}

__device__ float3 &operator-=(float3 &u, const float3 &v)
{
    u.x -= v.x;
    u.y -= v.y;
    u.z -= v.z;
    return u;
}

__device__ float3 operator*(const float3 u, float a)
{
    return make_float3(u.x * a, u.y * a, u.z * a);
}

__device__ float3 operator*(float a , const float3 u)
{
    return make_float3(u.x * a, u.y * a, u.z * a);
}

__device__ void normalize(float3 &v)
{
    float length = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    v.x /= length;
    v.y /= length;
    v.z /= length;
}

__device__ void normalize(float2 &v)
{
    float length = sqrt(v.x * v.x + v.y * v.y);
    v.x /= length;
    v.y /= length;
}

__device__ float3 operator-(const float3 &u)
{
    return make_float3(-u.x, -u.y, -u.z);
}

__device__ float length(const float3 &u)
{
    return sqrt(u.x * u.x + u.y * u.y + u.z * u.z);
}

__device__ float4 operator*(const float3 &v, const Matrix4x4 &m)
{
    return make_float4
        (v.x * m._m[0][0] + v.y * m._m[1][0] + v.z * m._m[2][0] + m._m[3][0],
         v.x * m._m[0][1] + v.y * m._m[1][1] + v.z * m._m[2][1] + m._m[3][1],
         v.x * m._m[0][2] + v.y * m._m[1][2] + v.z * m._m[2][2] + m._m[3][2],
         v.x * m._m[0][3] + v.y * m._m[1][3] + v.z * m._m[2][3] + m._m[3][3]);
}

__device__ float3 mult3(const float3 &v, const Matrix4x4 &m)
{
    return make_float3
        (v.x * m._m[0][0] + v.y * m._m[1][0] + v.z * m._m[2][0] + m._m[3][0],
         v.x * m._m[0][1] + v.y * m._m[1][1] + v.z * m._m[2][1] + m._m[3][1],
         v.x * m._m[0][2] + v.y * m._m[1][2] + v.z * m._m[2][2] + m._m[3][2]);
}

__device__ float3 project(const float3 &v, const Matrix4x4 &m)
{
    float w =
        v.x * m._m[0][3] + v.y * m._m[1][3] + v.z * m._m[2][3] + m._m[3][3];
    float3 t = make_float3
        (v.x * m._m[0][0] + v.y * m._m[1][0] + v.z * m._m[2][0] + m._m[3][0],
         v.x * m._m[0][1] + v.y * m._m[1][1] + v.z * m._m[2][1] + m._m[3][1],
         v.x * m._m[0][2] + v.y * m._m[1][2] + v.z * m._m[2][2] + m._m[3][2]);
    t.x /= w;
    t.y /= w;
    t.z /= w;
    return t;
}

__device__ float3 get_scale(const Matrix4x4 &m)
{
    return make_float3
        (length(make_float3(m._m[0][0], m._m[0][1], m._m[0][2])),
         length(make_float3(m._m[1][0], m._m[1][1], m._m[1][2])),
         length(make_float3(m._m[2][0], m._m[2][1], m._m[2][2])));
}

#endif
