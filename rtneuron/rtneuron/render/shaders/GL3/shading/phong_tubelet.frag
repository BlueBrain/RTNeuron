/* Copyright (c) 2006-2018, École Polytechnique Fédérale de Lausanne (EPFL) /
 *                           Blue Brain Project and
 *                          Universidad Politécnica de Madrid (UPM)
 *                          Juan Hernando <juan.hernando@epfl.ch>
 *
 * This file is part of RTNeuron <https://github.com/BlueBrain/RTNeuron>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#version 410

#extension GL_EXT_geometry_shader4 : enable

#define MULTISAMPLE
$DEFINES

uniform mat4x4 osg_ProjectionMatrix;

flat in vec4 c1;
flat in vec4 c2;
flat in vec4 color1;
flat in vec4 color2;
flat in vec4 planes[2];

#ifdef ACCURATE_HEADLIGHT
in vec3 light;
#endif
in float reportedVariable;

#ifdef MULTISAMPLE
sample in vec3 eye;
#else
in vec3 eye;
#endif

bool solved = false;
vec3 normal = vec3(0.0, 0.0, 1.0);
vec4 interpColor;
vec3 normalized_eye;

/**
 */
float cathetus(float hypot, float cat)
{
    return sqrt(hypot * hypot - cat * cat);
}
float hypot(float a, float b)
{
    return sqrt(a * a + b * b);
}

float xAxisIntersection(vec2 P, vec2 v)
{
    return P.x - P.y * v.x / v.y;
}

float sphereLineTest(vec3 c, float r, vec3 d)
{
    /* Real Time Rendering pg. 571 */
    float s = dot(c, d);
    float c2 = dot(c, c);
    float r2 = r * r;
    if (s < 0 && c2 > r2)
        return -1;
    float m2 = c2 - s * s;
    if (m2 > r2)
        return -1;
    float q = sqrt(r2 - m2);

    return (c2 > r2 ? s - q : s + q);
}

/**
   @param V the cone apex.
   @param v the cone axis direction.
   @param cos_aperture cos(a), a = aperture angle.
   @param d ray direction, the ray is supposed to pass through the origin.
 */
vec2 coneLineTest(vec3 V, vec3 v, float cos_aperture, vec3 d)
{
    float t;

    /* Computing the normal of the plane containing the line and the
       cone apex. */
    vec3 n = cross(d, -V);
    float tmp = length(n);
    n /= tmp;
    if (tmp < 1e-16)
    {
        /* The line contains the apex */
        if (abs(dot(d, v) - cos_aperture) < 1e-16)
        {
            /* The line is on the cone surface. The value of t is undefined
               because the cone and the line are infinite.
               We chose to return 0. */
            t = 0;
            return vec2(-1, 0);
        }
        else
        {
            t = length(V);
            return vec2(-1, 0);
        }
    }
    float v_dot_n = dot(n, v);
    if (dot(v, n) > 0)
    {
        n = -n;
        v_dot_n = -v_dot_n;
    }
    vec3 y = cross(n, d);

    /* Fast discard */
    float cos_a = cos_aperture;
    float cos_t = sqrt(1 - v_dot_n * v_dot_n);
    if (cos_t < cos_a)
    {
        t = 0;
        return vec2(-1, 0);
    }

    if (cos_t == cos_a)
    {
        /* The plane is tangent to the cone. However, note that the line
           can only be tangent to the cone, not on the surface. */
        /* Projecting v on the plane defined by x = d, y = n x d. */
        normalize(y);
        vec2 pV = vec2(dot(d, V), dot(y, V));
        vec2 pv = vec2(dot(d, v), dot(y, v));
        /* Solving the intersections between pV + t * pv and the x axis */
        t = xAxisIntersection(pV, pv);
        return vec2(t, hypot(pV.x - t, pV.y) * cos_a);
    }

    /* The line intersects the cone twice we return the nearest valid
       intersection (beware that the cone is floatd) */
    float cos_p = cos_a / cos_t;
    float sin_p = sqrt(1 - cos_p * cos_p);

    vec2 pV = vec2(dot(d, V), dot(y, V));
    vec2 pv = vec2(dot(d, v), dot(y, v));
    vec2 pv1 = vec2(cos_p * pv.x - sin_p * pv.y, sin_p * pv.x + cos_p * pv.y);
    vec2 pv2 = vec2(cos_p * pv.x + sin_p * pv.y, -sin_p * pv.x + cos_p * pv.y);
    normalize(pv1);
    normalize(pv2);
    float x1 = -1, x2 = -1;
    if (pV.y * pv1.y < 0)
        x1 = xAxisIntersection(pV, pv1);
    if (pV.y * pv2.y < 0)
        x2 = xAxisIntersection(pV, pv2);
    if (x1 > 0)
    {
        if (x2 > 0)
        {
            t = min(x1, x2);
        }
        else
            t = x1;
    }
    else
    {
        if (x2 > 0)
            t = x2;
        else
            return vec2(-1, 0);
    }
    return vec2(t, hypot(pV.x - t, pV.y) * cos_a);
}

/**
   Returns the coordinates of the tangency point of a line passing by
   a point located at (d, 0) with d > 0 and tangent to a circle
   centered at origin and radius r.
 */
vec2 tangencyPoint(float radius, float distance)
{
    float a = (radius * radius) / distance;
    return vec2(a, cathetus(radius, a));
}

/**
   Returns the tangency points of the line tangent to two circles whose centers
   are on the x-axis. The radius r1 refers to the circle on the left and r2
   to the circle on the right. The tangency points returned are relative to
   the circle on the left (r1) if distance > 0 and for the circle on the
   right (r2) otherwise.
 */
vec2 tangencyPoints(float distance, float r1, float r2)
{
    if (distance > 0)
    {
        /* Returning tangency points for circle 1 */
        if (r1 > r2)
        {
            float r = r1 - r2;
            vec2 t = tangencyPoint(r, distance);
            return vec2(t * (r1 / r));
        }
        else if (r1 < r2)
        {
            float r = r2 - r1;
            vec2 t = tangencyPoint(r, distance);
            t.x = -t.x;
            return vec2(t * (r2 / r - 1));
        }
        else
        {
            return vec2(0, r1);
        }
    }
    else
    {
        /* Returning tangency points for circle 2 */
        if (r2 > r1)
        {
            float r = r2 - r1;
            vec2 t = tangencyPoint(r, -distance);
            t.x = -t.x;
            return vec2(t * (r2 / r));
        }
        else if (r2 < r1)
        {
            float r = r1 - r2;
            vec2 t = tangencyPoint(r, -distance);
            return vec2(t * (r1 / r - 1));
        }
        else
        {
            return vec2(0, r2);
        }
    }
}

vec2 cylinderLineTest(vec3 V, vec3 v, float r, vec3 d)
{
    float t;

    /* Computing the normal of the plane containing the line and parallel
       to the cylinder axis. */
    vec3 n = normalize(cross(d, v));
    vec3 y = cross(n, d);

    /* Fast discard */
    float x = abs(dot(V, n));
    if (x > r)
        return vec2(-1, 0);

    /* The line intersects the cylinder once or twice. We return the nearest
       valid intersection. */
    float shift = cathetus(r, x);

    vec2 pV = vec2(dot(d, V), dot(y, V));
    vec2 pv = normalize(vec2(dot(d, v), dot(y, v)));
    vec2 pV1 = pV + shift * vec2(-pv.y, pv.x);
    vec2 pV2 = pV - shift * vec2(-pv.y, pv.x);
    float x1 = -1, x2 = -1;
    x1 = xAxisIntersection(pV1, pv);
    x2 = xAxisIntersection(pV2, pv);
    if (x1 > 0)
    {
        if (x2 > 0)
        {
            t = min(x1, x2);
        }
        else
            t = x1;
    }
    else
    {
        if (x2 > 0)
            t = x2;
        else
            return vec2(-1, 0);
    }
    return vec2(t, dot((d * t - V), v));
}

float cappedConeLineTest(vec3 d, vec3 c1, vec3 c2, float r1, float r2)
{
    vec3 v = c2 - c1;
    float len = length(v);
    v /= len;
    vec2 th;
    float high;
    float low;
    float alpha;
    if (abs(1 - r1 / r2) < 0.05)
    {
        /* Assuming equal radius */
        float r = max(r1, r2);
        low = len;
        high = 0;
        th = cylinderLineTest(c1, v, r, d);
        alpha = (th.y - high) / (low - high);
    }
    else
    {
        vec2 t1 = tangencyPoints(len, r1, r2);
        vec2 t2 = tangencyPoints(-len, r1, r2);
        if (t2.y < t1.y)
        {
            v = -v;
            float l = r2 * len / (r1 - r2);
            vec3 V = c2 - v * l;
            float cos_aperture = sqrt(1 - r2 * r2 / (l * l));
            high = l - t2.x;
            low = l + len - t1.x;
            th = coneLineTest(V, v, cos_aperture, d);
            alpha = 1 - (th.y - high) / (low - high);
        }
        else
        {
            float l = r1 * len / (r2 - r1);
            vec3 V = c1 - v * l;
            float cos_aperture = sqrt(1 - r1 * r1 / (l * l));
            high = l + t1.x;
            low = l + len + t2.x;
            th = coneLineTest(V, v, cos_aperture, d);
            alpha = (th.y - high) / (low - high);
        }
    }
    if (th.x == -1)
        return -1;

    if (th.y < high || th.y > low)
    {
        float a = sphereLineTest(c1, r1, d);
        float b = sphereLineTest(c2, r2, d);
        if (a != -1 && (b == -1 || a < b))
        {
            interpColor = color1;
            normal = (d * a) - c1;
#ifdef SMOOTH_TUBELETS
            normal -= dot(normal, planes[0].xyz) * planes[0].xyz;
#endif
            normal = normalize(normal);
            return a;
        }
        else if (b != -1)
        {
            interpColor = color2;
            normal = (d * b) - c2;
#ifdef SMOOTH_TUBELETS
            normal -= dot(normal, planes[1].xyz) * planes[1].xyz;
#endif
            normal = normalize(normal);
            return b;
        }
        else
        {
            return -1;
        }
    }
    else
    {
        interpColor = color1 * (1 - alpha) + color2 * alpha;
        normal = d * th.x - c1 * (1 - alpha) - c2 * alpha;
#ifdef SMOOTH_TUBELETS
        vec3 tangent = planes[0].xyz * (1 - alpha) + planes[1].xyz * alpha;
        normal -= dot(normal, tangent) * tangent;
#endif
        normal = normalize(normal);
        return th.x;
    }
}

vec4 phong(const vec4 color, const vec3 normal, const vec3 eye,
           const vec3 light);
vec4 getBaseColor(const vec4 inColor);
float getBaseAlpha(const float inAlpha);

float depthCache;

void solveIntersection()
{
    vec3 e = normalize(eye);
    float t = cappedConeLineTest(e, c1.xyz, c2.xyz, c1.w, c2.w);

#ifdef USE_ALPHA_BLENDING
    if (t == -1 || dot(vec4(e * t, 1), planes[0]) < 0 ||
        dot(vec4(e * t, 1), planes[1]) < 0)
        discard;
#else
    if (t == -1)
        discard;
#endif
    normalized_eye = e;
    depthCache = 0.5 * (1.0 - osg_ProjectionMatrix[2][2] -
                        osg_ProjectionMatrix[3][2] / (e.z * t));
}

vec4 shadeFragment()
{
    if (!solved)
        solveIntersection();

#ifdef ACCURATE_HEADLIGHT
    vec3 l = normalize(light);
#else
    vec3 l = vec3(0, 0, 1);
#endif
    vec4 _color = phong(getBaseColor(interpColor), normal, -normalized_eye, l);
#if defined USE_ALPHA_BLENDING
    _color.a = _color.a * (2 - _color.a);
#endif
    return _color;
}

float fragmentDepth()
{
    if (!solved)
        solveIntersection();
    return depthCache;
}

float fragmentAlpha()
{
    if (!solved)
        solveIntersection();
    float a = getBaseAlpha(interpColor.a);
#if defined USE_ALPHA_BLENDING
    a = a * (2 - a);
#endif
    return a;
}
