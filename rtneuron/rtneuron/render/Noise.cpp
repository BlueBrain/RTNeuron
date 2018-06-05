/************************************************************************
 *                                                                      *
 *                   Copyright (C) 2002  3Dlabs Inc. Ltd.               *
 *                                                                      *
 ************************************************************************/

#include <math.h>
#include <stdlib.h>

#include <osg/Image>
#include <osg/Texture3D>

/* Coherent noise function over 1, 2 or 3 dimensions */
/* (copyright Ken Perlin) */

#define MAXB 0x100
#define N 0x1000
#define NP 12 /* 2^N */
#define NM 0xfff

#define s_curve(t) (t * t * (3. - 2. * t))
#define lerp(t, a, b) (a + t * (b - a))
#define setup(i, b0, b1, r0, r1) \
    t = vec[i] + N;              \
    b0 = ((int)t) & BM;          \
    b1 = (b0 + 1) & BM;          \
    r0 = t - (int)t;             \
    r1 = r0 - 1.;
#define at2(rx, ry) (rx * q[0] + ry * q[1])
#define at3(rx, ry, rz) (rx * q[0] + ry * q[1] + rz * q[2])

void SetNoiseFrequency(int frequency);

// static double noise1(double arg);
// static double noise2(double vec[2]);
static double noise3(double vec[3]);
static void normalize2(double vec[2]);
static void normalize3(double vec[3]);

/*
   In what follows "alpha" is the weight when the sum is formed.
   Typically it is 2, As this approaches 1 the function is noisier.
   "beta" is the harmonic scaling/spacing, typically 2.
*/

static void initNoise(int seed);

static int p[MAXB + MAXB + 2];
static double g3[MAXB + MAXB + 2][3];
static double g2[MAXB + MAXB + 2][2];
static double g1[MAXB + MAXB + 2];

static int B;
static int BM;

static unsigned long next = 1;

/* From the Linux rand manual page */
/* RAND_MAX assumed to be 32767 */
int myrand()
{
    next = next * 1103515245 + 12345;
    return ((unsigned)(next / 65536) % 32768);
}

void mysrand(unsigned int seed)
{
    next = seed;
}

void SetNoiseFrequency(int frequency)
{
    B = frequency;
    BM = B - 1;
}

double noise3(double vec[3])
{
    int bx0, bx1, by0, by1, bz0, bz1, b00, b10, b01, b11;
    double rx0, rx1, ry0, ry1, rz0, rz1, *q, sy, sz, a, b, c, d, t, u, v;
    int i, j;

    setup(0, bx0, bx1, rx0, rx1);
    setup(1, by0, by1, ry0, ry1);
    setup(2, bz0, bz1, rz0, rz1);

    i = p[bx0];
    j = p[bx1];

    b00 = p[i + by0];
    b10 = p[j + by0];
    b01 = p[i + by1];
    b11 = p[j + by1];

    t = s_curve(rx0);
    sy = s_curve(ry0);
    sz = s_curve(rz0);

    q = g3[b00 + bz0];
    u = at3(rx0, ry0, rz0);
    q = g3[b10 + bz0];
    v = at3(rx1, ry0, rz0);
    a = lerp(t, u, v);

    q = g3[b01 + bz0];
    u = at3(rx0, ry1, rz0);
    q = g3[b11 + bz0];
    v = at3(rx1, ry1, rz0);
    b = lerp(t, u, v);

    c = lerp(sy, a, b);

    q = g3[b00 + bz1];
    u = at3(rx0, ry0, rz1);
    q = g3[b10 + bz1];
    v = at3(rx1, ry0, rz1);
    a = lerp(t, u, v);

    q = g3[b01 + bz1];
    u = at3(rx0, ry1, rz1);
    q = g3[b11 + bz1];
    v = at3(rx1, ry1, rz1);
    b = lerp(t, u, v);

    d = lerp(sy, a, b);

    return lerp(sz, c, d);
}

void normalize2(double v[2])
{
    double s;

    s = sqrt(v[0] * v[0] + v[1] * v[1]);
    v[0] = v[0] / s;
    v[1] = v[1] / s;
}

void normalize3(double v[3])
{
    double s;

    s = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    v[0] = v[0] / s;
    v[1] = v[1] / s;
    v[2] = v[2] / s;
}

void initNoise(int seed)
{
    int i, j;

    mysrand(seed);

    for (i = 0; i < B; i++)
    {
        p[i] = i;
        g1[i] = (double)((myrand() % (B + B)) - B) / B;

        for (j = 0; j < 2; j++)
            g2[i][j] = (double)((myrand() % (B + B)) - B) / B;
        normalize2(g2[i]);

        for (j = 0; j < 3; j++)
            g3[i][j] = (double)((myrand() % (B + B)) - B) / B;
        normalize3(g3[i]);
    }

    while (--i)
    {
        int k = p[i];
        p[i] = p[j = myrand() % B];
        p[j] = k;
    }

    for (i = 0; i < B + 2; i++)
    {
        p[B + i] = p[i];
        g1[B + i] = g1[i];
        for (j = 0; j < 2; j++)
            g2[B + i][j] = g2[i][j];
        for (j = 0; j < 3; j++)
            g3[B + i][j] = g3[i][j];
    }
}

osg::Image* make3DNoiseImage(int texSize, int seed)
{
    osg::Image* image = new osg::Image;
    image->setImage(texSize, texSize, texSize, 4, GL_RGBA, GL_UNSIGNED_BYTE,
                    new unsigned char[4 * texSize * texSize * texSize],
                    osg::Image::USE_NEW_DELETE);

    const int startFrequency = 4;
    const int numOctaves = 4;

    int f, i, j, inc;
    double ni[3];
    double incj, inck;
    int frequency = startFrequency;
    double amp = 0.5;

    for (f = 0, inc = 0; f < numOctaves; ++f, frequency *= 2, ++inc, amp *= 0.5)
    {
        SetNoiseFrequency(frequency);
        initNoise(seed);
        GLubyte* ptr = image->data();
        ni[0] = ni[1] = ni[2] = 0;

        const double inci = 1.0 / (texSize / frequency);
        for (i = 0; i < texSize; ++i, ni[0] += inci)
        {
            incj = 1.0 / (texSize / frequency);
            for (j = 0; j < texSize; ++j, ni[1] += incj)
            {
                inck = 1.0 / (texSize / frequency);
                for (int k = 0; k < texSize; ++k, ni[2] += inck, ptr += 4)
                {
                    *(ptr + inc) =
                        (GLubyte)(((noise3(ni) + 1.0) * amp) * 128.0);
                }
            }
        }
    }

    return image;
}

osg::Texture3D* make3DNoiseTexture(int texSize, int seed)
{
    osg::Texture3D* noiseTexture = new osg::Texture3D;
    noiseTexture->setFilter(osg::Texture3D::MIN_FILTER, osg::Texture3D::LINEAR);
    noiseTexture->setFilter(osg::Texture3D::MAG_FILTER, osg::Texture3D::LINEAR);
    noiseTexture->setWrap(osg::Texture3D::WRAP_S, osg::Texture3D::REPEAT);
    noiseTexture->setWrap(osg::Texture3D::WRAP_T, osg::Texture3D::REPEAT);
    noiseTexture->setWrap(osg::Texture3D::WRAP_R, osg::Texture3D::REPEAT);
    noiseTexture->setImage(make3DNoiseImage(texSize, seed));

    return noiseTexture;
}
