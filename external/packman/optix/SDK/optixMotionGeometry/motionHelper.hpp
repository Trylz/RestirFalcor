//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//


#pragma once

#include <memory>

namespace {

struct Matrix3x4
{
    // row-major matrix with 4 cols, 3 rows
    float m[12];

    inline static const Matrix3x4& Identity()
    {
        static Matrix3x4 m = { { 1.0, 0.0, 0.0, 0, 0.0, 1.0, 0.0, 0, 0.0, 0.0, 1.0, 0 } };
        return m;
    }
    inline static Matrix3x4 Scale( const float3& s )
    {
        Matrix3x4 m = { { s.x, 0.0, 0.0, 0, 0.0, s.y, 0.0, 0, 0.0, 0.0, s.z, 0 } };
        return m;
    }
    inline static Matrix3x4 Translation( const float3& t )
    {
        Matrix3x4 m = { { 1, 0.0, 0.0, t.x, 0.0, 1, 0.0, t.y, 0.0, 0.0, 1, t.z } };
        return m;
    }

    Matrix3x4 operator*( const Matrix3x4& b ) const
    {
        Matrix3x4 result;

        for( unsigned int r = 0; r < 3; ++r )
        {
            for( unsigned int c = 0; c < 4; ++c )
            {
                float sum = 0.0f;
                for( unsigned int k = 0; k < 3; ++k )
                {
                    float rk = this->m[r * 4 + k];
                    float kc = b.m[k * 4 + c];
                    sum += rk * kc;
                }
                if( c == 3 )
                    sum += this->m[r * 4 + c];
                result.m[r * 4 + c] = sum;
            }
        }
        return result;
    }
    float3 operator*( const float3& v ) const
    {
        float3 res;

        res.x = v.x * m[0] + v.y * m[1] + v.z * m[2] + m[3];
        res.y = v.x * m[4] + v.y * m[5] + v.z * m[6] + m[7];
        res.z = v.x * m[8] + v.y * m[9] + v.z * m[10] + m[11];

        return res;
    }

    // Returns the determinant of the matrix.
    float determinant() const
    {
        float d =
            m[0]*m[5]*m[10]*1-
            m[0]*m[5]*m[11]*0+m[0]*m[9]*0*m[7]-
            m[0]*m[9]*m[6]*1+m[0]*0*m[6]*m[11]-
            m[0]*0*m[10]*m[7]-m[4]*m[1]*m[10]*1+m[4]*m[1]*m[11]*0-
            m[4]*m[9]*0*m[3]+m[4]*m[9]*m[2]*1-
            m[4]*0*m[2]*m[11]+m[4]*0*m[10]*m[3]+m[8]*m[1]*m[6]*1-
            m[8]*m[1]*0*m[7]+m[8]*m[5]*0*m[3]-
            m[8]*m[5]*m[2]*1+m[8]*0*m[2]*m[7]-
            m[8]*0*m[6]*m[3]-
            0*m[1]*m[6]*m[11]+0*m[1]*m[10]*m[7]-
            0*m[5]*m[10]*m[3]+0*m[5]*m[2]*m[11]-
            0*m[9]*m[2]*m[7]+0*m[9]*m[6]*m[3];
        return d;
    }

    // Returns the inverse of the matrix.
    Matrix3x4 inverse() const
    {
        Matrix3x4 result;

        const float d = 1.0f / determinant();

        result.m[0]  = d * (m[5] * (m[10] * 1 - 0 * m[11]) + m[9] * (0 * m[7] - m[6] * 1) + 0 * (m[6] * m[11] - m[10] * m[7]));
        result.m[4]  = d * (m[6] * (m[8] * 1 - 0 * m[11]) + m[10] * (0 * m[7] - m[4] * 1) + 0 * (m[4] * m[11] - m[8] * m[7]));
        result.m[8]  = d * (m[7] * (m[8] * 0 - 0 * m[9]) + m[11] * (0 * m[5] - m[4] * 0) + 1 * (m[4] * m[9] - m[8] * m[5]));
        result.m[1]  = d * (m[9] * (m[2] * 1 - 0 * m[3]) + 0 * (m[10] * m[3] - m[2] * m[11]) + m[1] * (0 * m[11] - m[10] * 1));
        result.m[5]  = d * (m[10] * (m[0] * 1 - 0 * m[3]) + 0 * (m[8] * m[3] - m[0] * m[11]) + m[2] * (0 * m[11] - m[8] * 1));
        result.m[9]  = d * (m[11] * (m[0] * 0 - 0 * m[1]) + 1 * (m[8] * m[1] - m[0] * m[9]) + m[3] * (0 * m[9] - m[8] * 0));
        result.m[2]  = d * (0 * (m[2] * m[7] - m[6] * m[3]) + m[1] * (m[6] * 1 - 0 * m[7]) + m[5] * (0 * m[3] - m[2] * 1));
        result.m[6]  = d * (0 * (m[0] * m[7] - m[4] * m[3]) + m[2] * (m[4] * 1 - 0 * m[7]) + m[6] * (0 * m[3] - m[0] * 1));
        result.m[10] = d * (1 * (m[0] * m[5] - m[4] * m[1]) + m[3] * (m[4] * 0 - 0 * m[5]) + m[7] * (0 * m[1] - m[0] * 0));
        result.m[3]  = d * (m[1] * (m[10] * m[7] - m[6] * m[11]) + m[5] * (m[2] * m[11] - m[10] * m[3]) + m[9] * (m[6] * m[3] - m[2] * m[7]));
        result.m[7]  = d * (m[2] * (m[8] * m[7] - m[4] * m[11]) + m[6] * (m[0] * m[11] - m[8] * m[3]) + m[10] * (m[4] * m[3] - m[0] * m[7]));
        result.m[11] = d * (m[3] * (m[8] * m[5] - m[4] * m[9]) + m[7] * (m[0] * m[9] - m[8] * m[1]) + m[11] * (m[4] * m[1] - m[0] * m[5]));

        return result;
    }
};


class Quaternion
{
public:
    Quaternion();
    Quaternion( float x, float y, float z, float w );
    Quaternion( const float3& axis, double angle );

    Quaternion& operator*=( const Quaternion& q1 );

    /** quaternion x, y, z, w */
    float4 m_q;
};
Quaternion::Quaternion()
{
    m_q = make_float4( 0.0f, 0.0f, 0.0f, 1.0f );
}
/*
Quaternion::Quaternion( float x, float y, float z, float w )
{
    m_q = make_float4( x, y, z, w );
}
*/
Quaternion::Quaternion( const float3& axis, double angle )
{
    const float3 naxis  = normalize( axis );
    const double radian = angle * ( M_PI / 180 );
    const float  s      = (float)sin( radian / 2 );
    m_q.x               = naxis.x * s;
    m_q.y               = naxis.y * s;
    m_q.z               = naxis.z * s;
    m_q.w               = (float)cos( radian / 2 );
}
/*
Quaternion& Quaternion::operator*=( const Quaternion& q1 )
{
    m_q = make_float4( m_q.w * q1.m_q.x + m_q.x * q1.m_q.w + m_q.y * q1.m_q.z - m_q.z * q1.m_q.y,
        m_q.w * q1.m_q.y + m_q.y * q1.m_q.w + m_q.z * q1.m_q.x - m_q.x * q1.m_q.z,
        m_q.w * q1.m_q.z + m_q.z * q1.m_q.w + m_q.x * q1.m_q.y - m_q.y * q1.m_q.x,
        m_q.w * q1.m_q.w - m_q.x * q1.m_q.x - m_q.y * q1.m_q.y - m_q.z * q1.m_q.z );

    return *this;
}
*/
static inline Quaternion nlerp( const Quaternion& quat0, const Quaternion& quat1, float t )
{
    Quaternion q;
    q.m_q = lerp( quat0.m_q, quat1.m_q, t );
    q.m_q = normalize( q.m_q );
    return q;
}


OptixSRTData lerp( const OptixSRTData& a, const OptixSRTData& b, float t )
{
    OptixSRTData r;
    r.sx  = ::lerp( a.sx, b.sx, t );
    r.a   = ::lerp( a.a, b.a, t );
    r.b   = ::lerp( a.b, b.b, t );
    r.pvx = ::lerp( a.pvx, b.pvx, t );
    r.sy  = ::lerp( a.sy, b.sy, t );
    r.c   = ::lerp( a.c, b.c, t );
    r.pvy = ::lerp( a.pvy, b.pvy, t );
    r.sz  = ::lerp( a.sz, b.sz, t );
    r.pvz = ::lerp( a.pvz, b.pvz, t );
    r.qx  = ::lerp( a.qx, b.qx, t );
    r.qy  = ::lerp( a.qy, b.qy, t );
    r.qz  = ::lerp( a.qz, b.qz, t );
    r.qw  = ::lerp( a.qw, b.qw, t );
    r.tx  = ::lerp( a.tx, b.tx, t );
    r.ty  = ::lerp( a.ty, b.ty, t );
    r.tz  = ::lerp( a.tz, b.tz, t );

    const float inv_qLength = 1.f / sqrtf( r.qx * r.qx + r.qy * r.qy + r.qz * r.qz + r.qw * r.qw );
    r.qx *= inv_qLength;
    r.qy *= inv_qLength;
    r.qz *= inv_qLength;
    r.qw *= inv_qLength;

    return r;
}

void srtToMatrix( const OptixSRTData& srt, float* m )
{
    const float4 q = make_float4( srt.qx, srt.qy, srt.qz, srt.qw );

    // q is assumed to be normalized, but to be sure, normalize again
    const float  inv_sql = 1.f / ( srt.qx * srt.qx + srt.qy * srt.qy + srt.qz * srt.qz + srt.qw * srt.qw );
    const float4 nq      = make_float4( q.x * inv_sql, q.y * inv_sql, q.z * inv_sql, q.w * inv_sql );

    const float sqw = q.w * nq.w;
    const float sqx = q.x * nq.x;
    const float sqy = q.y * nq.y;
    const float sqz = q.z * nq.z;

    const float xy = q.x * nq.y;
    const float zw = q.z * nq.w;
    const float xz = q.x * nq.z;
    const float yw = q.y * nq.w;
    const float yz = q.y * nq.z;
    const float xw = q.x * nq.w;

    m[0] = ( sqx - sqy - sqz + sqw );
    m[1] = 2.0f * ( xy - zw );
    m[2] = 2.0f * ( xz + yw );

    m[4] = 2.0f * ( xy + zw );
    m[5] = ( -sqx + sqy - sqz + sqw );
    m[6] = 2.0f * ( yz - xw );

    m[8]  = 2.0f * ( xz - yw );
    m[9]  = 2.0f * ( yz + xw );
    m[10] = ( -sqx - sqy + sqz + sqw );

    m[3]  = m[0] * srt.pvx + m[1] * srt.pvy + m[2] * srt.pvz + srt.tx;
    m[7]  = m[4] * srt.pvx + m[5] * srt.pvy + m[6] * srt.pvz + srt.ty;
    m[11] = m[8] * srt.pvx + m[9] * srt.pvy + m[10] * srt.pvz + srt.tz;

    m[2]  = m[0] * srt.b + m[1] * srt.c + m[2] * srt.sz;
    m[6]  = m[4] * srt.b + m[5] * srt.c + m[6] * srt.sz;
    m[10] = m[8] * srt.b + m[9] * srt.c + m[10] * srt.sz;

    m[1] = m[0] * srt.a + m[1] * srt.sy;
    m[5] = m[4] * srt.a + m[5] * srt.sy;
    m[9] = m[8] * srt.a + m[9] * srt.sy;

    m[0] = m[0] * srt.sx;
    m[4] = m[4] * srt.sx;
    m[8] = m[8] * srt.sx;
}

template <unsigned int motionKeys>
struct alignas( 64 ) MatrixMotionTransformFixedSize : OptixMatrixMotionTransform
{
    // must be strictly after OptixMatrixMotionTransform::transform
    float additionalTransforms[motionKeys - 2][12];

    MatrixMotionTransformFixedSize()
    {
        //static_assert(sizeof(MatrixMotionTransform<motionKeys>) == sizeof(OptixMatrixMotionTransform)+(motionKeys-2)*12*sizeof(float), "size/alignment error");
        motionOptions.numKeys = motionKeys;
    }
    float* motionKey( unsigned int key ) { return transform[key]; }
};

template <>
struct alignas( 64 ) MatrixMotionTransformFixedSize<2> : OptixMatrixMotionTransform
{
    MatrixMotionTransformFixedSize() { motionOptions.numKeys = 2; }
    float* motionKey( unsigned int key ) { return transform[key]; }
};

template <class Derived, typename OptixDataType>
class MotionTransformArrayBase
{
  public:
    MotionTransformArrayBase( size_t numTransforms, unsigned int numKeys )
        : m_numTransforms( numTransforms )
        , m_numKeys( numKeys )
    {
        if( numKeys < 2 )
            numKeys = 2;
        if( numTransforms )
            m_data = std::unique_ptr<char[]>( new char[numTransforms * Derived::byteSizePerTransform( numKeys )] );
    }
    MotionTransformArrayBase( const MotionTransformArrayBase& other )
        : MotionTransformArrayBase( other.m_numTransforms, other.m_numKeys )
    {
        memcpy( data(), other.data(), byteSize() );
    }

    MotionTransformArrayBase( MotionTransformArrayBase&& other ) noexcept
        : MotionTransformArrayBase( 0, 0 )
    {
        swap( *this, other );
    }

    friend void swap( MotionTransformArrayBase& a, MotionTransformArrayBase& b )
    {
        using std::swap;
        swap( a.m_numTransforms, b.m_numTransforms );
        swap( a.m_numKeys, b.m_numKeys );
        swap( a.m_data, b.m_data );
    }

    MotionTransformArrayBase& operator=( MotionTransformArrayBase other )
    {
        swap( *this, other );
        return *this;
    }
    MotionTransformArrayBase& operator=( MotionTransformArrayBase&& other )
    {
        swap( *this, other );
        return *this;
    }

    OptixDataType& transform( size_t transformIdx )
    {
        return *(OptixDataType*)( (char*)m_data.get() + Derived::byteSizePerTransform( m_numKeys ) * transformIdx );
    }
    const OptixDataType& transform( size_t transformIdx ) const
    {
        return *(OptixDataType*)( (char*)m_data.get() + Derived::byteSizePerTransform( m_numKeys ) * transformIdx );
    }

    void*       data() { return m_data.get(); }
    const void* data() const { return m_data.get(); }

    unsigned int numKeys() const { return m_numKeys; }

    size_t numTransforms() const { return m_numTransforms; }

    size_t byteSizePerTransform() const { return Derived::byteSizePerTransform( m_numKeys ); }
    size_t byteSize() const { return m_numTransforms * Derived::byteSizePerTransform( m_numKeys ); }

  protected:
    static size_t roundUp64( size_t i ) { return ( ( i + 64 - 1 ) / 64 ) * 64; }

  private:
    size_t                  m_numTransforms = 0;
    unsigned int            m_numKeys       = 2;
    std::unique_ptr<char[]> m_data;
};


class MatrixMotionTransformArray : public MotionTransformArrayBase<MatrixMotionTransformArray, OptixMatrixMotionTransform>
{
  public:
    typedef MotionTransformArrayBase<MatrixMotionTransformArray, OptixMatrixMotionTransform> Base;

    MatrixMotionTransformArray( size_t numTransforms = 0, unsigned int numKeys = 2 )
        : Base( numTransforms, numKeys )
    {
    }

    float*       motionData( size_t transformIdx ) { return &transform( transformIdx ).transform[0][0]; }
    const float* motionData( size_t transformIdx ) const { return &transform( transformIdx ).transform[0][0]; }
    float* motionData( size_t transformIdx, unsigned int key ) { return &transform( transformIdx ).transform[key][0]; }
    const float* motionData( size_t transformIdx, unsigned int key ) const
    {
        return &transform( transformIdx ).transform[key][0];
    }

    using Base::byteSizePerTransform;
    static size_t byteSizePerTransform( unsigned int numKeys )
    {
        // pad to 64 bytes to ensure 64 byte alignment when using byteSize() to compute size for array of motion transforms with N keys
        return roundUp64( sizeof( OptixMatrixMotionTransform ) + sizeof( float ) * 12 * ( numKeys - 2 ) );
    }
};

class SRTMotionTransformArray : public MotionTransformArrayBase<SRTMotionTransformArray, OptixSRTMotionTransform>
{
  public:
    typedef MotionTransformArrayBase<SRTMotionTransformArray, OptixSRTMotionTransform> Base;

    SRTMotionTransformArray( size_t numTransforms = 0, unsigned int numKeys = 2 )
        : Base( numTransforms, numKeys )
    {
    }

    OptixSRTData*       motionData( size_t transformIdx ) { return transform( transformIdx ).srtData; }
    const OptixSRTData* motionData( size_t transformIdx ) const { return transform( transformIdx ).srtData; }
    OptixSRTData& motionData( size_t transformIdx, unsigned int key ) { return transform( transformIdx ).srtData[key]; }
    const OptixSRTData& motionData( size_t transformIdx, unsigned int key ) const
    {
        return transform( transformIdx ).srtData[key];
    }

    using Base::byteSizePerTransform;
    static size_t byteSizePerTransform( unsigned int numKeys )
    {
        // pad to 64 bytes to ensure 64 byte alignment when using byteSize() to compute size for array of motion transforms with N keys
        return roundUp64( sizeof( OptixSRTMotionTransform ) + sizeof( OptixSRTData ) * ( numKeys - 2 ) );
    }
};

class MatrixMotionTransform : public MatrixMotionTransformArray
{
  public:
    typedef MatrixMotionTransformArray Base;

    MatrixMotionTransform( unsigned int numKeys = 2 )
        : MatrixMotionTransformArray( 1, numKeys )
    {
    }
    float*       motionData( unsigned int key ) { return Base::motionData( 0, key ); }
    const float* motionData( unsigned int key ) const { return Base::motionData( 0, key ); }

  private:
    using Base::numTransforms;
};


OptixSRTData buildSRT( const float3&     scale,
                       const float3&     shear,
                       const float3&     scaleShearPivot,
                       const Quaternion& q,
                       const float3&     rotationPivot,
                       const float3&     translation )
{
    // Note that a pivot is a point and to do a transformation wrt. a pivot point, we need to apply an inverse translation (hence -pivot) before the transformation
    // to go back to 'world' space, the inverse of the pivot translation is applied (+pivot)

    // multiply scale and shear with the scaleShearPivot to bake the pivot into the S transformation, like: S * (p - p') = S*p - S*p'
    float3 rotationPivotScalePivot = {
        scale.x * -scaleShearPivot.x + shear.x * -scaleShearPivot.y + shear.y * -scaleShearPivot.z,
        scale.y * -scaleShearPivot.y + shear.z * -scaleShearPivot.z, scale.z * -scaleShearPivot.z };
    // undo scale pivot after applying scale transformation
    rotationPivotScalePivot += scaleShearPivot;
    // apply pivot for rotation
    // SRT definition actually wants the pivot point, instead of the transformation for the pivot point
    // hence, we need to add the pivot point instead of subtracting it
    rotationPivotScalePivot -= rotationPivot;

    // apply translation and undo rotation pivot
    float3 translationM1RotationPivot = translation + rotationPivot;

    return { scale.x,
             shear.x,
             shear.y,
             rotationPivotScalePivot.x,
             scale.y,
             shear.z,
             rotationPivotScalePivot.y,
             scale.z,
             rotationPivotScalePivot.z,
             q.m_q.x,
             q.m_q.y,
             q.m_q.z,
             q.m_q.w,
             translationM1RotationPivot.x,
             translationM1RotationPivot.y,
             translationM1RotationPivot.z };
}

OptixSRTData buildSRT( const float3& scale, const Quaternion& q, const float3& translation )
{
    return buildSRT( scale, make_float3( 0.0f ), make_float3( 0.0f ), q, make_float3( 0.0f ), translation );
}

}  // namespace