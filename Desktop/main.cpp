#include <iostream>
#include <string>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include "BMP.h"

using namespace cv;
using namespace std;

// parameters
const std::string img_path = "../../Dataset/";

float tau            = 0.25;
float lambda         = 0.15;
float theta          = 0.3;
int nscales        = 5;
int warps          = 5;
float epsilon        = 0.01;
int iterations     = 300;
bool useInitialFlow = false;

// buffers
std::vector<Mat_<float> > I0s;
std::vector<Mat_<float> > I1s;
std::vector<Mat_<float> > u1s;
std::vector<Mat_<float> > u2s;

Mat_<float> I1x_buf;
Mat_<float> I1y_buf;

Mat_<float> flowMap1_buf;
Mat_<float> flowMap2_buf;

Mat_<float> I1w_buf;
Mat_<float> I1wx_buf;
Mat_<float> I1wy_buf;

Mat_<float> grad_buf;
Mat_<float> rho_c_buf;

Mat_<float> v1_buf;
Mat_<float> v2_buf;

Mat_<float> p11_buf;
Mat_<float> p12_buf;
Mat_<float> p21_buf;
Mat_<float> p22_buf;

Mat_<float> div_p1_buf;
Mat_<float> div_p2_buf;

Mat_<float> u1x_buf;
Mat_<float> u1y_buf;
Mat_<float> u2x_buf;
Mat_<float> u2y_buf;

// buildFlowMap

void buildFlowMap(const Mat_<float>& u1, const Mat_<float>& u2, Mat_<float>& map1, Mat_<float>& map2)
{
    CV_DbgAssert( u2.size() == u1.size() );
    CV_DbgAssert( map1.size() == u1.size() );
    CV_DbgAssert( map2.size() == u1.size() );

    for (int y = 0; y < u1.rows; y ++)
    {
        const float* u1Row = u1[y];
        const float* u2Row = u2[y];

        float* map1Row = map1[y];
        float* map2Row = map2[y];

        for (int x = 0; x < u1.cols; ++x)
        {
            map1Row[x] = x + u1Row[x];
            map2Row[x] = y + u2Row[x];
        }
    }
}

////////////////////////////////////////////////////////////
// centeredGradient


void centeredGradient(const Mat_<float>& src, Mat_<float>& dx, Mat_<float>& dy)
{
    CV_DbgAssert( src.rows > 2 && src.cols > 2 );
    CV_DbgAssert( dx.size() == src.size() );
    CV_DbgAssert( dy.size() == src.size() );

    const int last_row = src.rows - 1;
    const int last_col = src.cols - 1;

    // compute the gradient on the center body of the image

    for (int y = 1; y < last_row; y++)
    {
        const float* srcPrevRow = src[y - 1];
        const float* srcCurRow = src[y];
        const float* srcNextRow = src[y + 1];

        float* dxRow = dx[y];
        float* dyRow = dy[y];

        for (int x = 1; x < last_col; ++x)
        {
            dxRow[x] = (srcCurRow[x + 1] - srcCurRow[x - 1]) / 2;
            dyRow[x] = (srcNextRow[x] - srcPrevRow[x]) / 2;
        }
    }

    // compute the gradient on the first and last rows
    for (int x = 1; x < last_col; ++x)
    {
        dx(0, x) = (src(0, x + 1) - src(0, x - 1)) / 2;
        dy(0, x) = (src(1, x) - src(0, x)) / 2;

        dx(last_row, x) = (src(last_row, x + 1) - src(last_row, x - 1)) / 2;
        dy(last_row, x) = (src(last_row, x) - src(last_row - 1, x)) / 2;
    }

    // compute the gradient on the first and last columns
    for (int y = 1; y < last_row; ++y)
    {
        dx(y, 0) = (src(y, 1) - src(y, 0)) / 2;
        dy(y, 0) = (src(y + 1, 0) - src(y - 1, 0)) / 2;

        dx(y, last_col) = (src(y, last_col) - src(y, last_col - 1)) / 2;
        dy(y, last_col) = (src(y + 1, last_col) - src(y - 1, last_col)) / 2;
    }

    // compute the gradient at the four corners
    dx(0, 0) = (src(0, 1) - src(0, 0)) / 2;
    dy(0, 0) = (src(1, 0) - src(0, 0)) / 2;

    dx(0, last_col) = (src(0, last_col) - src(0, last_col - 1)) / 2;
    dy(0, last_col) = (src(1, last_col) - src(0, last_col)) / 2;

    dx(last_row, 0) = (src(last_row, 1) - src(last_row, 0)) / 2;
    dy(last_row, 0) = (src(last_row, 0) - src(last_row - 1, 0)) / 2;

    dx(last_row, last_col) = (src(last_row, last_col) - src(last_row, last_col - 1)) / 2;
    dy(last_row, last_col) = (src(last_row, last_col) - src(last_row - 1, last_col)) / 2;
}

////////////////////////////////////////////////////////////
// forwardGradient


void forwardGradient(const Mat_<float>& src, Mat_<float>& dx, Mat_<float>& dy)
{
    CV_DbgAssert( src.rows > 2 && src.cols > 2 );
    CV_DbgAssert( dx.size() == src.size() );
    CV_DbgAssert( dy.size() == src.size() );

    const int last_row = src.rows - 1;
    const int last_col = src.cols - 1;

    // compute the gradient on the central body of the image

    for (int y = 0; y < last_row; y++)
    {
        const float* srcCurRow = src[y];
        const float* srcNextRow = src[y + 1];

        float* dxRow = dx[y];
        float* dyRow = dy[y];

        for (int x = 0; x < last_col; ++x)
        {
            dxRow[x] = srcCurRow[x + 1] - srcCurRow[x];
            dyRow[x] = srcNextRow[x] - srcCurRow[x];
        }
    }

    // compute the gradient on the last row
    for (int x = 0; x < last_col; ++x)
    {
        dx(last_row, x) = src(last_row, x + 1) - src(last_row, x);
        dy(last_row, x) = 0.0f;
    }

    // compute the gradient on the last column
    for (int y = 0; y < last_row; ++y)
    {
        dx(y, last_col) = 0.0f;
        dy(y, last_col) = src(y + 1, last_col) - src(y, last_col);
    }

    dx(last_row, last_col) = 0.0f;
    dy(last_row, last_col) = 0.0f;
}

////////////////////////////////////////////////////////////
// divergence

void divergence(const Mat_<float>& v1, const Mat_<float>& v2, Mat_<float>& div)
{
    CV_DbgAssert( v1.rows > 2 && v1.cols > 2 );
    CV_DbgAssert( v2.size() == v1.size() );
    CV_DbgAssert( div.size() == v1.size() );

    for (int y = 1; y < v1.rows; y++)
    {
        const float* v1Row = v1[y];
        const float* v2PrevRow = v2[y - 1];
        const float* v2CurRow = v2[y];

        float* divRow = div[y];

        for(int x = 1; x < v1.cols; ++x)
        {
            const float v1x = v1Row[x] - v1Row[x - 1];
            const float v2y = v2CurRow[x] - v2PrevRow[x];

            divRow[x] = v1x + v2y;
        }
    }

}

////////////////////////////////////////////////////////////
// calcGradRho

void calcGradRho(const Mat_<float>& I0, const Mat_<float>& I1w, const Mat_<float>& I1wx, const Mat_<float>& I1wy, const Mat_<float>& u1, const Mat_<float>& u2,
                 Mat_<float>& grad, Mat_<float>& rho_c)
{
    CV_DbgAssert( I1w.size() == I0.size() );
    CV_DbgAssert( I1wx.size() == I0.size() );
    CV_DbgAssert( I1wy.size() == I0.size() );
    CV_DbgAssert( u1.size() == I0.size() );
    CV_DbgAssert( u2.size() == I0.size() );
    CV_DbgAssert( grad.size() == I0.size() );
    CV_DbgAssert( rho_c.size() == I0.size() );

    for (int y = 0; y < I0.rows; y++)
    {
        const float* I0Row = I0[y];
        const float* I1wRow = I1w[y];
        const float* I1wxRow = I1wx[y];
        const float* I1wyRow = I1wy[y];
        const float* u1Row = u1[y];
        const float* u2Row = u2[y];

        float* gradRow = grad[y];
        float* rhoRow = rho_c[y];

        for (int x = 0; x < I0.cols; ++x)
        {
            const float Ix2 = I1wxRow[x] * I1wxRow[x];
            const float Iy2 = I1wyRow[x] * I1wyRow[x];

            // store the |Grad(I1)|^2
            gradRow[x] = Ix2 + Iy2;

            // compute the constant part of the rho function
            rhoRow[x] = (I1wRow[x] - I1wxRow[x] * u1Row[x] - I1wyRow[x] * u2Row[x] - I0Row[x]);
        }
    }
}

////////////////////////////////////////////////////////////
// estimateV

void estimateV(const Mat_<float>& I1wx, const Mat_<float>& I1wy, const Mat_<float>& u1, const Mat_<float>& u2,
               const Mat_<float>& grad, const Mat_<float>& rho_c, Mat_<float>& v1, Mat_<float>& v2, float l_t)
{
    CV_DbgAssert( I1wy.size() == I1wx.size() );
    CV_DbgAssert( u1.size() == I1wx.size() );
    CV_DbgAssert( u2.size() == I1wx.size() );
    CV_DbgAssert( grad.size() == I1wx.size() );
    CV_DbgAssert( rho_c.size() == I1wx.size() );
    CV_DbgAssert( v1.size() == I1wx.size() );
    CV_DbgAssert( v2.size() == I1wx.size() );

    for (int y = 0; y < I1wx.rows; y++)
    {
        const float* I1wxRow = I1wx[y];
        const float* I1wyRow = I1wy[y];
        const float* u1Row = u1[y];
        const float* u2Row = u2[y];
        const float* gradRow = grad[y];
        const float* rhoRow = rho_c[y];

        float* v1Row = v1[y];
        float* v2Row = v2[y];

        for (int x = 0; x < I1wx.cols; ++x)
        {
            const float rho = rhoRow[x] + (I1wxRow[x] * u1Row[x] + I1wyRow[x] * u2Row[x]);

            float d1 = 0.0f;
            float d2 = 0.0f;

            if (rho < -l_t * gradRow[x])
            {
                d1 = l_t * I1wxRow[x];
                d2 = l_t * I1wyRow[x];
            }
            else if (rho > l_t * gradRow[x])
            {
                d1 = -l_t * I1wxRow[x];
                d2 = -l_t * I1wyRow[x];
            }
            else
            {
                float fi = -rho / gradRow[x];
                d1 = fi * I1wxRow[x];
                d2 = fi * I1wyRow[x];
            }

            v1Row[x] = u1Row[x] + d1;
            v2Row[x] = u2Row[x] + d2;
        }
    }
}

////////////////////////////////////////////////////////////
// estimateU

float estimateU(const Mat_<float>& v1, const Mat_<float>& v2, const Mat_<float>& div_p1, const Mat_<float>& div_p2, Mat_<float>& u1, Mat_<float>& u2, float theta)
{
    CV_DbgAssert( v2.size() == v1.size() );
    CV_DbgAssert( div_p1.size() == v1.size() );
    CV_DbgAssert( div_p2.size() == v1.size() );
    CV_DbgAssert( u1.size() == v1.size() );
    CV_DbgAssert( u2.size() == v1.size() );

    float error = 0.0f;
    for (int y = 0; y < v1.rows; ++y)
    {
        const float* v1Row = v1[y];
        const float* v2Row = v2[y];
        const float* divP1Row = div_p1[y];
        const float* divP2Row = div_p2[y];

        float* u1Row = u1[y];
        float* u2Row = u2[y];

        for (int x = 0; x < v1.cols; ++x)
        {
            const float u1k = u1Row[x];
            const float u2k = u2Row[x];

            u1Row[x] = v1Row[x] + theta * divP1Row[x];
            u2Row[x] = v2Row[x] + theta * divP2Row[x];

            error += (u1Row[x] - u1k) * (u1Row[x] - u1k) + (u2Row[x] - u2k) * (u2Row[x] - u2k);
        }
    }

    return error;
}

////////////////////////////////////////////////////////////
// estimateDualVariables

void estimateDualVariables(const Mat_<float>& u1x, const Mat_<float>& u1y, const Mat_<float>& u2x, const Mat_<float>& u2y,
                           Mat_<float>& p11, Mat_<float>& p12, Mat_<float>& p21, Mat_<float>& p22, float taut)
{
    CV_DbgAssert( u1y.size() == u1x.size() );
    CV_DbgAssert( u2x.size() == u1x.size() );
    CV_DbgAssert( u2y.size() == u1x.size() );
    CV_DbgAssert( p11.size() == u1x.size() );
    CV_DbgAssert( p12.size() == u1x.size() );
    CV_DbgAssert( p21.size() == u1x.size() );
    CV_DbgAssert( p22.size() == u1x.size() );

    for (int y = 0; y < u1x.rows; y++)
    {
        const float* u1xRow = u1x[y];
        const float* u1yRow = u1y[y];
        const float* u2xRow = u2x[y];
        const float* u2yRow = u2y[y];

        float* p11Row = p11[y];
        float* p12Row = p12[y];
        float* p21Row = p21[y];
        float* p22Row = p22[y];

        for (int x = 0; x < u1x.cols; ++x)
        {
            const float g1 = static_cast<float>(hypot(u1xRow[x], u1yRow[x]));
            const float g2 = static_cast<float>(hypot(u2xRow[x], u2yRow[x]));

            const float ng1  = 1.0f + taut * g1;
            const float ng2  = 1.0f + taut * g2;

            p11Row[x] = (p11Row[x] + taut * u1xRow[x]) / ng1;
            p12Row[x] = (p12Row[x] + taut * u1yRow[x]) / ng1;
            p21Row[x] = (p21Row[x] + taut * u2xRow[x]) / ng2;
            p22Row[x] = (p22Row[x] + taut * u2yRow[x]) / ng2;
        }
    }

}

template<typename T>
void my_remap(const Mat_<T>& src, Mat_<T>& dst, const Mat_<float> map1, const Mat_<float> map2)
{
    Mat_<float> sum;
    Mat_<float> value;

    sum.create(src.size());
    value.create(src.size());

    sum.setTo(Scalar::all(0));
    value.setTo(Scalar::all(0));

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++)
        {
            int dst_x = map1(y, x);
            int dst_y = map2(y, x);

            for (int i = -4; i <= 4; i++)
                for (int j = -4; j <= 4; j++)
                {
                    if (dst_x + i >= 0 && dst_x + i < src.cols && dst_y + j >= 0 && dst_y + j < src.rows)
                    {
                        value(dst_y + j, dst_x + i) += 1.0 / (0.03 + i*i + j*j);
                        sum(dst_y + j, dst_x + i) += (float)src(y, x) / (0.03 + i*i + j*j);
                    }
                }

//            if (dst_x >= 0 && dst_x < src.cols && dst_y >= 0 && dst_y < src.rows)
//                dst(dst_y, dst_x) = src(y, x);
        }
    }

    for (int y = 0; y < src.rows; y++)
    {
        for (int x = 0; x < src.cols; x++) {
            if (value(y, x) > 1e-5)
                dst(y, x) = (float) (sum(y, x) / value(y, x));
            else
                dst(y, x) = 0;
//            printf("%d ", dst(y, x));
        }
//        printf("\n");
    }
}

void procOneScale(const Mat_<float>& I0, const Mat_<float>& I1, Mat_<float>& u1, Mat_<float>& u2)
{

    const float scaledEpsilon = static_cast<float>(epsilon * epsilon * I0.size().area());

    CV_DbgAssert( I1.size() == I0.size() );
    CV_DbgAssert( I1.type() == I0.type() );
    CV_DbgAssert( u1.size() == I0.size() );
    CV_DbgAssert( u2.size() == u1.size() );

    Mat_<float> I1x = I1x_buf(Rect(0, 0, I0.cols, I0.rows));
    Mat_<float> I1y = I1y_buf(Rect(0, 0, I0.cols, I0.rows));
    centeredGradient(I1, I1x, I1y);

    Mat_<float> flowMap1 = flowMap1_buf(Rect(0, 0, I0.cols, I0.rows));
    Mat_<float> flowMap2 = flowMap2_buf(Rect(0, 0, I0.cols, I0.rows));

    Mat_<float> I1w = I1w_buf(Rect(0, 0, I0.cols, I0.rows));
    Mat_<float> I1wx = I1wx_buf(Rect(0, 0, I0.cols, I0.rows));
    Mat_<float> I1wy = I1wy_buf(Rect(0, 0, I0.cols, I0.rows));

    Mat_<float> grad = grad_buf(Rect(0, 0, I0.cols, I0.rows));
    Mat_<float> rho_c = rho_c_buf(Rect(0, 0, I0.cols, I0.rows));

    Mat_<float> v1 = v1_buf(Rect(0, 0, I0.cols, I0.rows));
    Mat_<float> v2 = v2_buf(Rect(0, 0, I0.cols, I0.rows));

    Mat_<float> p11 = p11_buf(Rect(0, 0, I0.cols, I0.rows));
    Mat_<float> p12 = p12_buf(Rect(0, 0, I0.cols, I0.rows));
    Mat_<float> p21 = p21_buf(Rect(0, 0, I0.cols, I0.rows));
    Mat_<float> p22 = p22_buf(Rect(0, 0, I0.cols, I0.rows));
    p11.setTo(Scalar::all(0));
    p12.setTo(Scalar::all(0));
    p21.setTo(Scalar::all(0));
    p22.setTo(Scalar::all(0));

    Mat_<float> div_p1 = div_p1_buf(Rect(0, 0, I0.cols, I0.rows));
    Mat_<float> div_p2 = div_p2_buf(Rect(0, 0, I0.cols, I0.rows));

    Mat_<float> u1x = u1x_buf(Rect(0, 0, I0.cols, I0.rows));
    Mat_<float> u1y = u1y_buf(Rect(0, 0, I0.cols, I0.rows));
    Mat_<float> u2x = u2x_buf(Rect(0, 0, I0.cols, I0.rows));
    Mat_<float> u2y = u2y_buf(Rect(0, 0, I0.cols, I0.rows));

    const float l_t = static_cast<float>(lambda * theta);
    const float taut = static_cast<float>(tau / theta);

    int block_x = min(640, I0.cols);
    int block_y = min(480, I0.rows);

    for (int warpings = 0; warpings < warps; ++warpings)
    {
        // compute the warping of the target image and its derivatives
        buildFlowMap(u1, u2, flowMap1, flowMap2);
//        my_remap<float>(I1, I1w, flowMap1, flowMap2);
//        my_remap<float>(I1x, I1wx, flowMap1, flowMap2);
//        my_remap<float>(I1y, I1wy, flowMap1, flowMap2);
//        Mat flowMap1_, flowMap2_;
//        flowMap1.convertTo(flowMap1_, CV_32F);
//        flowMap2.convertTo(flowMap2_, CV_32F)

//        printf("before remap ...");
//        cout << flowMap1 << endl;
        remap(I1, I1w, flowMap1, flowMap2, INTER_CUBIC);
        remap(I1x, I1wx, flowMap1, flowMap2, INTER_CUBIC);
        remap(I1y, I1wy, flowMap1, flowMap2, INTER_CUBIC);

//        printf("warpings = %d\n", warpings);
//        cv::imshow("I1", I1);
//        cv::imshow("I1w", I1w);
//        cv::waitKey(0);

        // rho = I1(x + u) - I1(x + u) * u - I0(x)
        calcGradRho(I0, I1w, I1wx, I1wy, u1, u2, grad, rho_c);

//        for (int sy = 0; sy < I0.rows; sy += block_y){
//            for (int sx = 0; sx < I0.cols; sx += block_x) {
//                float error = 1e7;
//
//                Mat_<float> I1wx_ = I1wx(Rect(sx, sy, block_x, block_y));
//                Mat_<float> I1wy_ = I1wy(Rect(sx, sy, block_x, block_y));
//                Mat_<float> u1_ = u1(Rect(sx, sy, block_x, block_y));
//                Mat_<float> u2_ = u2(Rect(sx, sy, block_x, block_y));
//                Mat_<float> grad_ = grad(Rect(sx, sy, block_x, block_y));
//                Mat_<float> rho_c_ = rho_c(Rect(sx, sy, block_x, block_y));
//                Mat_<float> v1_ = v1(Rect(sx, sy, block_x, block_y));
//                Mat_<float> v2_ = v2(Rect(sx, sy, block_x, block_y));
//                Mat_<float> p11_ = p11(Rect(sx, sy, block_x, block_y));
//                Mat_<float> p12_ = p12(Rect(sx, sy, block_x, block_y));
//                Mat_<float> p21_ = p21(Rect(sx, sy, block_x, block_y));
//                Mat_<float> p22_ = p22(Rect(sx, sy, block_x, block_y));
//                Mat_<float> div_p1_ = div_p1(Rect(sx, sy, block_x, block_y));
//                Mat_<float> div_p2_ = div_p2(Rect(sx, sy, block_x, block_y));
//                Mat_<float> u1x_ = u1x(Rect(sx, sy, block_x, block_y));
//                Mat_<float> u1y_ = u1y(Rect(sx, sy, block_x, block_y));
//                Mat_<float> u2x_ = u2x(Rect(sx, sy, block_x, block_y));
//                Mat_<float> u2y_ = u2y(Rect(sx, sy, block_x, block_y));
//
//                for (int n = 0; error > scaledEpsilon && n < iterations; ++n) {
//                    // estimate the values of the variable (v1, v2) (thresholding operator TH)
//                    estimateV(I1wx_, I1wy_, u1_, u2_, grad_, rho_c_, v1_, v2_, l_t);
//
//                    // compute the divergence of the dual variable (p1, p2)
//                    divergence(p11_, p12_, div_p1_);
//                    divergence(p21_, p22_, div_p2_);
//
//                    // estimate the values of the optical flow (u1, u2)
//                    error = estimateU(v1_, v2_, div_p1_, div_p2_, u1_, u2_, static_cast<float>(theta));
//
//                    // compute the gradient of the optical flow (Du1, Du2)
//                    forwardGradient(u1_, u1x_, u1y_);
//                    forwardGradient(u2_, u2x_, u2y_);
//
//                    // estimate the values of the dual variable (p1, p2)
//                    estimateDualVariables(u1x_, u1y_, u2x_, u2y_, p11_, p12_, p21_, p22_, taut);
//                }
//            }
//        }

        float error = 1e7;
        for (int n = 0; error > scaledEpsilon && n < iterations; ++n) {
            // estimate the values of the variable (v1, v2) (thresholding operator TH)
            estimateV(I1wx, I1wy, u1, u2, grad, rho_c, v1, v2, l_t);

            // compute the divergence of the dual variable (p1, p2)
            divergence(p11, p12, div_p1);
            divergence(p21, p22, div_p2);

            // estimate the values of the optical flow (u1, u2)
            error = estimateU(v1, v2, div_p1, div_p2, u1, u2, static_cast<float>(theta));

            // compute the gradient of the optical flow (Du1, Du2)
            forwardGradient(u1, u1x, u1y);
            forwardGradient(u2, u2x, u2y);

            // estimate the values of the dual variable (p1, p2)
            estimateDualVariables(u1x, u1y, u2x, u2y, p11, p12, p21, p22, taut);
        }
    }
}

void calc(const Mat &I0, const Mat &I1, Mat_<Point2f> &flow)
{

    CV_Assert( I0.type() == CV_8UC1 || I0.type() == CV_32FC1 );
    CV_Assert( I0.size() == I1.size() );
    CV_Assert( I0.type() == I1.type() );
    CV_Assert( nscales > 0 );

    // allocate memory for the pyramid structure
    I0s.resize(nscales);
    I1s.resize(nscales);
    u1s.resize(nscales);
    u2s.resize(nscales);

    I0.convertTo(I0s[0], I0s[0].depth(), I0.depth() == CV_8U ? 1.0 : 255.0);
    I1.convertTo(I1s[0], I1s[0].depth(), I1.depth() == CV_8U ? 1.0 : 255.0);

    u1s[0].create(I0.size());
    u2s[0].create(I0.size());

    I1x_buf.create(I0.size());
    I1y_buf.create(I0.size());

    flowMap1_buf.create(I0.size());
    flowMap2_buf.create(I0.size());

    I1w_buf.create(I0.size());
    I1wx_buf.create(I0.size());
    I1wy_buf.create(I0.size());

    grad_buf.create(I0.size());
    rho_c_buf.create(I0.size());

    v1_buf.create(I0.size());
    v2_buf.create(I0.size());

    p11_buf.create(I0.size());
    p12_buf.create(I0.size());
    p21_buf.create(I0.size());
    p22_buf.create(I0.size());

    div_p1_buf.create(I0.size());
    div_p2_buf.create(I0.size());

    u1x_buf.create(I0.size());
    u1y_buf.create(I0.size());
    u2x_buf.create(I0.size());
    u2y_buf.create(I0.size());

    // create the scales
    for (int s = 1; s < nscales; ++s)
    {
        pyrDown(I0s[s - 1], I0s[s]);
        pyrDown(I1s[s - 1], I1s[s]);

//        if (I0s[s].cols < 16 || I0s[s].rows < 16)
//        {
//            nscales = s;
//            break;
//        }

        u1s[s].create(I0s[s].size());
        u2s[s].create(I0s[s].size());
    }

    u1s[nscales-1].setTo(Scalar::all(0));
    u2s[nscales-1].setTo(Scalar::all(0));

    // pyramidal structure for computing the optical flow
    for (int s = nscales - 1; s >= 0; --s)
    {
        printf("%d\n", s);

        // compute the optical flow at the current scale
        procOneScale(I0s[s], I1s[s], u1s[s], u2s[s]);
//        printf("");

        // if this was the last scale, finish now
        if (s == 0)
            break;

        // otherwise, upsample the optical flow

        // zoom the optical flow for the next finer scale
        resize(u1s[s], u1s[s - 1], I0s[s - 1].size());
        resize(u2s[s], u2s[s - 1], I0s[s - 1].size());

        // scale the optical flow with the appropriate zoom factor
        multiply(u1s[s - 1], Scalar::all(2), u1s[s - 1]);
        multiply(u2s[s - 1], Scalar::all(2), u2s[s - 1]);
    }

    flow.create(u1s[0].size());
    for (int i = 0; i < u1s[0].rows; i++)
        for (int j = 0; j < u1s[0].cols; j++)
        {
            flow(i, j).x = u1s[0](i, j);
            flow(i, j).y = u2s[0](i, j);
        }

}

int main(int argc, char **argv) {

    BMP_FILE curr_bmp;

    curr_bmp.file_read((img_path + "1.bmp").c_str());

    BMP_output(curr_bmp);
    cv::Mat curr_img(curr_bmp.info.Height, curr_bmp.info.Width, CV_8UC3);

    for (int i = 0; i < curr_bmp.info.Height; i++)
        for (int j = 0; j < curr_bmp.info.Width; j++)
        {
            int rgb = curr_bmp.get_color(j, i);
            curr_img.at<cv::Vec3b>(i, j)[0] = rgb & 0xFF;
            curr_img.at<cv::Vec3b>(i, j)[1] = (rgb >> 8) & 0xFF;
            curr_img.at<cv::Vec3b>(i, j)[2] = (rgb >> 16) & 0xFF;
        }
    cv::imshow("img", curr_img);
    cv::waitKey(0);

    curr_bmp.file_write((img_path + "test.bmp").c_str());

//    cv::Mat curr_img;
//    cv::Mat prev_img;
//    cv::Mat show_img;
//    cv::Mat_<cv::Point2f> flow;
//    cv::Mat_<cv::Point2f> cv_flow;
//    cv::Mat error_img(480, 640, CV_8UC1);
//
//    curr_img = cv::imread(img_path + "1.jpg");
//    cv::imwrite(img_path + "1.bmp", curr_img);
//    curr_img = curr_img(Rect(0, 0, 640, 480));
//    show_img = curr_img.clone();
//    cv::cvtColor(curr_img, curr_img, cv::COLOR_BGR2GRAY);
//
//    std::vector<cv::Point2f> kpts;
//
//    cv::goodFeaturesToTrack(curr_img, kpts, 2000, 0.01, 10);
//    prev_img = curr_img;
//
//    for (int i = 2; i < 20; i++)
//    {
//        curr_img = cv::imread(img_path + std::to_string(i) + ".jpg");
//        // cv::imwrite(img_path + std::to_string(i) + ".bmp", curr_img);
//        std::cout << 'path: ' << img_path + std::to_string(i) + ".jpg" << std::endl;
//        std::cout << i << ' ' << curr_img.size() << std::endl;
//        cv::imshow("curr_img", curr_img);
//        cv::waitKey(1);

//        curr_img = curr_img(Rect(0, 0, 640, 480));
//        show_img = curr_img.clone();
//
//        cv::cvtColor(curr_img, curr_img, cv::COLOR_BGR2GRAY);
//
//        printf("before calc...\n");
//
//        int t = clock();
//        calc(prev_img, curr_img, flow);
//        printf("calc: %d\n", clock() - t);
//
//        t = clock();
//        cv::Ptr<cv::DenseOpticalFlow> tvl1 = cv::createOptFlow_DualTVL1();
//        tvl1->calc(prev_img, curr_img, cv_flow);
//        printf("opencv calc: %d\n", clock() - t);
//
//        for (int y = 0; y < 480; y++)
//            for (int x = 0; x < 640; x++)
//                error_img.at<uchar>(y, x) = min(255.0, (abs(flow(y, x).x - cv_flow(y, x).x) + abs(flow(y, x).y - cv_flow(y, x).y)) * 5.0);
//        cv::imshow("error img", error_img);
//
////        for (int y = 0; y < 480; y += 8)
////        {
////            for (int x = 0; x < 640; x += 8)
////            {
////                printf("(%d, %d) ", int(flow(y, x).x), int(flow(y, x).y));
////            }
////            printf("\n");
////        }
////        printf("after calc...\n");
//
//        // char ch;
//        // std::cin >> ch;
//
//        int cnt = 0;
//        for (int j = 0; j < kpts.size(); j++) {
//            int x = kpts[j].x;
//            int y = kpts[j].y;
//
//            kpts[j].x += flow(y, x).x;
//            kpts[j].y += flow(y, x).y;
//
//            if (kpts[j].x > 0 && kpts[j].x < 640 && kpts[j].y > 0 && kpts[j].y < 480) {
//                kpts[cnt] = kpts[j];
//
//                cv::circle(show_img, kpts[cnt], 2, cv::Scalar(255, 0, 0), 2);
//
//                cnt++;
//            }
//        }
//        kpts.resize(cnt);
//
//        cv::imshow("img", show_img);
//        cv::waitKey(0);
//        prev_img.release();
//        show_img.release();
//
//        prev_img = curr_img;

//    }

    return 0;
}

