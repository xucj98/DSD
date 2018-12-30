//
// Created by jay on 18-12-30.
//

#include <stdio.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

const std::string data_path = "../../Dataset/";

float tau            = 0.25;
float lambda         = 0.15;
float theta          = 0.3;
int nscales        = 5;
int warps          = 5;
float epsilon        = 0.01;
int iterations     = 300;
bool useInitialFlow = false;

Mat_<float> I1wx_(30, 40);
Mat_<float> I1wy_(30, 40);
Mat_<float> u1_(30, 40);
Mat_<float> u2_(30, 40);
Mat_<float> grad_(30, 40);
Mat_<float> rho_c_(30, 40);
Mat_<float> v1_(30, 40);
Mat_<float> v2_(30, 40);
Mat_<float> p11_(30, 40);
Mat_<float> p12_(30, 40);
Mat_<float> p21_(30, 40);
Mat_<float> p22_(30, 40);
Mat_<float> div_p1_(30, 40);
Mat_<float> div_p2_(30, 40);
Mat_<float> u1x_(30, 40);
Mat_<float> u1y_(30, 40);
Mat_<float> u2x_(30, 40);
Mat_<float> u2y_(30, 40);

Mat_<float> ansp11_(30, 40);
Mat_<float> ansp12_(30, 40);
Mat_<float> ansp21_(30, 40);
Mat_<float> ansp22_(30, 40);
Mat_<float> ansu1_(30, 40);
Mat_<float> ansu2_(30, 40);

const int block_x = 40, block_y = 30;
const float scaledEpsilon = epsilon * epsilon * block_x * block_y;
const float l_t = lambda * theta;
const float taut = tau / theta;

////////////////////////////////////////////////////////////
// forwardGradient


void forwardGradient(const Mat_<float>& src, Mat_<float>& dx, Mat_<float>& dy)
{
//    CV_DbgAssert( src.rows > 2 && src.cols > 2 );
//    CV_DbgAssert( dx.size() == src.size() );
//    CV_DbgAssert( dy.size() == src.size() );

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
//    CV_DbgAssert( v1.rows > 2 && v1.cols > 2 );
//    CV_DbgAssert( v2.size() == v1.size() );
//    CV_DbgAssert( div.size() == v1.size() );

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
// estimateV

void estimateV(const Mat_<float>& I1wx, const Mat_<float>& I1wy, const Mat_<float>& u1, const Mat_<float>& u2,
               const Mat_<float>& grad, const Mat_<float>& rho_c, Mat_<float>& v1, Mat_<float>& v2, float l_t)
{
//    CV_DbgAssert( I1wy.size() == I1wx.size() );
//    CV_DbgAssert( u1.size() == I1wx.size() );
//    CV_DbgAssert( u2.size() == I1wx.size() );
//    CV_DbgAssert( grad.size() == I1wx.size() );
//    CV_DbgAssert( rho_c.size() == I1wx.size() );
//    CV_DbgAssert( v1.size() == I1wx.size() );
//    CV_DbgAssert( v2.size() == I1wx.size() );

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
//    CV_DbgAssert( v2.size() == v1.size() );
//    CV_DbgAssert( div_p1.size() == v1.size() );
//    CV_DbgAssert( div_p2.size() == v1.size() );
//    CV_DbgAssert( u1.size() == v1.size() );
//    CV_DbgAssert( u2.size() == v1.size() );

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
//    CV_DbgAssert( u1y.size() == u1x.size() );
//    CV_DbgAssert( u2x.size() == u1x.size() );
//    CV_DbgAssert( u2y.size() == u1x.size() );
//    CV_DbgAssert( p11.size() == u1x.size() );
//    CV_DbgAssert( p12.size() == u1x.size() );
//    CV_DbgAssert( p21.size() == u1x.size() );
//    CV_DbgAssert( p22.size() == u1x.size() );

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

int main()
{
    FILE *fin, *fout;
    float a[10], b[10];

    fin = fopen((data_path + "data.in").c_str(), "rb");
    for (int i = 0; i < block_y; i++)
    {
        fread(I1wx_[i], 4, block_x, fin);
        fread(I1wy_[i], 4, block_x, fin);
        fread(u1_[i], 4, block_x, fin);
        fread(u2_[i], 4, block_x, fin);
        fread(grad_[i], 4, block_x, fin);
        fread(rho_c_[i], 4, block_x, fin);
        fread(p11_[i], 4, block_x, fin);
        fread(p12_[i], 4, block_x, fin);
        fread(p21_[i], 4, block_x, fin);
        fread(p22_[i], 4, block_x, fin);
    }
    fclose(fin);


    float error = 1e7;
    for (int n = 0; error > scaledEpsilon && n < iterations; ++n) {

        // estimate the values of the variable (v1, v2) (thresholding operator TH)
        estimateV(I1wx_, I1wy_, u1_, u2_, grad_, rho_c_, v1_, v2_, l_t);

        // compute the divergence of the dual variable (p1, p2)
        divergence(p11_, p12_, div_p1_);
        divergence(p21_, p22_, div_p2_);

        // estimate the values of the optical flow (u1, u2)
        error = estimateU(v1_, v2_, div_p1_, div_p2_, u1_, u2_, theta);


        // compute the gradient of the optical flow (Du1, Du2)
        forwardGradient(u1_, u1x_, u1y_);
        forwardGradient(u2_, u2x_, u2y_);

        // estimate the values of the dual variable (p1, p2)
        estimateDualVariables(u1x_, u1y_, u2x_, u2y_, p11_, p12_, p21_, p22_, taut);
    }

    fin = fopen((data_path + "data.out").c_str(), "rb");
    for (int i = 0; i < block_y; i++)
    {
        fread(ansu1_[i], 4, block_x, fin);
        fread(ansu2_[i], 4, block_x, fin);
        fread(ansp11_[i], 4, block_x, fin);
        fread(ansp12_[i], 4, block_x, fin);
        fread(ansp21_[i], 4, block_x, fin);
        fread(ansp22_[i], 4, block_x, fin);
    }
    fclose(fin);

    float anserror = 0;
    for (int i = 0; i < block_y; i++)
        for (int j = 0; j < block_x; j++)
        {
            anserror += fabs(ansu1_[i][j] - u1_[i][j]);
            anserror += fabs(ansu2_[i][j] - u2_[i][j]);
            anserror += fabs(ansp11_[i][j] - p11_[i][j]);
            anserror += fabs(ansp12_[i][j] - p12_[i][j]);
            anserror += fabs(ansp21_[i][j] - p21_[i][j]);
            anserror += fabs(ansp22_[i][j] - p22_[i][j]);
        }

    printf("%f\n", anserror);
    return 0;
}