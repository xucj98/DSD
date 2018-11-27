//
// Created by jay on 18-11-27.
//

#ifndef FPGA_TVL1_ALGORITHM_MAT_H
#define FPGA_TVL1_ALGORITHM_MAT_H

#include <stdlib.h>
#include <malloc.h>

template <typename T>
class Mat_ {

public:
    typedef struct MatSize_s
    {
        int rows, cols;
    }MatSize_t;

    Mat_() {
        data = NULL;
        mem_rows = mem_cols = 0;
    }

    Mat_(int rows_, int cols_) {
        data = NULL;
        mem_rows = mem_cols = 0;
        resize(rows_, cols_);
    }

    ~Mat_() {
        freeData();
    }

    void resize(const int rows_, const int cols_) {
        if (rows_ <= mem_rows && cols_ <= mem_cols) {
            rows = rows_; cols = cols_;
        }else {
            freeData();
            data = new T*[cols_];
            for(int i=0; i<rows_; ++i) {
                data[i] = new T[cols_];
            }
            mem_rows = rows = rows_;
            mem_cols = cols = cols_;
        }
    }

    void create(const MatSize_t size_) {
        resize(size_.rows, size_.cols);
    }

    MatSize_t size(){
        MatSize_t ret;
        ret.rows = rows;
        ret.cols = cols;
        return ret;
    }

    const MatSize_t size() const {
        MatSize_t ret;
        ret.rows = rows;
        ret.cols = cols;
        return ret;
    }

    void freeData() {
        if (data != NULL) {
            for(int i=0; i<mem_rows; i++) {
                delete[] data[i];
            }
            delete[] data;
        }
    }

    T* operator[](int i)
    {
        return data[i];
    }

    const T* operator[](int i) const
    {
        return data[i];
    }

    T& operator() (int i, int j)
    {
        return data[i][j];
    }

    const T& operator() (int i, int j) const
    {
        return data[i][j];
    }

    void setTo(T value) {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i][j] = value;
    }

    int rows, cols;

private:

    T **data;
    int mem_rows, mem_cols;

};


#endif //FPGA_TVL1_ALGORITHM_MAT_H
