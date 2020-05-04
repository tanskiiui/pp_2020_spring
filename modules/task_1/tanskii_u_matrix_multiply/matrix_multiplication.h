// Copyright 2020 Tanskii Yurii
#ifndef MODULES_TASK_1_TANSKII_U_MATRIX_MULTIPLY_MATRIX_MULTIPLICATION_H_
#define MODULES_TASK_1_TANSKII_U_MATRIX_MULTIPLY_MATRIX_MULTIPLICATION_H_
#include<vector>
#include <iostream>

class MMatrix {
    std::vector<double> value;
    std::vector<size_t> indexRow;
    std::vector<size_t> indexCol;
    size_t row, col;

 public:
    explicit MMatrix(std::vector<std::vector<double>> Matrix);
    explicit MMatrix(const size_t& valSize = 0, const size_t& colSize = 0, const size_t& rowSize = 0,
        const size_t& _col = 0, const size_t& _row = 0) : value(valSize, 0), indexCol(colSize, 0),
        indexRow(rowSize, 0), row(_row), col(_col) {}
    MMatrix(std::vector<double> val, std::vector<size_t> iRow, std::vector<size_t> iCol,
        size_t _row, size_t _col);
    MMatrix Transpose();
    MMatrix DischargeMultiply(const MMatrix &Matrix2);
    bool operator == (const MMatrix& Matrix) const&;
};

std::vector<std::vector<double>> NaiveMultiplication(const std::vector<std::vector<double>> Matrix1,
    std::vector<std::vector<double>> Matrix2);
std::vector<std::vector<double>> GetRandomMatrix(size_t row, size_t col, double percent);
#endif  // MODULES_TASK_1_TANSKII_U_MATRIX_MULTIPLY_MATRIX_MULTIPLICATION_H_
