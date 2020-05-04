// Copyright 2020 Tanskii Yurii
#include <limits>
#include<stdexcept>
#include <random>
#include <vector>
#include"../../../modules/task_1/tanskii_u_matrix_multiply/matrix_multiplication.h"

double eps = std::numeric_limits<double>::epsilon();

MMatrix::MMatrix(std::vector<std::vector<double>> Matrix) {
    if (Matrix.size() <= 0 || Matrix[0].size() <= 0)
        throw std::runtime_error("Incorrect size of matrix");
    row = Matrix.size();
    col = Matrix[0].size();
    indexRow.push_back(0);
    int PositiveElems = 0;
    for (size_t i = 0; i < row; i++) {
        for (size_t j = 0; j < col; j++) {
            if (Matrix[i][j] > eps) {
                value.push_back(Matrix[i][j]);
                indexCol.push_back(j);
                PositiveElems++;
            }
        }
        indexRow.push_back(PositiveElems);
    }
}

MMatrix MMatrix::Transpose() {
    MMatrix result;
    result.col = col;
    result.row = row;
    result.indexRow.push_back(0);
    std::vector<std::vector<size_t>> v_index(col);
    std::vector<std::vector<double>> v_values(col);
    for (size_t i = 1; i < indexRow.size(); i++) {
        for (size_t j = indexRow[i - 1]; j < indexRow[i]; j++) {
            v_index[indexCol[j]].push_back(i - 1);
            v_values[indexCol[j]].push_back(value[j]);
        }
    }
    int size = 0;
    for (size_t i = 0; i < col; ++i) {
        for (size_t j = 0; j < v_index[i].size(); ++j) {
            result.value.push_back(v_values[i][j]);
            result.indexCol.push_back(v_index[i][j]);
        }
        size += v_index[i].size();
        result.indexRow.push_back(size);
    }
    return result;
}

std::vector<std::vector<double>> NaiveMultiplication(std::vector<std::vector<double>> Matrix1,
    std::vector<std::vector<double>> Matrix2) {
    if (Matrix1[0].size() != Matrix2.size())
        throw std::runtime_error("Incorrect size of Multiplying matrix");
    std::vector<std::vector<double>> result;
    size_t rows_m1 = Matrix1.size();
    size_t cols_m2 = Matrix2[0].size();
    size_t cols_m1 = Matrix1[0].size();
    result.resize(rows_m1);
    for (size_t i = 0; i < rows_m1; i++) {
        result[i].resize(cols_m2);
    }
    for (size_t i = 0; i < rows_m1; i++)
        for (size_t j = 0; j < cols_m2; j++) {
            result[i][j] = 0;
            for (size_t k = 0; k < cols_m1; k++)
                result[i][j] += Matrix1[i][k] * Matrix2[k][j];
        }
    return result;
}

MMatrix MMatrix::DischargeMultiply(const MMatrix& Matrix2) {
    MMatrix result;
    result.row = row;
    result.col = Matrix2.col;
    result.indexRow.push_back(0);
    std::vector<double> tmp_vec(Matrix2.col, 0);

    for (size_t i = 0; i < row; ++i) {
        for (size_t j = indexRow[i]; j < indexRow[i + 1]; ++j) {
            for (size_t k = Matrix2.indexRow[indexCol[j]]; k < Matrix2.indexRow[indexCol[j] + 1]; ++k) {
                tmp_vec[Matrix2.indexCol[k]] += value[j] * Matrix2.value[k];
            }
        }
        for (size_t k = 0; k < Matrix2.col; ++k) {
            if (tmp_vec[k] != 0) {
                result.value.push_back(tmp_vec[k]);
                result.indexCol.push_back(k);
                tmp_vec[k] = 0;
            }
        }
        result.indexRow.push_back(result.value.size());
    }
    return result;
}

bool MMatrix::operator== (const MMatrix& Matrix) const& {
    if (row != Matrix.row || col!= Matrix.col ||
        value.size() != Matrix.value.size() ||
        indexRow.size() != Matrix.indexRow.size() ||
        indexCol.size() != Matrix.indexCol.size()) {
        return false;
    }
    for (size_t i = 0; i < value.size(); i++) {
        if (std::abs(value[i] - Matrix.value[i]) > eps)
            return false;
    }
    return true;
}

std::vector<std::vector<double>>  GetRandomMatrix(size_t row, size_t col, double percent) {
    if (percent <= 0 || percent > 100) {
        throw std::runtime_error("Incorrect value of percent. Must be ""0 < percent < 1""");
    }
    std::vector<std::vector<double>> result;
    result.resize(row);
    for (size_t i = 0; i < row; i++) {
        result[i].resize(col);
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> disValue(0.0, 100.0);
    std::uniform_int_distribution<int> disPercent(0, 100);
    for (size_t i = 0; i < row; i++) {
        for (size_t j = 0; j < col; j++) {
            result[i][j] = 0.0;
            if (disPercent(gen) < percent) {
                result[i][j] = disValue(gen);
            }
        }
    }
    return result;
}
