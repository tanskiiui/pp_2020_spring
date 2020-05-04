// Copyright 2020 Tanskii Yurii
#include <gtest/gtest.h>
#include <vector>
#include "../../../modules/task_1/tanskii_u_matrix_multiply/matrix_multiplication.h"

TEST(Matrix_Multiplication, Can_Random_Matrix) {
    size_t row = 5;
    size_t col = 5;
    std::vector<std::vector<double>> result = GetRandomMatrix(row, col, 5);
    size_t n = 25;
    size_t elems_c = result.size() * result[0].size();
    EXPECT_EQ(n, elems_c);
}

TEST(Matrix_Multiplication, Can_Sparse_Matrix) {
    std::vector<std::vector<double>> Matrix { { 0.0, 2.0, 0.0, 0.0, 6.5, 0.0 },
                   { 0.0, 0.0, 7.3, 0.0, 0.0, 0.0 },
                   { 0.0, 0.0, 0.0, 8.2, 0.0, 0.0 },
                   { 11.1, 0.0, 0.0, 0.0, 0.0, 13.0 },
                   { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
                   { 0.0, 0.0, 15.6, 0.0, 0.0, 0.0} };
    std::vector<double> value = { 2.0, 6.5, 7.3, 8.2, 11.1, 13.0, 15.6 };
    std::vector<size_t> indexCol = { 1, 4, 2, 3, 0, 5, 2 };
    std::vector<size_t> indexRow = { 0, 2, 3, 4, 6, 6, 7 };
    MMatrix M1(Matrix);
    MMatrix M2(value, indexRow, indexCol, indexRow.size(), indexCol.size());
    EXPECT_EQ(M1, M2);
}

TEST(Matrix_Multiplication, Can_Throw_Incorrect_Matrix) {
    std::vector<std::vector<double>> Matrix;
    MMatrix M1;
    ASSERT_ANY_THROW(M1(Matrix));
}

TEST(Matrix_Multiplication, Can_Multiply_Sparse_Matrix) {
    std::vector<vector<double>> A { { 1.2, 0.0, 0.0, 0.0 },
                                       { 0.0, 0.0, 3.4, 0.0 },
                                       { 0.0, 2.5, 0.0, 0.0 },
                                       { 0.0, 0.0, 0.0, 0.0 } };
    std::vector<vector<double>> B { { 4, 0.0, 0.0, 0.0 },
                                       { 0.0, 2.5, 0.0, 0.0 },
                                       { 0.0, 0.0, 0.0, 0.0 },
                                       { 0.0, 0.0, 0.0, 3.3 } };
    std::vector<vector<double>> C { { 4.8, 0.0, 0.0, 0.0 },
                                       { 0.0, 0.0, 0.0, 0.0 },
                                       { 0.0, 6.5, 0.0, 0.0 },
                                       { 0.0, 0.0, 0.0, 0.0 } };
    MMatrix sparseA(A);
    MMatrix sparseB(B);
    MMatrix sparseC(C);
    MMatrix tmp;
    tmp = sparseA.DischargeMultiply(sparseB);
    ASSERT_EQ(tmp, sparseC);
}

TEST(Matrix_Multiplication, Sparse_Multiply_Equal_naive) {
    std::vector<vector<double>> A { { 1.2, 0.0, 0.0, 0.0 },
                                       { 0.0, 0.0, 3.4, 0.0 },
                                       { 0.0, 2.5, 0.0, 0.0 },
                                       { 0.0, 0.0, 0.0, 0.0 } };
    std::vector<vector<double>> B { { 4, 0.0, 0.0, 0.0 },
                                       { 0.0, 2.5, 0.0, 0.0 },
                                       { 0.0, 0.0, 0.0, 0.0 },
                                       { 0.0, 0.0, 0.0, 3.3 } };
    std::vector<vector<double>> C { { 4.8, 0.0, 0.0, 0.0 },
                                       { 0.0, 0.0, 0.0, 0.0 },
                                       { 0.0, 6.5, 0.0, 0.0 },
                                       { 0.0, 0.0, 0.0, 0.0 } };
    std::vector<vector<double>> naiveResult;
    MMatrix sparseA(A);
    MMatrix sparseB(B);
    MMatrix sparseC(C);
    MMatrix sparseResult;
    sparseResult = sparseA.DischargeMultiply(sparseB);
    naiveResult = NaiveMultiplication(A, B);
    MMatrix sparseNaive(naiveResult);
    ASSERT_EQ(naiveResult, sparseResult);
}

TEST(Matrix_Multiplication, Can_Transpose_Matrix) {
    std::vector<std::vector<double>> A { { 0.0, 3.0, 0.0, 7.0  },
                                         { 0.0, 0.0, 8.0, 8.0 },
                                         { 0.0, 0.0, 0.0, 0.0 },
                                         { 9.0, 0.0, 15.0, 16.0  } };
    std::vector<std::vector<double>> AT { { 0.0, 0.0, 0.0, 9.0  },
                                          { 3.0, 0.0, 0.0, 0.0 },
                                          { 0.0, 8.0, 0.0, 15.0 },
                                          { 7.0, 0.0, 0.0, 16.0  } };
    MMatrix sparseMatrix(Matrix);
    MMatrix transMatrix = sparseMatrix.Transpose();
    MMatrix sparseAT(AT);
    EXPECT_EQ(AT, transMatrix);
}
