#include "operators/matmul.h"
#include "utils/operator_utils.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        IT_ASSERT(inputs.size() == 2);
        const auto A = inputs[0];
        const auto B = inputs[1];
        const auto &shapeA = A->getDims();
        const auto &shapeB = B->getDims();
        const auto rankA = static_cast<int>(shapeA.size());
        const auto rankB = static_cast<int>(shapeB.size());
        IT_ASSERT(rankA >= 2 && rankB >= 2);

        const int mA = transA ? shapeA[rankA - 1] : shapeA[rankA - 2];
        const int kA = transA ? shapeA[rankA - 2] : shapeA[rankA - 1];
        const int kB = transB ? shapeB[rankB - 1] : shapeB[rankB - 2];
        const int nB = transB ? shapeB[rankB - 2] : shapeB[rankB - 1];
        IT_ASSERT(kA == kB);

        Shape batchA(shapeA.begin(), shapeA.end() - 2);
        Shape batchB(shapeB.begin(), shapeB.end() - 2);
        Shape out = infer_broadcast(batchA, batchB);
        out.emplace_back(mA);
        out.emplace_back(nB);

        m = mA;
        n = nB;
        k = kA;

        return {{out}};
    }

} // namespace infini
