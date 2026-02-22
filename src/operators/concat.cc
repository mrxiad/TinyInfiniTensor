#include "operators/concat.h"
#include "utils/operator_utils.h"

namespace infini {
ConcatObj::ConcatObj(GraphObj *graph, TensorVec inputs, Tensor output, int _dim)
    : OperatorObj(OpType::Concat, inputs, {output}) {
    int rank = inputs[0]->getRank();
    dim = get_real_axis(_dim, rank);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ConcatObj::inferShape(const TensorVec &inputs) {
    IT_ASSERT(!inputs.empty());
    Shape dims = inputs[0]->getDims();
    const auto rank = inputs[0]->getRank();
    IT_ASSERT(dim >= 0 && dim < static_cast<int>(rank));

    // =================================== 作业 ===================================
    // TODO：修改 dims，返回正确的 concat 后的 shape
    // REF: https://onnx.ai/onnx/operators/onnx__Concat.html#concat-13
    // =================================== 作业 ===================================
    for (size_t i = 1; i < inputs.size(); ++i) {
        const auto &cur = inputs[i]->getDims();
        IT_ASSERT(cur.size() == rank);
        for (size_t axis = 0; axis < rank; ++axis) {
            if (static_cast<int>(axis) == dim) {
                dims[axis] += cur[axis];
            } else {
                IT_ASSERT(dims[axis] == cur[axis]);
            }
        }
    }

    return {{dims}};
}

std::string ConcatObj::toString() const {
    std::ostringstream os;
    os << "Concat[" << getGuid() << "]";
    os << "(";
    for (auto input : inputs)
        os << vecToString(input->getDims()) << ",";
    os << "dim=" << dim << ",";
    os << "input=";
    for (auto input : inputs)
        os << input->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

} // namespace infini
