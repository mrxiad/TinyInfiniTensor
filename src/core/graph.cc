#include "core/graph.h"
#include "operators/matmul.h"
#include "operators/transpose.h"
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <queue>

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================
        auto isInversePermute = [](const std::vector<int> &p1,
                                   const std::vector<int> &p2)
        {
            if (p1.size() != p2.size())
            {
                return false;
            }
            const int rank = static_cast<int>(p1.size());
            for (int i = 0; i < rank; ++i)
            {
                if (p2[p1[i]] != i)
                {
                    return false;
                }
            }
            return true;
        };

        auto isLastTwoSwap = [](const std::vector<int> &perm, int rank)
        {
            if (static_cast<int>(perm.size()) != rank || rank < 2)
            {
                return false;
            }
            for (int i = 0; i < rank - 2; ++i)
            {
                if (perm[i] != i)
                {
                    return false;
                }
            }
            return perm[rank - 2] == rank - 1 && perm[rank - 1] == rank - 2;
        };

        auto eraseOp = [&](const Operator &victim)
        {
            ops.erase(std::remove(ops.begin(), ops.end(), victim), ops.end());
        };

        auto eraseTensor = [&](const Tensor &victim)
        {
            tensors.erase(std::remove(tensors.begin(), tensors.end(), victim),
                          tensors.end());
        };

        auto rebuildConnections = [&]()
        {
            for (auto &tensor : tensors)
            {
                tensor->targets.clear();
                tensor->source.reset();
            }
            for (auto &op : ops)
            {
                op->predecessors.clear();
                op->successors.clear();
            }
            for (auto &op : ops)
            {
                for (auto &output : op->outputs)
                {
                    output->source = op;
                }
            }
            for (auto &op : ops)
            {
                for (auto &input : op->inputs)
                {
                    input->targets.emplace_back(op);
                    if (auto pred = input->source.lock())
                    {
                        pred->successors.emplace_back(op);
                        op->predecessors.emplace_back(pred);
                    }
                }
            }
        };

        rebuildConnections();
        bool changed = true;
        while (changed)
        {
            changed = false;

            // Rule 1: eliminate inverse transpose pairs
            for (const auto &op : ops)
            {
                if (op->getOpType() != OpType::Transpose)
                {
                    continue;
                }
                const auto trans2Op = op;
                auto trans2 = as<TransposeObj>(trans2Op);
                auto trans2Input = trans2->getInputs(0);
                auto trans1Op = trans2Input->getSource();
                if (!trans1Op || trans1Op->getOpType() != OpType::Transpose)
                {
                    continue;
                }
                auto trans1 = as<TransposeObj>(trans1Op);
                auto trans1Out = trans1->getOutput();
                if (trans1Out->getTargets().size() != 1 ||
                    trans1Out->getTargets()[0] != op)
                {
                    continue;
                }
                if (!isInversePermute(trans1->getPermute(), trans2->getPermute()))
                {
                    continue;
                }

                auto trans1Input = trans1->getInputs(0);
                auto trans2Out = trans2->getOutput();
                auto successors = trans2Out->getTargets();
                for (const auto &succ : successors)
                {
                    succ->replaceInput(trans2Out, trans1Input);
                }

                eraseOp(trans1Op);
                eraseOp(trans2Op);
                eraseTensor(trans1Out);
                eraseTensor(trans2Out);
                changed = true;
                break;
            }
            if (changed)
            {
                rebuildConnections();
                continue;
            }

            // Rule 2: fuse transpose into matmul transA/transB
            for (const auto &op : ops)
            {
                if (op->getOpType() != OpType::MatMul)
                {
                    continue;
                }
                auto matmul = as<MatmulObj>(op);
                for (int inputId = 0; inputId < 2; ++inputId)
                {
                    auto input = matmul->getInputs(inputId);
                    auto transOp = input->getSource();
                    if (!transOp || transOp->getOpType() != OpType::Transpose)
                    {
                        continue;
                    }
                    if (input->getTargets().size() != 1 || input->getTargets()[0] != op)
                    {
                        continue;
                    }
                    auto trans = as<TransposeObj>(transOp);
                    if (!isLastTwoSwap(trans->getPermute(),
                                       static_cast<int>(input->getRank())))
                    {
                        continue;
                    }

                    matmul->replaceInput(input, trans->getInputs(0));
                    if (inputId == 0)
                    {
                        matmul->setTransA(!matmul->getTransA());
                    }
                    else
                    {
                        matmul->setTransB(!matmul->getTransB());
                    }

                    eraseOp(transOp);
                    eraseTensor(input);
                    changed = true;
                    break;
                }
                if (changed)
                {
                    break;
                }
            }
            if (changed)
            {
                rebuildConnections();
            }
        }

        rebuildConnections();
        for (auto it = tensors.begin(); it != tensors.end();)
        {
            auto tensor = *it;
            if (tensor->targets.empty() && tensor->source.expired())
            {
                it = tensors.erase(it);
            }
            else
            {
                ++it;
            }
        }
        rebuildConnections();
        sorted = false;
        (void)topo_sort();
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================
        std::unordered_map<TensorObj *, size_t> offsets;
        std::unordered_map<TensorObj *, size_t> bytes;
        std::unordered_map<TensorObj *, int> remainUses;
        std::unordered_set<TensorObj *> graphOutputs;

        for (const auto &tensor : tensors)
        {
            remainUses.emplace(tensor.get(),
                               static_cast<int>(tensor->getTargets().size()));
        }
        for (const auto &tensor : getOutputs())
        {
            graphOutputs.emplace(tensor.get());
        }

        auto ensureAllocated = [&](const Tensor &tensor)
        {
            auto *key = tensor.get();
            if (offsets.find(key) != offsets.end())
            {
                return;
            }
            auto nbytes = tensor->getBytes();
            auto offset = allocator.alloc(nbytes);
            offsets.emplace(key, offset);
            bytes.emplace(key, nbytes);
        };

        for (const auto &op : ops)
        {
            for (const auto &input : op->getInputs())
            {
                ensureAllocated(input);
            }
            for (const auto &output : op->getOutputs())
            {
                ensureAllocated(output);
            }

            for (const auto &input : op->getInputs())
            {
                auto *key = input.get();
                auto it = remainUses.find(key);
                IT_ASSERT(it != remainUses.end());
                IT_ASSERT(it->second > 0);
                --(it->second);
                if (it->second == 0 && graphOutputs.find(key) == graphOutputs.end())
                {
                    allocator.free(offsets.at(key), bytes.at(key));
                }
            }
        }

        auto *basePtr = reinterpret_cast<uint8_t *>(allocator.getPtr());
        for (const auto &tensor : tensors)
        {
            ensureAllocated(tensor);
            auto addr = offsets.at(tensor.get());
            tensor->setDataBlob(make_ref<BlobObj>(runtime, basePtr + addr));
        }

        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini
