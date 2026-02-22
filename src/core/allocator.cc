#include "core/allocator.h"
#include <algorithm>
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);
        IT_ASSERT(size > 0);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        for (auto it = freeBlocks.begin(); it != freeBlocks.end(); ++it)
        {
            if (it->second < size)
            {
                continue;
            }
            const auto addr = it->first;
            const auto blockSize = it->second;
            freeBlocks.erase(it);
            if (blockSize > size)
            {
                freeBlocks.emplace(addr + size, blockSize - size);
            }
            return addr;
        }

        const auto addr = used;
        used += size;
        peak = std::max(peak, used);
        return addr;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);
        IT_ASSERT(size > 0);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        auto trim_tail = [&]()
        {
            while (!freeBlocks.empty())
            {
                auto it = freeBlocks.lower_bound(used);
                if (it == freeBlocks.begin())
                {
                    break;
                }
                --it;
                if (it->first + it->second != used)
                {
                    break;
                }
                used = it->first;
                freeBlocks.erase(it);
            }
        };

        if (addr + size == used)
        {
            used = addr;
            trim_tail();
            return;
        }

        auto [it, inserted] = freeBlocks.emplace(addr, size);
        IT_ASSERT(inserted, "Duplicated free block");

        if (it != freeBlocks.begin())
        {
            auto prev = std::prev(it);
            if (prev->first + prev->second == it->first)
            {
                const auto mergedStart = prev->first;
                const auto mergedSize = prev->second + it->second;
                freeBlocks.erase(prev);
                freeBlocks.erase(it);
                it = freeBlocks.emplace(mergedStart, mergedSize).first;
            }
        }

        while (true)
        {
            auto next = std::next(it);
            if (next == freeBlocks.end() || it->first + it->second != next->first)
            {
                break;
            }
            const auto mergedStart = it->first;
            const auto mergedSize = it->second + next->second;
            freeBlocks.erase(next);
            freeBlocks.erase(it);
            it = freeBlocks.emplace(mergedStart, mergedSize).first;
        }

        if (it->first + it->second == used)
        {
            used = it->first;
            freeBlocks.erase(it);
            trim_tail();
        }
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        if (size == 0)
        {
            return 0;
        }
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
