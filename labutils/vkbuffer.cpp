#include "vkbuffer.hpp"

#include <utility>

#include <cassert>

#include "error.hpp"
#include "to_string.hpp"



namespace labutils
{
	Buffer::Buffer() noexcept = default;

	Buffer::~Buffer()
	{
		if( VK_NULL_HANDLE != buffer )
		{
			assert( VK_NULL_HANDLE != mAllocator );
			assert( VK_NULL_HANDLE != allocation );
			vmaDestroyBuffer( mAllocator, buffer, allocation );
		}
	}

	Buffer::Buffer( VmaAllocator aAllocator, VkBuffer aBuffer, VmaAllocation aAllocation ) noexcept
		: buffer( aBuffer )
		, allocation( aAllocation )
		, mAllocator( aAllocator )
	{}

	Buffer::Buffer( Buffer&& aOther ) noexcept
		: buffer( std::exchange( aOther.buffer, VK_NULL_HANDLE ) )
		, allocation( std::exchange( aOther.allocation, VK_NULL_HANDLE ) )
		, mAllocator( std::exchange( aOther.mAllocator, VK_NULL_HANDLE ) )
	{}
	Buffer& Buffer::operator=( Buffer&& aOther ) noexcept
	{
		std::swap( buffer, aOther.buffer );
		std::swap( allocation, aOther.allocation );
		std::swap( mAllocator, aOther.mAllocator );
		return *this;
	}
}

namespace labutils
{
Buffer create_buffer( Allocator const& aAllocator, VkDeviceSize aSize, VkBufferUsageFlags aBufferUsage, VmaMemoryUsage aMemoryUsage )
{
	VkBufferCreateInfo bufferInfo{}; {
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;

		bufferInfo.size = aSize;
		bufferInfo.usage = aBufferUsage;
	}

	VmaAllocationCreateInfo allocationInfo{}; {
		allocationInfo.usage = aMemoryUsage;
	}

	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	if (auto const res = vmaCreateBuffer(aAllocator.allocator, &bufferInfo, &allocationInfo, &buffer, &allocation, nullptr); res != VK_SUCCESS)
	{
		throw Error("Unable to Allocate Buffer.\n"
			"vmaCreateBuffer() Returned %s", to_string(res).c_str());
	}

	return Buffer(aAllocator.allocator, buffer, allocation);
}
}
