#include "buffer.hpp"

#include <utility>

Buffer::Buffer() noexcept = default;

Buffer::~Buffer()
{
	if( VK_NULL_HANDLE != buffer )
		vkDestroyBuffer( mDevice, buffer, nullptr );

	if( VK_NULL_HANDLE != memory )
		vkFreeMemory( mDevice, memory, nullptr );
}

Buffer::Buffer( VkDevice aDevice, VkBuffer aBuffer, VkDeviceMemory aMemory ) noexcept
	: buffer( aBuffer )
	, memory( aMemory )
	, mDevice( aDevice )
{}

Buffer::Buffer( Buffer&& aOther ) noexcept
	: buffer( std::exchange( aOther.buffer, VK_NULL_HANDLE ) )
	, memory( std::exchange( aOther.memory, VK_NULL_HANDLE ) )
	, mDevice( std::exchange( aOther.mDevice, VK_NULL_HANDLE ) )
{}

Buffer& Buffer::operator= (Buffer&& aOther) noexcept
{
	std::swap( buffer, aOther.buffer );
	std::swap( memory, aOther.memory );
	std::swap( mDevice, aOther.mDevice );
	return *this;
}

