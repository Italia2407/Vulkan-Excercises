#include "image.hpp"

#include <utility>

Image::Image() noexcept = default;

Image::~Image()
{
	if( VK_NULL_HANDLE != image )
		vkDestroyImage( mDevice, image, nullptr );

	if( VK_NULL_HANDLE != memory )
		vkFreeMemory( mDevice, memory, nullptr );
}

Image::Image( VkDevice aDevice, VkImage aImage, VkDeviceMemory aMemory ) noexcept
	: image( aImage )
	, memory( aMemory )
	, mDevice( aDevice )
{}

Image::Image( Image&& aOther ) noexcept
	: image( std::exchange( aOther.image, VK_NULL_HANDLE ) )
	, memory( std::exchange( aOther.memory, VK_NULL_HANDLE ) )
	, mDevice( std::exchange( aOther.mDevice, VK_NULL_HANDLE ) )
{}

Image& Image::operator= (Image&& aOther) noexcept
{
	std::swap( image, aOther.image );
	std::swap( memory, aOther.memory );
	std::swap( mDevice, aOther.mDevice );
	return *this;
}

