#include "vkimage.hpp"

#include <limits>
#include <vector>
#include <utility>
#include <algorithm>

#include <cstdio>
#include <cassert>
#include <cstring> // for std::memcpy()

#include <stb_image.h>

#include "error.hpp"
#include "vkutil.hpp"
#include "vkbuffer.hpp"
#include "to_string.hpp"

namespace
{
	// Unfortunately, std::countl_zero() isn't available in C++17; it was added
	// in C++20. This provides a fallback implementation. Unlike C++20, this
	// returns a std::uint32_t and not a signed int.
	//
	// See https://graphics.stanford.edu/~seander/bithacks.html for this and
	// other methods like it.
	//
	// Note: that this is unlikely to be the most efficient implementation on
	// most processors. Many instruction sets have dedicated instructions for
	// this operation. E.g., lzcnt (x86 ABM/BMI), bsr (x86).
	inline 
	std::uint32_t countl_zero_( std::uint32_t aX )
	{
		if( !aX ) return 32;

		uint32_t res = 0;

		if( !(aX & 0xffff0000) ) (res += 16), (aX <<= 16);
		if( !(aX & 0xff000000) ) (res +=  8), (aX <<=  8);
		if( !(aX & 0xf0000000) ) (res +=  4), (aX <<=  4);
		if( !(aX & 0xc0000000) ) (res +=  2), (aX <<=  2);
		if( !(aX & 0x80000000) ) (res +=  1);

		return res;
	}
}

namespace labutils
{
	Image::Image() noexcept = default;

	Image::~Image()
	{
		if( VK_NULL_HANDLE != image )
		{
			assert( VK_NULL_HANDLE != mAllocator );
			assert( VK_NULL_HANDLE != allocation );
			vmaDestroyImage( mAllocator, image, allocation );
		}
	}

	Image::Image( VmaAllocator aAllocator, VkImage aImage, VmaAllocation aAllocation ) noexcept
		: image( aImage )
		, allocation( aAllocation )
		, mAllocator( aAllocator )
	{}

	Image::Image( Image&& aOther ) noexcept
		: image( std::exchange( aOther.image, VK_NULL_HANDLE ) )
		, allocation( std::exchange( aOther.allocation, VK_NULL_HANDLE ) )
		, mAllocator( std::exchange( aOther.mAllocator, VK_NULL_HANDLE ) )
	{}
	Image& Image::operator=( Image&& aOther ) noexcept
	{
		std::swap( image, aOther.image );
		std::swap( allocation, aOther.allocation );
		std::swap( mAllocator, aOther.mAllocator );
		return *this;
	}
}

namespace labutils
{
Image load_image_texture2d( char const* aPath, VulkanContext const& aContext, VkCommandPool aCmdPool, Allocator const& aAllocator )
{
	stbi_set_flip_vertically_on_load(true);

	int inBaseWidth, inBaseHeight, inBaseChannels;
	stbi_uc* imageData = stbi_load(aPath, &inBaseWidth, &inBaseHeight, &inBaseChannels, 4);
	if (!imageData)
	{
		throw Error("%s: Unable to Load Texture Base Image (%s)",
			aPath, 0, stbi_failure_reason());
	}

	auto const baseWidth = std::uint32_t(inBaseWidth);
	auto const baseHeight = std::uint32_t(inBaseHeight);

	auto const bytesSize = baseWidth * baseHeight * 4;

	auto staging = create_buffer(aAllocator, bytesSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

	void* sptr = nullptr;
	if (auto const res = vmaMapMemory(aAllocator.allocator, staging.allocation, &sptr);
		res != VK_SUCCESS)
	{
		throw Error("Mapping Memory for Writing\n"
			"vmaMapMemory() Returned %s", to_string(res).c_str());
	}

	std::memcpy(sptr, imageData, bytesSize);
	vmaUnmapMemory(aAllocator.allocator, staging.allocation);

	stbi_image_free(imageData);

	Image image = create_image_texture2d(
		aAllocator, baseWidth, baseHeight,
		VK_FORMAT_R8G8B8A8_SRGB,
		VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
	);

	VkCommandBufferBeginInfo commandBufferBeginInfo{}; {
		commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

		commandBufferBeginInfo.flags = 0;
		commandBufferBeginInfo.pInheritanceInfo = nullptr;
	}

	VkCommandBuffer commandBuffer = alloc_command_buffer(aContext, aCmdPool);
	if (auto const res = vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
		res != VK_SUCCESS)
	{
		throw Error("Beginning Command Buffer Recording\n"
			"vkBeginCommandBuffer() Returned %s", to_string(res).c_str());
	}

	auto const mipLevels = compute_mip_level_count(baseWidth, baseHeight);

	image_barrier(
		commandBuffer,
		image.image,
		0, VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
		VkImageSubresourceRange{
			VK_IMAGE_ASPECT_COLOR_BIT,
			0, mipLevels,
			0, 1
		}
	);

	VkBufferImageCopy imageCopy; {
		imageCopy.bufferOffset = 0;

		imageCopy.bufferRowLength = 0;
		imageCopy.bufferImageHeight = 0;

		imageCopy.imageSubresource = VkImageSubresourceLayers{
			VK_IMAGE_ASPECT_COLOR_BIT,
			0,
			0, 1
		};

		imageCopy.imageOffset = VkOffset3D{0, 0, 0};
		imageCopy.imageExtent = VkExtent3D{baseWidth, baseHeight, 1};
	}

	vkCmdCopyBufferToImage(commandBuffer, staging.buffer, image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL ,1, &imageCopy);

	image_barrier(
		commandBuffer,
		image.image,
		VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
		VkImageSubresourceRange{
			VK_IMAGE_ASPECT_COLOR_BIT,
			0, 1,
			0, 1
		}
	);

	std::uint32_t width = baseWidth;
	std::uint32_t height = baseHeight;

	for (std::uint32_t mipLevel = 1; mipLevel < mipLevels; mipLevel++)
	{
		VkImageBlit imageBlit{}; {
			imageBlit.srcSubresource = VkImageSubresourceLayers{
				VK_IMAGE_ASPECT_COLOR_BIT,
				mipLevel - 1,
				0, 1
			};

			imageBlit.srcOffsets[0] = {0, 0, 0};
			imageBlit.srcOffsets[1] = {std::int32_t(width), std::int32_t(height), 1};

			width >>= 1; if (width == 0) width = 1;
			height >>= 1; if (height == 0) height = 1;

			imageBlit.dstSubresource = VkImageSubresourceLayers{
				VK_IMAGE_ASPECT_COLOR_BIT,
				mipLevel,
				0, 1
			};

			imageBlit.dstOffsets[0] = {0, 0, 0};
			imageBlit.dstOffsets[1] = {std::int32_t(width), std::int32_t(height), 1};
		}

		vkCmdBlitImage(
			commandBuffer,
			image.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1, &imageBlit,
			VK_FILTER_LINEAR
		);

		image_barrier(
			commandBuffer,
			image.image,
			VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
			VkImageSubresourceRange{
				VK_IMAGE_ASPECT_COLOR_BIT,
				mipLevel, 1,
				0, 1
			}
		);
	}

	image_barrier(
		commandBuffer,
		image.image,
		VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
		VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		VkImageSubresourceRange{
			VK_IMAGE_ASPECT_COLOR_BIT,
			0, mipLevels,
			0, 1
		}
	);

	if (auto const res = vkEndCommandBuffer(commandBuffer);
		res != VK_SUCCESS)
	{
		throw Error("Ending Command Buffer Recording\n"
			"vkEndCOmmandBuffer() Returned %s", to_string(res).c_str());
	}

	Fence uploadComplete = create_fence(aContext);

	VkSubmitInfo submitInfo{}; {
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;
	}

	if (auto const res = vkQueueSubmit(aContext.graphicsQueue, 1, &submitInfo, uploadComplete.handle);
		res != VK_SUCCESS)
	{
		throw Error("Submitting Commands\n"
			"vkQueueSubmit() Returned %s", to_string(res).c_str());
	}

	if (auto const res = vkWaitForFences(aContext.device, 1, &uploadComplete.handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max());
		res != VK_SUCCESS)
	{
		throw Error("Waiting for Upload to Complete\n"
			"vkWaitForFences() Returned %s", to_string(res).c_str());
	}

	vkFreeCommandBuffers(aContext.device, aCmdPool, 1, &commandBuffer);

	return image;

	throw Error( "Not yet implemented" ); //TODO- (Section 4) implement me!
}
Image create_image_texture2d( Allocator const& aAllocator, std::uint32_t aWidth, std::uint32_t aHeight, VkFormat aFormat, VkImageUsageFlags aUsage )
{
	auto const mipLevels = compute_mip_level_count(aWidth, aHeight);

	VkImageCreateInfo imageInfo{}; {
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;

		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.format = aFormat;

		imageInfo.extent.width = aWidth;
		imageInfo.extent.height = aHeight;
		imageInfo.extent.depth = 1;

		imageInfo.mipLevels = mipLevels;
		imageInfo.arrayLayers = 1;

		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;

		imageInfo.usage = aUsage;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	}

	VmaAllocationCreateInfo allocationInfo{}; {
		allocationInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	}

	VkImage image = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	if (auto const res = vmaCreateImage(aAllocator.allocator, &imageInfo, &allocationInfo, &image, &allocation, nullptr);
		res != VK_SUCCESS)
	{
		throw Error("Unable to Allocate Image\n"
			"vmaCreateImage() Returned %s", to_string(res).c_str());
	}

	return Image(aAllocator.allocator, image, allocation);
}

	std::uint32_t compute_mip_level_count( std::uint32_t aWidth, std::uint32_t aHeight )
	{
		std::uint32_t const bits = aWidth | aHeight;
		std::uint32_t const leadingZeros = countl_zero_( bits );
		return 32-leadingZeros;
	}
}
