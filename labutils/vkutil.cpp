#include "vkutil.hpp"

#include <vector>

#include <cstdio>
#include <cassert>

#include "error.hpp"
#include "to_string.hpp"

namespace labutils
{
ShaderModule load_shader_module( VulkanContext const& aContext, char const* aSpirvPath )
{
	assert(aSpirvPath);

	std::FILE* fileIn = std::fopen(aSpirvPath, "rb");
	if (!fileIn)
	{
		throw Error("Cannot Open '%s' for Reading", aSpirvPath);
	}

	std::fseek(fileIn, 0, SEEK_END);
	auto const numBytes = std::size_t(std::ftell(fileIn));
	std::fseek(fileIn, 0, SEEK_SET);

	assert((numBytes % 4) == 0);
	auto const numWords = numBytes / 4;

	std::vector<std::uint32_t> code(numWords);

	std::size_t offset = 0;
	while (offset != numWords)
	{
		auto const read = std::fread(code.data() + offset, sizeof(std::uint32_t), numWords - offset, fileIn);
		if (read == 0)
		{
			std::fclose(fileIn);

			throw Error("Error Reading '%s': ferror = %d, feof = %d", aSpirvPath, std::ferror(fileIn), std::feof(fileIn));
		}

		offset += read;
	}

	std::fclose(fileIn);

	VkShaderModuleCreateInfo shaderModuleInfo{}; {
		shaderModuleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		shaderModuleInfo.codeSize = numBytes;
		shaderModuleInfo.pCode = code.data();
	}

	VkShaderModule shaderModule = VK_NULL_HANDLE;
	if(auto const res = vkCreateShaderModule( aContext.device, &shaderModuleInfo, nullptr, &shaderModule); res != VK_SUCCESS)
	{
		throw Error("Unable to Create Shader Module from %s\n"
			"vkCreateShaderModule() Returned %s", aSpirvPath, to_string(res).c_str());
	}

	return ShaderModule(aContext.device, shaderModule);
}

CommandPool create_command_pool( VulkanContext const& aContext, VkCommandPoolCreateFlags aFlags )
{
	VkCommandPoolCreateInfo commandPoolInfo{}; {
		commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;

		commandPoolInfo.queueFamilyIndex = aContext.graphicsFamilyIndex;
		commandPoolInfo.flags = aFlags;
	}

	VkCommandPool commandPool = VK_NULL_HANDLE;
	if (auto const res = vkCreateCommandPool(aContext.device, &commandPoolInfo, nullptr, &commandPool); res != VK_SUCCESS)
	{
		throw Error("Unable to Create Command Pool\n"
			"vkCreateCommandPool() Returned %s", to_string(res).c_str());
	}

	return CommandPool(aContext.device, commandPool);
}
VkCommandBuffer alloc_command_buffer( VulkanContext const& aContext, VkCommandPool aCmdPool )
{
	VkCommandBufferAllocateInfo commandBufferInfo{}; {
		commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;

		commandBufferInfo.commandPool = aCmdPool;
		commandBufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		commandBufferInfo.commandBufferCount = 1;
	}

	VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
	if (auto const res = vkAllocateCommandBuffers(aContext.device, &commandBufferInfo, &commandBuffer); res != VK_SUCCESS)
	{
		throw Error("Unable to Allocate Command Buffer\n"
			"vkAllocateCommandBuffers() Returned %s", to_string(res).c_str());
	}

	return commandBuffer;
}

Fence create_fence( VulkanContext const& aContext, VkFenceCreateFlags aFlags )
{
	VkFenceCreateInfo fenceInfo{}; {
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

		fenceInfo.flags = aFlags;
	}

	VkFence fence = VK_NULL_HANDLE;
	if (auto const res = vkCreateFence(aContext.device, &fenceInfo, nullptr, &fence); res != VK_SUCCESS)
	{
		throw Error("Unable to Create Fence\n"
			"vkCreateFence() Returned %s", to_string(res).c_str());
	}

	return Fence(aContext.device, fence);
}
Semaphore create_semaphore( VulkanContext const& aContext )
{
	VkSemaphoreCreateInfo semaphoreInfo{}; {
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	}

	VkSemaphore semaphore = VK_NULL_HANDLE;
	if (auto const res = vkCreateSemaphore(aContext.device, &semaphoreInfo, nullptr, &semaphore); res != VK_SUCCESS)
	{
		throw Error("Unable to Create Semaphore\n"
			"vkCreateSemaphore() Returned %s", to_string(res).c_str());
	}

	return Semaphore(aContext.device, semaphore);
}

void buffer_barrier(
	VkCommandBuffer aCommandBuffer, VkBuffer aBuffer,
	VkAccessFlags aSrcAccessMask, VkAccessFlags aDstAccessMask,
	VkPipelineStageFlags aSrcStageMask, VkPipelineStageFlags aDstStageMask,
	VkDeviceSize aSize, VkDeviceSize aOffset,
	uint32_t aSrcQueueFamilyIndex,
	uint32_t aDstQueueFamilyIndex)
{
	VkBufferMemoryBarrier bufferMemoryBarrier{}; {
		bufferMemoryBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;

		bufferMemoryBarrier.srcAccessMask = aSrcAccessMask;
		bufferMemoryBarrier.dstAccessMask = aDstAccessMask;

		bufferMemoryBarrier.buffer = aBuffer;
		bufferMemoryBarrier.size = aSize;
		bufferMemoryBarrier.offset = aOffset;

		bufferMemoryBarrier.srcQueueFamilyIndex = aSrcQueueFamilyIndex;
		bufferMemoryBarrier.dstQueueFamilyIndex = aDstQueueFamilyIndex;
	}

	vkCmdPipelineBarrier(
		aCommandBuffer,
		aSrcStageMask, aDstStageMask,
		0,
		0, nullptr,
		1, &bufferMemoryBarrier,
		0, nullptr
	);
}
void image_barrier(
	VkCommandBuffer aCmdBuff, VkImage aImage,
	VkAccessFlags aSrcAccessMask, VkAccessFlags aDstAccessMask,
	VkImageLayout aSrcLayout, VkImageLayout aDstLayout,
	VkPipelineStageFlags aSrcStageMask, VkPipelineStageFlags aDstStageMask,
	VkImageSubresourceRange aRange,
	std::uint32_t aSrcQueueFamilyIndex,
	std::uint32_t aDstQueueFamilyIndex)
{
	VkImageMemoryBarrier imageMemoryBarrier{}; {
		imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;

		imageMemoryBarrier.image = aImage;

		imageMemoryBarrier.srcAccessMask = aSrcAccessMask;
		imageMemoryBarrier.dstAccessMask = aDstAccessMask;

		imageMemoryBarrier.srcQueueFamilyIndex = aSrcQueueFamilyIndex;
		imageMemoryBarrier.dstQueueFamilyIndex = aDstQueueFamilyIndex;

		imageMemoryBarrier.oldLayout = aSrcLayout;
		imageMemoryBarrier.newLayout = aDstLayout;
		imageMemoryBarrier.subresourceRange = aRange;
	}

	vkCmdPipelineBarrier(aCmdBuff, aSrcStageMask, aDstStageMask, 0, 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);
}

DescriptorPool create_descriptor_pool(
	VulkanContext const& aContext,
	std::uint32_t aMaxDescriptors, std::uint32_t aMaxSets)
{
	VkDescriptorPoolSize const descriptorPoolSizes[] = {
		{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, aMaxDescriptors},
		{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, aMaxDescriptors}
	};

	VkDescriptorPoolCreateInfo descriptorPoolInfo{}; {
		descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;

		descriptorPoolInfo.maxSets = aMaxSets;
		descriptorPoolInfo.poolSizeCount = sizeof(descriptorPoolSizes) / sizeof(descriptorPoolSizes[0]);
		descriptorPoolInfo.pPoolSizes = descriptorPoolSizes;
	}

	VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
	if (auto const res = vkCreateDescriptorPool(aContext.device, &descriptorPoolInfo, nullptr, &descriptorPool);
		res != VK_SUCCESS)
	{
		throw Error("Unable to Create Descriptor Pool\n"
			"vkCreateDescriptorPool() Returned %s", to_string(res).c_str());
	}

	return DescriptorPool(aContext.device, descriptorPool);
}
VkDescriptorSet alloc_desc_set(
	VulkanContext const& aContext,
	VkDescriptorPool aDescriptorPool, VkDescriptorSetLayout aDescriptorSetLayout)
{
	VkDescriptorSetAllocateInfo descriptorSetAllocateInfo{}; {
		descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;

		descriptorSetAllocateInfo.descriptorPool = aDescriptorPool;

		descriptorSetAllocateInfo.descriptorSetCount = 1;
		descriptorSetAllocateInfo.pSetLayouts = &aDescriptorSetLayout;
	}

	VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
	if (auto const res = vkAllocateDescriptorSets(aContext.device, &descriptorSetAllocateInfo, &descriptorSet);
		res != VK_SUCCESS)
	{
		throw Error("Unable to Allocate Descriptor Set\n"
			"vkAllocateDescriptorSets() Returned %s", to_string(res).c_str());
	}

	return descriptorSet;
}

ImageView create_image_view_texture2d(VulkanContext const& aContext, VkImage aImage, VkFormat aFormat)
{
	VkImageViewCreateInfo imageViewInfo{}; {
		imageViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;

		imageViewInfo.image = aImage;
		imageViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		imageViewInfo.format = aFormat;

		imageViewInfo.components = VkComponentMapping{};
		imageViewInfo.subresourceRange = VkImageSubresourceRange{
			VK_IMAGE_ASPECT_COLOR_BIT,
			0, VK_REMAINING_MIP_LEVELS,
			0, 1
		};
	}

	VkImageView imageView = VK_NULL_HANDLE;
	if (auto const res = vkCreateImageView(aContext.device, &imageViewInfo, nullptr, &imageView);
		res != VK_SUCCESS)
	{
		throw Error("Unable to Create Image View\n"
			"vkCreateImageView() Returned %s", to_string(res).c_str());
	}

	return ImageView(aContext.device, imageView);
}

Sampler create_default_sampler(VulkanContext const& aContext)
{
	VkSamplerCreateInfo samplerInfo{}; {
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;

		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;

		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;

		samplerInfo.minLod = 0.0f;
		samplerInfo.maxLod = VK_LOD_CLAMP_NONE;
		samplerInfo.mipLodBias = 0.0f;

		// Check if Anisotropic Filtering is Supported
		VkPhysicalDeviceFeatures deviceFeatures{};
		vkGetPhysicalDeviceFeatures(aContext.physicalDevice, &deviceFeatures);
		
		samplerInfo.anisotropyEnable = deviceFeatures.samplerAnisotropy;
		samplerInfo.maxAnisotropy = 8.0f;
	}

	VkSampler sampler = VK_NULL_HANDLE;
	if (auto const res = vkCreateSampler(aContext.device, &samplerInfo, nullptr, &sampler);
		res != VK_SUCCESS)
	{
		throw Error("Unable to Create Sampler\n"
			"vkCreateSampler() Returned %s", to_string(res).c_str());
	}

	return Sampler(aContext.device, sampler);
}
}
