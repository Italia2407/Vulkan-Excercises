#include "vertex_data.hpp"

#include <limits>

#include <cstring> // for std::memcpy()

#include "../labutils/error.hpp"
#include "../labutils/vkutil.hpp"
#include "../labutils/to_string.hpp"
namespace lut = labutils;



ColorizedMesh create_triangle_mesh( labutils::VulkanContext const& aContext, labutils::Allocator const& aAllocator )
{
	// Vertex data
	static float const positions[] = {
		 0.0f, -0.8f,
		-0.7f,  0.8f,
		 0.7f,  0.8f,
		 0.1f, -0.9f,
    	 0.5f, -0.1f,
    	 0.9f,  0.0f
	};
	static float const colors[] = {
		0.80f, 0.00f, 0.00f,
		0.00f, 0.80f, 0.00f,
		0.00f, 0.00f, 0.80f,
		0.25f, 1.00f, 1.00f,
		1.00f, 0.25f, 1.00f,
		1.00f, 1.00f, 0.25f
	};

	lut::Buffer vertexPositionGPU = lut::create_buffer(
		aAllocator,
		sizeof(positions),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY
	);
	lut::Buffer vertexColourGPU = lut::create_buffer(
		aAllocator,
		sizeof(colors),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY
	);

	lut::Buffer positionStaging = lut::create_buffer(
		aAllocator,
		sizeof(positions),
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_MEMORY_USAGE_CPU_TO_GPU
	);
	lut::Buffer colourStaging = lut::create_buffer(
		aAllocator,
		sizeof(colors),
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_MEMORY_USAGE_CPU_TO_GPU
	);

	void* positionPtr = nullptr;
	if (auto const res = vmaMapMemory(aAllocator.allocator, positionStaging.allocation, &positionPtr); res != VK_SUCCESS)
	{
		throw lut::Error("Mapping Memory for Writing\n"
			"vmaMapMemory() Returned %s", lut::to_string(res).c_str());
	}
	std::memcpy(positionPtr, positions, sizeof(positions));
	vmaUnmapMemory(aAllocator.allocator, positionStaging.allocation);

	void* colourPtr = nullptr;
	if (auto const res = vmaMapMemory(aAllocator.allocator, colourStaging.allocation, &colourPtr); res != VK_SUCCESS)
	{
		throw lut::Error("Mapping Memory for Writing\n"
			"vmaMapMemory() Returned %s", lut::to_string(res).c_str());
	}
	std::memcpy(colourPtr, colors, sizeof(colors));
	vmaUnmapMemory(aAllocator.allocator, colourStaging.allocation);

	lut::Fence uploadComplete = create_fence(aContext);

	lut::CommandPool uploadPool = create_command_pool(aContext);
	VkCommandBuffer uploadCommand = alloc_command_buffer(aContext, uploadPool.handle);

	VkCommandBufferBeginInfo commandBufferBeginInfo{}; {
		commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

		commandBufferBeginInfo.flags = 0;
		commandBufferBeginInfo.pInheritanceInfo = nullptr;
	}

	if (auto const res = vkBeginCommandBuffer(uploadCommand, &commandBufferBeginInfo); res != VK_SUCCESS)
	{
		throw lut::Error("Beginning Command Buffer Recording\n"
			"vkBeginCommandBuffer() Returned %s", lut::to_string(res).c_str());
	}

	VkBufferCopy positionsCopy{}; {
		positionsCopy.size = sizeof(positions);
	}

	vkCmdCopyBuffer(uploadCommand, positionStaging.buffer, vertexPositionGPU.buffer, 1, &positionsCopy);

	lut::buffer_barrier(
		uploadCommand,
		vertexPositionGPU.buffer,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
	);

	VkBufferCopy coloursCopy{}; {
		coloursCopy.size = sizeof(colors);
	}

	vkCmdCopyBuffer(uploadCommand, colourStaging.buffer, vertexColourGPU.buffer, 1, &coloursCopy);

	lut::buffer_barrier(
		uploadCommand,
		vertexColourGPU.buffer,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
	);

	if (auto const res = vkEndCommandBuffer(uploadCommand); res != VK_SUCCESS)
	{
		throw lut::Error("Ending Command Buffer Recording\n"
			"vkEndCommandBuffer() Returned %s", lut::to_string(res).c_str());
	}

	VkSubmitInfo submitInfo{}; {
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &uploadCommand;
	}

	if (auto const res = vkQueueSubmit(aContext.graphicsQueue, 1, &submitInfo, uploadComplete.handle); res != VK_SUCCESS)
	{
		throw lut::Error("Submitting Commands\n"
			"vkQueueSubmit() Returned %s", lut::to_string(res).c_str());
	}

	if (auto const res = vkWaitForFences(aContext.device, 1, &uploadComplete.handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); res != VK_SUCCESS)
	{
		throw lut::Error("Waiting for Upload to Complete\n"
			"vkWaitForFences() Returned %s", lut::to_string(res).c_str());
	}

	return ColorizedMesh{
		std::move(vertexPositionGPU),
		std::move(vertexColourGPU),
		(sizeof(positions) / sizeof(float)) / 2
	};
}

TexturedMesh create_plane_mesh(labutils::VulkanContext const& aContext, labutils::Allocator const& aAllocator)
{
	static float const positions[] = {
		-1.0f,  0.0f, -6.0f,	// v0
		-1.0f,  0.0f,  6.0f,	// v1
		 1.0f,  0.0f,  6.0f,	// v2

		-1.0f,  0.0f, -6.0f,	// v0
		 1.0f,  0.0f,  6.0f,	// v2
		 1.0f,  0.0f, -6.0f		// v3
	};

	static float const textureCoords[] = {
		0.0f, -6.0f,	// t0
		0.0f,  6.0f,	// t1
		1.0f,  6.0f,	// t2

		0.0f, -6.0f,	// t0
		1.0f,  6.0f,	// t2
		1.0f, -6.0f		// t3
	};

	lut::Buffer vertexPositionGPU = lut::create_buffer(
		aAllocator,
		sizeof(positions),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY
	);
	lut::Buffer vertexTextureCoordsGPU = lut::create_buffer(
		aAllocator,
		sizeof(textureCoords),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY
	);

	lut::Buffer positionStaging = lut::create_buffer(
		aAllocator,
		sizeof(positions),
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_MEMORY_USAGE_CPU_TO_GPU
	);
	lut::Buffer textureCoordsStaging = lut::create_buffer(
		aAllocator,
		sizeof(textureCoords),
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_MEMORY_USAGE_CPU_TO_GPU
	);

	void* positionPtr = nullptr;
	if (auto const res = vmaMapMemory(aAllocator.allocator, positionStaging.allocation, &positionPtr); res != VK_SUCCESS)
	{
		throw lut::Error("Mapping Memory for Writing\n"
			"vmaMapMemory() Returned %s", lut::to_string(res).c_str());
	}
	std::memcpy(positionPtr, positions, sizeof(positions));
	vmaUnmapMemory(aAllocator.allocator, positionStaging.allocation);

	void* textureCoordsPtr = nullptr;
	if (auto const res = vmaMapMemory(aAllocator.allocator, textureCoordsStaging.allocation, &textureCoordsPtr); res != VK_SUCCESS)
	{
		throw lut::Error("Mapping Memory for Writing\n"
			"vmaMapMemory() Returned %s", lut::to_string(res).c_str());
	}
	std::memcpy(textureCoordsPtr, textureCoords, sizeof(textureCoords));
	vmaUnmapMemory(aAllocator.allocator, textureCoordsStaging.allocation);

	lut::Fence uploadComplete = create_fence(aContext);

	lut::CommandPool uploadPool = create_command_pool(aContext);
	VkCommandBuffer uploadCommand = alloc_command_buffer(aContext, uploadPool.handle);

	VkCommandBufferBeginInfo commandBufferBeginInfo{}; {
		commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

		commandBufferBeginInfo.flags = 0;
		commandBufferBeginInfo.pInheritanceInfo = nullptr;
	}

	if (auto const res = vkBeginCommandBuffer(uploadCommand, &commandBufferBeginInfo); res != VK_SUCCESS)
	{
		throw lut::Error("Beginning Command Buffer Recording\n"
			"vkBeginCommandBuffer() Returned %s", lut::to_string(res).c_str());
	}

	VkBufferCopy positionsCopy{}; {
		positionsCopy.size = sizeof(positions);
	}

	vkCmdCopyBuffer(uploadCommand, positionStaging.buffer, vertexPositionGPU.buffer, 1, &positionsCopy);

	lut::buffer_barrier(
		uploadCommand,
		vertexPositionGPU.buffer,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
	);

	VkBufferCopy textureCoordsCopy{}; {
		textureCoordsCopy.size = sizeof(textureCoords);
	}

	vkCmdCopyBuffer(uploadCommand, textureCoordsStaging.buffer, vertexTextureCoordsGPU.buffer, 1, &textureCoordsCopy);

	lut::buffer_barrier(
		uploadCommand,
		vertexTextureCoordsGPU.buffer,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
	);

	if (auto const res = vkEndCommandBuffer(uploadCommand); res != VK_SUCCESS)
	{
		throw lut::Error("Ending Command Buffer Recording\n"
			"vkEndCommandBuffer() Returned %s", lut::to_string(res).c_str());
	}

	VkSubmitInfo submitInfo{}; {
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &uploadCommand;
	}

	if (auto const res = vkQueueSubmit(aContext.graphicsQueue, 1, &submitInfo, uploadComplete.handle); res != VK_SUCCESS)
	{
		throw lut::Error("Submitting Commands\n"
			"vkQueueSubmit() Returned %s", lut::to_string(res).c_str());
	}

	if (auto const res = vkWaitForFences(aContext.device, 1, &uploadComplete.handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); res != VK_SUCCESS)
	{
		throw lut::Error("Waiting for Upload to Complete\n"
			"vkWaitForFences() Returned %s", lut::to_string(res).c_str());
	}

	return TexturedMesh{
		std::move(vertexPositionGPU),
		std::move(vertexTextureCoordsGPU),
		(sizeof(positions) / sizeof(float)) / 3
	};
}
TexturedMesh create_sprite_mesh(labutils::VulkanContext const& aContext, labutils::Allocator const& aAllocator)
{
	// Vertex Data
	static float const positions[] = {
		-1.5f,  1.5f, -4.0f,	// v0
		-1.5f, -0.5f, -4.0f,	// v1
		 1.5f, -0.5f, -4.0f,	// v2

		-1.5f,  1.5f, -4.0f,	// v0
		 1.5f, -0.5f, -4.0f,	// v2
		 1.5f,  1.5f, -4.0f		// v3
	};

	static float const textureCoords[] = {
		0.0f, 1.0f,		// t0
		0.0f, 0.0f,		// t1
		1.0f, 0.0f,		// t2

		0.0f, 1.0f,		// t0
		1.0f, 0.0f,		// t2
		1.0f, 1.0f		// t3
	};

	lut::Buffer vertexPositionGPU = lut::create_buffer(
		aAllocator,
		sizeof(positions),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY
	);
	lut::Buffer vertexTextureCoordsGPU = lut::create_buffer(
		aAllocator,
		sizeof(textureCoords),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY
	);

	lut::Buffer positionStaging = lut::create_buffer(
		aAllocator,
		sizeof(positions),
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_MEMORY_USAGE_CPU_TO_GPU
	);
	lut::Buffer textureCoordsStaging = lut::create_buffer(
		aAllocator,
		sizeof(textureCoords),
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_MEMORY_USAGE_CPU_TO_GPU
	);

	void* positionPtr = nullptr;
	if (auto const res = vmaMapMemory(aAllocator.allocator, positionStaging.allocation, &positionPtr); res != VK_SUCCESS)
	{
		throw lut::Error("Mapping Memory for Writing\n"
			"vmaMapMemory() Returned %s", lut::to_string(res).c_str());
	}
	std::memcpy(positionPtr, positions, sizeof(positions));
	vmaUnmapMemory(aAllocator.allocator, positionStaging.allocation);

	void* textureCoordsPtr = nullptr;
	if (auto const res = vmaMapMemory(aAllocator.allocator, textureCoordsStaging.allocation, &textureCoordsPtr); res != VK_SUCCESS)
	{
		throw lut::Error("Mapping Memory for Writing\n"
			"vmaMapMemory() Returned %s", lut::to_string(res).c_str());
	}
	std::memcpy(textureCoordsPtr, textureCoords, sizeof(textureCoords));
	vmaUnmapMemory(aAllocator.allocator, textureCoordsStaging.allocation);

	lut::Fence uploadComplete = create_fence(aContext);

	lut::CommandPool uploadPool = create_command_pool(aContext);
	VkCommandBuffer uploadCommand = alloc_command_buffer(aContext, uploadPool.handle);

	VkCommandBufferBeginInfo commandBufferBeginInfo{}; {
		commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

		commandBufferBeginInfo.flags = 0;
		commandBufferBeginInfo.pInheritanceInfo = nullptr;
	}

	if (auto const res = vkBeginCommandBuffer(uploadCommand, &commandBufferBeginInfo); res != VK_SUCCESS)
	{
		throw lut::Error("Beginning Command Buffer Recording\n"
			"vkBeginCommandBuffer() Returned %s", lut::to_string(res).c_str());
	}

	VkBufferCopy positionsCopy{}; {
		positionsCopy.size = sizeof(positions);
	}

	vkCmdCopyBuffer(uploadCommand, positionStaging.buffer, vertexPositionGPU.buffer, 1, &positionsCopy);

	lut::buffer_barrier(
		uploadCommand,
		vertexPositionGPU.buffer,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
	);

	VkBufferCopy textureCoordsCopy{}; {
		textureCoordsCopy.size = sizeof(textureCoords);
	}

	vkCmdCopyBuffer(uploadCommand, textureCoordsStaging.buffer, vertexTextureCoordsGPU.buffer, 1, &textureCoordsCopy);

	lut::buffer_barrier(
		uploadCommand,
		vertexTextureCoordsGPU.buffer,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
	);

	if (auto const res = vkEndCommandBuffer(uploadCommand); res != VK_SUCCESS)
	{
		throw lut::Error("Ending Command Buffer Recording\n"
			"vkEndCommandBuffer() Returned %s", lut::to_string(res).c_str());
	}

	VkSubmitInfo submitInfo{}; {
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &uploadCommand;
	}

	if (auto const res = vkQueueSubmit(aContext.graphicsQueue, 1, &submitInfo, uploadComplete.handle); res != VK_SUCCESS)
	{
		throw lut::Error("Submitting Commands\n"
			"vkQueueSubmit() Returned %s", lut::to_string(res).c_str());
	}

	if (auto const res = vkWaitForFences(aContext.device, 1, &uploadComplete.handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); res != VK_SUCCESS)
	{
		throw lut::Error("Waiting for Upload to Complete\n"
			"vkWaitForFences() Returned %s", lut::to_string(res).c_str());
	}

	return TexturedMesh{
		std::move(vertexPositionGPU),
		std::move(vertexTextureCoordsGPU),
		(sizeof(positions) / sizeof(float)) / 3
	};
}