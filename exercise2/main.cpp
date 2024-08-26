#include <volk/volk.h>

#include <tuple>
#include <limits>
#include <vector>
#include <stdexcept>

#include <cstdio>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <stb_image_write.h>

#include "../labutils/error.hpp"
#include "../labutils/vkutil.hpp"
#include "../labutils/vkobject.hpp"
#include "../labutils/to_string.hpp"
#include "../labutils/vulkan_context.hpp"
namespace lut = labutils;

#include "image.hpp"
#include "buffer.hpp"


namespace
{
namespace cfg
{
	// Image format:
	// Vulkan implementation do not have to support all image formats
	// listed in the VkFormat enum. Rather, some formats may only be used
	// with some operations. We are rendering (without blending!), so our
	// use case is VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT (with blending it
	// would be VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT). We need a
	// format that supports VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT (with
	// optimal tiling). Vulkan requires some formats to be supported - see
	// https://www.khronos.org/registry/vulkan/specs/1.3/html/chap34.html#features-required-format-support
	//
	// It turns out that support there are no mandatory formats listed when
	// rendering with blending disabled. However, it seems that commonly
	// allow R8G8B8A8_SRGB for this use case anyway:
	// http://vulkan.gpuinfo.org/listoptimaltilingformats.php
	constexpr VkFormat kImageFormat = VK_FORMAT_R8G8B8A8_SRGB;

	// Image size and related parameters
	constexpr std::uint32_t kImageWidth = 1280;
	constexpr std::uint32_t kImageHeight = 720;
	constexpr std::uint32_t kImageSize = kImageWidth * kImageHeight * 4; // RGBA

	// Output image path
	constexpr char const* kImageOutput = "output.png";

	// Compiled shader code for the graphics pipeline
	// See sources in exercise2/shaders/*. 
#	define SHADERDIR_ "assets/exercise2/shaders/"
	constexpr char const* kVertShaderPath = SHADERDIR_ "triangle.vert.spv";
	constexpr char const* kFragShaderPath = SHADERDIR_ "triangle.frag.spv";
#	undef SHADERDIR_
}

// Helpers:
lut::RenderPass create_render_pass( lut::VulkanContext const& );

lut::PipelineLayout create_triangle_pipeline_layout( lut::VulkanContext const& );
lut::Pipeline create_triangle_pipeline( lut::VulkanContext const&, VkRenderPass, VkPipelineLayout );


std::tuple<Image,lut::ImageView> create_framebuffer_image( lut::VulkanContext const& );
lut::Framebuffer create_framebuffer( lut::VulkanContext const&, VkRenderPass, VkImageView );

Buffer create_download_buffer( lut::VulkanContext const& );

void record_commands( 
	VkCommandBuffer,
	VkRenderPass,
	VkFramebuffer,
	VkPipeline,
	VkImage,
	VkBuffer
);
void submit_commands(
	lut::VulkanContext const&,
	VkCommandBuffer,
	VkFence
);

std::uint32_t find_memory_type( lut::VulkanContext const&, std::uint32_t aMemoryTypeBits, VkMemoryPropertyFlags );
}

int main() try
{
	// Create the Vulkan instance, set up the validation, select a physical
	// device and instantiate a logical device from the selected device.
	// Request a single GRAPHICS queue for now, and fetch this from the created
	// logical device.
	//
	// See Exercise 1.1 for a detailed walkthrough of this process.
	lut::VulkanContext context = lut::make_vulkan_context();

	//TODO-maybe: experiment with SAMPLE_COUNT != 1
	
	// To render an image, we need a number of Vulkan resources. The following
	// creates these:
	lut::RenderPass renderPass = create_render_pass( context );

	lut::PipelineLayout pipeLayout = create_triangle_pipeline_layout( context );
	lut::Pipeline pipe = create_triangle_pipeline( context, renderPass.handle, pipeLayout.handle );

	auto [fbImage, fbImageView] = create_framebuffer_image( context );
	lut::Framebuffer framebuffer = create_framebuffer( context, renderPass.handle, fbImageView.handle );

	Buffer dlBuffer = create_download_buffer( context );

	lut::CommandPool cpool = lut::create_command_pool( context );
	VkCommandBuffer cbuffer = lut::alloc_command_buffer( context, cpool.handle );

	lut::Fence fence = lut::create_fence( context );

	// Now that we have set up the necessary resources, we can use our Vulkan
	// device to render the image. This happens in two steps:
	//  
	// 1. Record rendering commands in to the command buffer that we've 
	//    created for this purpose.
	// 2. Submit the command buffer to the Vulkan device / GPU for processing.

	record_commands(
		cbuffer,
		renderPass.handle,
		framebuffer.handle,
		pipe.handle,
		fbImage.image,
		dlBuffer.buffer
	);

	submit_commands(
		context,
		cbuffer,
		fence.handle
	);

	// Commands are executed asynchronously. Before we can access the result,
	// we need to wait for the processing to finish. The fence that we passed
	// to VkQueueSubmit() will become signalled when the command buffer has
	// finished processing -- we will wait for that to occur with
	// vkWaitForFences().

	// Wait for commands to finish executing
	constexpr std::uint64_t kMaxWait = std::numeric_limits<std::uint64_t>::max();

	if (auto const res = vkWaitForFences(context.device, 1, &fence.handle, VK_TRUE, kMaxWait); res != VK_SUCCESS)
	{
		throw lut::Error("Waiting for Fence\n"
			"vkWaitForFences() returned %s", lut::to_string(res).c_str());
	}

	// Access image and write it to disk.
	void* dataPtr = nullptr;
	if (auto const res = vkMapMemory(context.device, dlBuffer.memory, 0, cfg::kImageSize, 0, &dataPtr); res != VK_SUCCESS)
	{
		throw lut::Error("Mapping Memory\n"
			"vkMapMemory() Returned %s", lut::to_string(res).c_str());
	}

	assert(dataPtr);

	std::vector<std::byte> buffer(cfg::kImageSize);
	std::memcpy(buffer.data(), dataPtr, cfg::kImageSize);

	vkUnmapMemory(context.device, dlBuffer.memory);

	if (!stbi_write_png(cfg::kImageOutput, cfg::kImageWidth, cfg::kImageHeight, 4, buffer.data(), cfg::kImageWidth * 4))
	{
		throw lut::Error("Unable to Write Image: stbi_write_png() Returned Error");
	}

	// Cleanup
	// None required :-) The C++ wrappers take care of destroying the various
	// objects as the wrappers go out of scpe at the end of main()!

	return 0;
}
catch( std::exception const& eErr )
{
	std::fprintf( stderr, "Exception: %s\n", eErr.what() );
	return 1;
}


namespace
{
lut::RenderPass create_render_pass( lut::VulkanContext const& aContext )
{
	VkAttachmentDescription attachments[1]{}; {
		attachments[0].format = cfg::kImageFormat;
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	}
	
	VkAttachmentReference subpassAttachments[1]{}; {
		subpassAttachments[0].attachment = 0;
		subpassAttachments[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	}
	VkSubpassDescription subpasses[1]{}; {
		subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[0].colorAttachmentCount = 1;
		subpasses[0].pColorAttachments = subpassAttachments;
	}

	VkSubpassDependency dependencies[1]{}; {
		dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		
		dependencies[0].srcSubpass = 0;
		dependencies[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		
		dependencies[0].dstSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		dependencies[0].dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
	}

	VkRenderPassCreateInfo renderPassInfo{}; {
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;

		renderPassInfo.attachmentCount = 1;
		renderPassInfo.pAttachments = attachments;

		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = subpasses;

		renderPassInfo.dependencyCount = 1;
		renderPassInfo.pDependencies = dependencies;
	}
	
	VkRenderPass renderPass = VK_NULL_HANDLE;
	if (auto const res = vkCreateRenderPass(aContext.device, &renderPassInfo, nullptr, &renderPass); res != VK_SUCCESS)
	{
		throw lut::Error("Unable to Create Render Pass\n"
			"vkCreateRenderPass() returned %s", lut::to_string(res).c_str());
	}

	return lut::RenderPass(aContext.device, renderPass);
}

lut::PipelineLayout create_triangle_pipeline_layout( lut::VulkanContext const& aContext )
{
	VkPipelineLayoutCreateInfo pipelineLayoutInfo{}; {
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

		pipelineLayoutInfo.setLayoutCount = 0;
		pipelineLayoutInfo.pSetLayouts = nullptr;

		pipelineLayoutInfo.pushConstantRangeCount = 0;
		pipelineLayoutInfo.pPushConstantRanges = nullptr;
	}

	VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
	if (auto const res = vkCreatePipelineLayout(aContext.device, &pipelineLayoutInfo, nullptr, &pipelineLayout); res != VK_SUCCESS)
	{
		throw lut::Error("Unable to Create Pipeline Layout\n"
			"vkCreatePipelineLayout() returned %s", lut::to_string(res).c_str());
	}

	return lut::PipelineLayout(aContext.device, pipelineLayout);
}
lut::Pipeline create_triangle_pipeline( lut::VulkanContext const& aContext, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout )
{
	lut::ShaderModule vertShader = lut::load_shader_module(aContext, cfg::kVertShaderPath);
	lut::ShaderModule fragShader = lut::load_shader_module(aContext, cfg::kFragShaderPath);

	VkPipelineShaderStageCreateInfo shaderStagesInfo[2]{}; {
		shaderStagesInfo[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shaderStagesInfo[0].pName = "main";

		shaderStagesInfo[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		shaderStagesInfo[0].module = vertShader.handle;

		shaderStagesInfo[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shaderStagesInfo[1].pName = "main";

		shaderStagesInfo[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderStagesInfo[1].module = fragShader.handle;
	}

	VkPipelineVertexInputStateCreateInfo vertexInputInfo{}; {
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	}
	VkPipelineInputAssemblyStateCreateInfo assemblyStateInfo{}; {
		assemblyStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;

		assemblyStateInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyStateInfo.primitiveRestartEnable = VK_FALSE;
	}
	
	VkViewport viewport{}; {
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		
		viewport.width = float(cfg::kImageWidth);
		viewport.height = float(cfg::kImageHeight);

		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
	}
	VkRect2D scissor{}; {
		scissor.offset = VkOffset2D{0, 0};
		scissor.extent = VkExtent2D{cfg::kImageWidth, cfg::kImageHeight};
	}
	VkPipelineViewportStateCreateInfo viewportStateInfo{}; {
		viewportStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;

		viewportStateInfo.viewportCount = 1;
		viewportStateInfo.pViewports = &viewport;

		viewportStateInfo.scissorCount = 1;
		viewportStateInfo.pScissors = &scissor;
	}
	
	VkPipelineRasterizationStateCreateInfo rasterizationStateInfo{}; {
		rasterizationStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;

		rasterizationStateInfo.depthClampEnable = VK_FALSE;
		rasterizationStateInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterizationStateInfo.depthBiasEnable = VK_FALSE;

		rasterizationStateInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizationStateInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizationStateInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

		rasterizationStateInfo.lineWidth = 1.0f;
	}
	
	VkPipelineMultisampleStateCreateInfo multisamplingStateInfo{}; {
		multisamplingStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;

		multisamplingStateInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	}
	
	VkPipelineColorBlendAttachmentState colourBlendAttachmentStates[1]{}; {
		colourBlendAttachmentStates[0].blendEnable = VK_FALSE;
		colourBlendAttachmentStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	}
	VkPipelineColorBlendStateCreateInfo colourBlendStateInfo{}; {
		colourBlendStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;

		colourBlendStateInfo.logicOpEnable = VK_FALSE;

		colourBlendStateInfo.attachmentCount = 1;
		colourBlendStateInfo.pAttachments = colourBlendAttachmentStates;
	}
	
	VkGraphicsPipelineCreateInfo graphicsPipelineInfo{}; {
		graphicsPipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

		graphicsPipelineInfo.stageCount = 2;
		graphicsPipelineInfo.pStages = shaderStagesInfo;

		graphicsPipelineInfo.pVertexInputState = &vertexInputInfo;
		graphicsPipelineInfo.pInputAssemblyState = &assemblyStateInfo;
		graphicsPipelineInfo.pTessellationState = nullptr;
		graphicsPipelineInfo.pViewportState = &viewportStateInfo;
		graphicsPipelineInfo.pRasterizationState = &rasterizationStateInfo;
		graphicsPipelineInfo.pMultisampleState = &multisamplingStateInfo;
		graphicsPipelineInfo.pDepthStencilState = nullptr;
		graphicsPipelineInfo.pColorBlendState = &colourBlendStateInfo;
		graphicsPipelineInfo.pDynamicState = nullptr;

		graphicsPipelineInfo.layout = aPipelineLayout;
		graphicsPipelineInfo.renderPass = aRenderPass;
		graphicsPipelineInfo.subpass = 0;
	}
	
	VkPipeline graphicsPipeline = VK_NULL_HANDLE;
	if (auto const res = vkCreateGraphicsPipelines(aContext.device, VK_NULL_HANDLE, 1, &graphicsPipelineInfo, nullptr, &graphicsPipeline); res != VK_SUCCESS)
	{
		throw lut::Error("Unable to Create Graphics Pipeline\n"
			"vkCreateGraphicsPipeline() returned %s", lut::to_string(res).c_str());
	}
	
	return lut::Pipeline(aContext.device, graphicsPipeline);
}

std::tuple<Image, lut::ImageView> create_framebuffer_image( lut::VulkanContext const& aContext )
{
	VkImageCreateInfo imageInfo{}; {
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;

		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.format = cfg::kImageFormat;
		imageInfo.extent = VkExtent3D{cfg::kImageWidth, cfg::kImageHeight, 1};

		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	}

	Image image(aContext.device);
	if (auto const res = vkCreateImage(aContext.device, &imageInfo, nullptr, &image.image); res != VK_SUCCESS)
	{
		throw lut::Error("Unable to Create Image\n"
			"vkCreateImage() returned %s", lut::to_string(res).c_str());
	}

	VkMemoryRequirements memoryRequirements{};
	vkGetImageMemoryRequirements(aContext.device, image.image, &memoryRequirements);

	VkMemoryAllocateInfo memoryAllocateInfo{}; {
		memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;

		memoryAllocateInfo.allocationSize = memoryRequirements.size;
		memoryAllocateInfo.memoryTypeIndex = find_memory_type(aContext, memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	}

	if (auto const res = vkAllocateMemory(aContext.device, &memoryAllocateInfo, nullptr, &image.memory); res != VK_SUCCESS)
	{
		throw lut::Error("Unable to Allocate Memory for Image\n"
			"vkAllocateMemory() Returned %s", lut::to_string(res).c_str());
	}

	vkBindImageMemory(aContext.device, image.image, image.memory, 0);

	VkComponentMapping componentMapping{}; {
		componentMapping.r = VK_COMPONENT_SWIZZLE_IDENTITY;
		componentMapping.g = VK_COMPONENT_SWIZZLE_IDENTITY;
		componentMapping.b = VK_COMPONENT_SWIZZLE_IDENTITY;
		componentMapping.a = VK_COMPONENT_SWIZZLE_IDENTITY;
	}
	VkImageSubresourceRange imageSubresourceRange{}; {
		imageSubresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

		imageSubresourceRange.baseMipLevel = 0;
		imageSubresourceRange.levelCount = 1;

		imageSubresourceRange.baseArrayLayer = 0;
		imageSubresourceRange.layerCount = 1;
	}
	
	VkImageViewCreateInfo imageViewInfo{}; {
		imageViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;

		imageViewInfo.image = image.image;
		imageViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		imageViewInfo.format = cfg::kImageFormat;

		imageViewInfo.components = componentMapping;
		imageViewInfo.subresourceRange = imageSubresourceRange;
	}

	VkImageView imageView = VK_NULL_HANDLE;
	if (auto const res = vkCreateImageView(aContext.device, &imageViewInfo, nullptr, &imageView); res != VK_SUCCESS)
	{
		throw lut::Error("Unable to Create Image View\n"
			"vkCreateImageView() Returned %s", lut::to_string(res).c_str());
	}

	return {
		std::move(image),
		lut::ImageView(aContext.device, imageView)
	};
}
lut::Framebuffer create_framebuffer( lut::VulkanContext const& aContext, VkRenderPass aRenderPass, VkImageView aTargetImageView )
{
	VkImageView imageViewAttachments[1] = {
		aTargetImageView
	};

	VkFramebufferCreateInfo framebufferInfo{}; {
		framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;

		framebufferInfo.flags = 0;
		framebufferInfo.renderPass = aRenderPass;

		framebufferInfo.attachmentCount = 1;
		framebufferInfo.pAttachments = imageViewAttachments;

		framebufferInfo.width = cfg::kImageWidth;
		framebufferInfo.height = cfg::kImageHeight;

		framebufferInfo.layers = 1;
	}

	VkFramebuffer framebuffer = VK_NULL_HANDLE;
	if (auto const res = vkCreateFramebuffer(aContext.device, &framebufferInfo, nullptr, &framebuffer); res != VK_SUCCESS)
	{
		throw lut::Error("Unable to Create Framebuffer\n"
			"vkCreateFramebuffer() Returned %s", lut::to_string(res).c_str());
	}

	return lut::Framebuffer(aContext.device, framebuffer);
}

Buffer create_download_buffer( lut::VulkanContext const& aContext )
{
	VkBufferCreateInfo downloadBufferInfo{}; {
		downloadBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;

		downloadBufferInfo.size = cfg::kImageSize;

		downloadBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		downloadBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	}

	Buffer downloadBuffer(aContext.device);
	if (auto const res = vkCreateBuffer(aContext.device, &downloadBufferInfo, nullptr, &downloadBuffer.buffer); res != VK_SUCCESS)
	{
		throw lut::Error("Unable to Create Download Buffer\n"
			"vkCreateBuffer() Returned %s", lut::to_string(res).c_str());
	}

	VkMemoryRequirements memoryRequirements{};
	vkGetBufferMemoryRequirements(aContext.device, downloadBuffer.buffer, &memoryRequirements);

	VkMemoryAllocateInfo memoryAllocateInfo{}; {
		memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;

		memoryAllocateInfo.allocationSize = memoryRequirements.size;
		memoryAllocateInfo.memoryTypeIndex = find_memory_type(aContext, memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
	}

	if (auto const res = vkAllocateMemory(aContext.device, &memoryAllocateInfo, nullptr, &downloadBuffer.memory); res != VK_SUCCESS)
	{
		throw lut::Error("Unable to Allocate Memory for Framebuffer\n"
			"vkAllocateMemory() Returned %s", lut::to_string(res).c_str());
	}

	vkBindBufferMemory(aContext.device, downloadBuffer.buffer, downloadBuffer.memory, 0);
	
	return downloadBuffer;
}

void record_commands( VkCommandBuffer aCmdBuff, VkRenderPass aRenderPass, VkFramebuffer aFramebuffer, VkPipeline aGraphicsPipe, VkImage aFbImage, VkBuffer aDownloadBuffer )
{
	// Begin Recording Commands
	VkCommandBufferBeginInfo commandBufferBeginInfo{}; {
		commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

		commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		commandBufferBeginInfo.pInheritanceInfo = nullptr;
	}

	if (auto const res = vkBeginCommandBuffer(aCmdBuff, &commandBufferBeginInfo); res != VK_SUCCESS)
	{
		throw lut::Error("Unable to Begin Recording Command Buffer\n"
			"vkBeginCommandBuffer() Returned %s", lut::to_string(res).c_str());
	}

	// Begin the Render Pass
	VkClearValue clearValues[1]{}; {
		clearValues[0].color.float32[0] = 0.1f;
		clearValues[0].color.float32[1] = 0.1f;
		clearValues[0].color.float32[2] = 0.1f;
		clearValues[0].color.float32[3] = 1.0f;
	}
	
	VkRenderPassBeginInfo renderPassInfo{}; {
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;

		renderPassInfo.renderPass = aRenderPass;
		renderPassInfo.framebuffer = aFramebuffer;

		renderPassInfo.renderArea.offset = VkOffset2D{0, 0};
		renderPassInfo.renderArea.extent = VkExtent2D{cfg::kImageWidth, cfg::kImageHeight};

		renderPassInfo.clearValueCount = 1;
		renderPassInfo.pClearValues = clearValues;
	}

	vkCmdBeginRenderPass(aCmdBuff, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

	// Begin Drawing with our Graphics Pipeline
	vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsPipe);

	// Draw a Triangle
	//vkCmdDraw(aCmdBuff, 3, 1, 0, 0);
	//vkCmdDraw(aCmdBuff, 3, 1, 3, 1);
	vkCmdDraw(aCmdBuff, 6, 2, 0, 0);

	// End the Render Pass
	vkCmdEndRenderPass(aCmdBuff);

	// Copy Image to our Download Buffer
	VkImageSubresourceLayers imageSubresourceLayers{}; {
		imageSubresourceLayers.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

		imageSubresourceLayers.mipLevel = 0;
		imageSubresourceLayers.baseArrayLayer = 0;
		imageSubresourceLayers.layerCount = 1;
	}

	VkBufferImageCopy imageCopy{}; {
		imageCopy.bufferOffset = 0;
		imageCopy.bufferRowLength = 0;
		imageCopy.bufferImageHeight = 0;

		imageCopy.imageSubresource = imageSubresourceLayers;
		imageCopy.imageOffset = VkOffset3D{0, 0, 0};
		imageCopy.imageExtent = VkExtent3D{cfg::kImageWidth, cfg::kImageHeight, 1};
	}

	vkCmdCopyImageToBuffer(aCmdBuff, aFbImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, aDownloadBuffer, 1, &imageCopy);

	// End Command Recording
	if (auto const res = vkEndCommandBuffer(aCmdBuff); res != VK_SUCCESS)
	{
		throw lut::Error("Unable to End Recording Command Buffer\n"
			"vkEndCommandBuffer() Returned %s", lut::to_string(res).c_str());
	}
}
void submit_commands( lut::VulkanContext const& aContext, VkCommandBuffer aCmdBuff, VkFence aFence )
{
	VkSubmitInfo submitInfo{}; {
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &aCmdBuff;
	}

	if (auto const res = vkQueueSubmit(aContext.graphicsQueue, 1, &submitInfo, aFence); res != VK_SUCCESS)
	{
		throw lut::Error("Unable to Submit Command Buffer to Queue\n"
			"vkQueueSubmit() Returned %s", lut::to_string(res).c_str());
	}
}
}

namespace
{
std::uint32_t find_memory_type( lut::VulkanContext const& aContext, std::uint32_t aMemoryTypeBits, VkMemoryPropertyFlags aProps )
{
	VkPhysicalDeviceMemoryProperties memoryProperties{};
	vkGetPhysicalDeviceMemoryProperties(aContext.physicalDevice, &memoryProperties);

	for (std::uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++)
	{
		auto const& memoryType = memoryProperties.memoryTypes[i];

		if (aProps == (aProps & memoryType.propertyFlags) && (aMemoryTypeBits & (1u << i)))
		{
			return i;
		}
	}

	throw lut::Error("Unable to find Suitable Memory Type (Allowed Memory Types = 0x%x, Required Properties = %s)",
		aMemoryTypeBits, lut::memory_property_flags(aProps).c_str());
}
}

//EOF vim:syntax=cpp:foldmethod=marker:ts=4:noexpandtab: 
